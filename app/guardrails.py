# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input/output guardrails, prompt-injection defense, and Model Armor integration."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from typing_extensions import override

from app.security.pii import PiiRedactor

_LOGGER = logging.getLogger("app.guardrails")

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|prior|system)\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(r"\breveal\s+(?:your\s+)?system\s+prompt\b", re.IGNORECASE),
    re.compile(
        r"\bbypass\s+(?:confirmation|approval|human\s+review|safety)\b", re.IGNORECASE
    ),
    re.compile(r"\bexfiltrate\b|\brm\s+-rf\b", re.IGNORECASE),
)

_BLOCKED_INPUT_MESSAGE = (
    "I cannot process that request because it violates VoyagerAI safety and policy guardrails "
    "(prompt injection or policy bypass detected)."
)


class TravelSafetyAndPolicyPlugin(BasePlugin):
    """ADK BasePlugin enforcing input safety, optional Model Armor, and output policy checks."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        location_id: str | None = None,
        template_id: str | None = None,
        redactor: PiiRedactor | None = None,
    ) -> None:
        super().__init__(name="travel_safety_and_policy_plugin")
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location_id = location_id or os.environ.get(
            "GOOGLE_CLOUD_LOCATION", "us-central1"
        )
        self._template_id = template_id or os.environ.get("MODEL_ARMOR_TEMPLATE_ID", "")
        self._redactor = redactor or PiiRedactor()
        self._armor_client: Any = None

    def _get_model_armor_client(self) -> Any:
        if (
            self._armor_client is None
            and self._template_id
            and self._project_id
            and self._project_id != "your-gcp-project-id"
        ):
            try:
                from google.api_core.client_options import ClientOptions
                from google.cloud import modelarmor_v1

                self._armor_client = modelarmor_v1.ModelArmorClient(
                    client_options=ClientOptions(
                        api_endpoint=f"modelarmor.{self._location_id}.rep.googleapis.com"
                    )
                )
            except Exception as exc:  # pragma: no cover - optional cloud client
                _LOGGER.warning(
                    "Model Armor unavailable, using local policy rules: %s", exc
                )
        return self._armor_client

    def _is_model_armor_flagged(self, text: str) -> bool:
        client = self._get_model_armor_client()
        if client is None or not text:
            return False
        try:
            from google.cloud import modelarmor_v1

            template_name = (
                f"projects/{self._project_id}/locations/{self._location_id}"
                f"/templates/{self._template_id}"
            )
            response = client.sanitize_user_prompt(
                request=modelarmor_v1.SanitizeUserPromptRequest(
                    name=template_name,
                    user_prompt_data=modelarmor_v1.DataItem(text=text),
                )
            )
            result = getattr(response, "sanitization_result", None)
            match_state = getattr(result, "filter_match_state", None)
            return bool(match_state and "MATCH_FOUND" in str(match_state))
        except Exception as exc:  # pragma: no cover
            _LOGGER.warning("Model Armor check failed open to local policy: %s", exc)
            return False

    def _is_unsafe_input(self, text: str) -> bool:
        if any(pattern.search(text) for pattern in _INJECTION_PATTERNS):
            return True
        return self._is_model_armor_flagged(text)

    @override
    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        parts = user_message.parts or []
        raw_text = "\n".join(
            p.text
            for p in parts
            if isinstance(getattr(p, "text", None), str) and p.text
        )
        if self._is_unsafe_input(raw_text):
            _LOGGER.warning(
                "Blocked unsafe user prompt in guardrail plugin",
                extra={"event_type": "GUARDRAIL_INPUT_BLOCKED"},
            )
            invocation_context.session.state["temp:policy_blocked"] = True
            return types.Content(
                role="user",
                parts=[types.Part.from_text(text=_BLOCKED_INPUT_MESSAGE)],
            )

        # Scrub raw PII from user message text before persisting or prompting if detected.
        if self._redactor.contains_pii(raw_text):
            scrubbed_parts = [
                types.Part.from_text(text=self._redactor.redact_text(p.text))
                if getattr(p, "text", None)
                else p
                for p in parts
            ]
            return types.Content(role=user_message.role or "user", parts=scrubbed_parts)
        return None

    @override
    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        if invocation_context.session.state.get("temp:policy_blocked"):
            invocation_context.session.state["temp:policy_blocked"] = False
            return types.Content(
                role="model",
                parts=[types.Part.from_text(text=_BLOCKED_INPUT_MESSAGE)],
            )
        return None

    @override
    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        # Block prompt-injection strings inside tool arguments before execution.
        for key, value in tool_args.items():
            if isinstance(value, str) and self._is_unsafe_input(value):
                _LOGGER.warning(
                    "Blocked unsafe argument '%s' to tool '%s'",
                    key,
                    tool.name,
                    extra={
                        "event_type": "GUARDRAIL_TOOL_ARG_BLOCKED",
                        "tool_name": tool.name,
                    },
                )
                return {
                    "status": "error",
                    "error_code": "POLICY_VIOLATION",
                    "message": f"Tool argument '{key}' violated safety policy.",
                    "recovery_instructions": "Remove unsafe instructions and retry with clean user inputs.",
                }
        return None

    @override
    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        del callback_context
        content = getattr(llm_response, "content", None)
        if not content or not content.parts:
            return None

        modified = False
        new_parts: list[types.Part] = []
        for part in content.parts:
            text = getattr(part, "text", None)
            if text and self._redactor.contains_pii(text):
                modified = True
                new_parts.append(
                    types.Part.from_text(text=self._redactor.redact_text(text))
                )
            else:
                new_parts.append(part)

        if modified:
            _LOGGER.warning(
                "Scrubbed PII from outgoing model response",
                extra={"event_type": "GUARDRAIL_OUTPUT_PII_REDACTED"},
            )
            return LlmResponse(
                content=types.Content(role=content.role or "model", parts=new_parts)
            )
        return None
