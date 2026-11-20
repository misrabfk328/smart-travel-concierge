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

"""ADK plugin capturing structured Intent vs. Outcome audit events and OTel spans."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from opentelemetry import trace
from typing_extensions import override

from app.observability.tracing import configure_tracing
from app.security.pii import PiiRedactor

_LOGGER = logging.getLogger("app.audit")
_AUDIT_RECORDS: list[dict[str, Any]] = []
_MAX_AUDIT_RECORDS = 500


def get_audit_records() -> list[dict[str, Any]]:
    """Return captured Intent vs. Outcome audit records."""
    return list(_AUDIT_RECORDS)


def clear_audit_records() -> None:
    """Clear audit records (for test isolation)."""
    _AUDIT_RECORDS.clear()


def _record_audit(event: dict[str, Any]) -> None:
    _AUDIT_RECORDS.append(event)
    if len(_AUDIT_RECORDS) > _MAX_AUDIT_RECORDS:
        del _AUDIT_RECORDS[:-_MAX_AUDIT_RECORDS]


class IntentOutcomeAuditPlugin(BasePlugin):
    """Records paired INTENT (before) and OUTCOME (after/error) audit trails."""

    def __init__(self, *, redactor: PiiRedactor | None = None) -> None:
        super().__init__(name="intent_outcome_audit_plugin")
        self._redactor = redactor or PiiRedactor()
        self._tracer = configure_tracing()
        self._in_flight: dict[str, tuple[str, float, Any]] = {}

    @override
    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        span = self._tracer.start_span(
            "concierge.invocation",
            attributes={
                "session.id": str(getattr(invocation_context.session, "id", "")),
                "user.id": str(getattr(invocation_context.session, "user_id", "")),
                "agent.name": str(getattr(invocation_context.agent, "name", "")),
            },
        )
        run_key = f"run:{invocation_context.invocation_id}"
        self._in_flight[run_key] = (uuid.uuid4().hex[:12], time.monotonic(), span)
        return None

    @override
    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        run_key = f"run:{invocation_context.invocation_id}"
        entry = self._in_flight.pop(run_key, None)
        if entry is not None:
            _, _, span = entry
            span.end()
        return None

    @override
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        intent_id = uuid.uuid4().hex[:12]
        model_name = getattr(llm_request, "model", "unknown")
        self._in_flight[f"model:{callback_context.agent_name}"] = (
            intent_id,
            time.monotonic(),
            None,
        )
        payload = {
            "event_type": "MODEL_INTENT",
            "intent_id": intent_id,
            "agent_name": callback_context.agent_name,
            "model": str(model_name),
            "contents_count": len(llm_request.contents or []),
        }
        _record_audit(payload)
        _LOGGER.info("Model intent recorded", extra=payload)
        return None

    @override
    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        entry = self._in_flight.pop(f"model:{callback_context.agent_name}", None)
        intent_id = entry[0] if entry else uuid.uuid4().hex[:12]
        duration_ms = round((time.monotonic() - entry[1]) * 1000, 2) if entry else 0.0
        parts = getattr(getattr(llm_response, "content", None), "parts", None) or []
        function_calls = [
            p.function_call.name for p in parts if getattr(p, "function_call", None)
        ]
        payload = {
            "event_type": "MODEL_OUTCOME",
            "intent_id": intent_id,
            "agent_name": callback_context.agent_name,
            "duration_ms": duration_ms,
            "requested_tools": function_calls,
            "has_text": any(bool(getattr(p, "text", None)) for p in parts),
        }
        _record_audit(payload)
        _LOGGER.info("Model outcome recorded", extra=payload)
        return None

    @override
    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        intent_id = uuid.uuid4().hex[:12]
        call_key = f"tool:{tool_context.function_call_id or tool.name}"
        span = self._tracer.start_span(
            f"concierge.tool.{tool.name}",
            attributes={
                "tool.name": tool.name,
                "agent.name": tool_context.agent_name,
                "intent.id": intent_id,
            },
        )
        self._in_flight[call_key] = (intent_id, time.monotonic(), span)
        sanitized_args = self._redactor.redact_value(tool_args)
        payload = {
            "event_type": "TOOL_INTENT",
            "intent_id": intent_id,
            "agent_name": tool_context.agent_name,
            "tool_name": tool.name,
            "intended_args": sanitized_args,
        }
        _record_audit(payload)
        _LOGGER.info("Tool intent recorded", extra=payload)
        return None

    @override
    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        call_key = f"tool:{tool_context.function_call_id or tool.name}"
        entry = self._in_flight.pop(call_key, None)
        intent_id = entry[0] if entry else uuid.uuid4().hex[:12]
        duration_ms = round((time.monotonic() - entry[1]) * 1000, 2) if entry else 0.0
        span = entry[2] if entry else None

        sanitized_result = self._redactor.redact_value(result)
        outcome_status = (
            sanitized_result.get("status", "success")
            if isinstance(sanitized_result, dict)
            else "success"
        )
        if isinstance(sanitized_result, dict) and "error" in sanitized_result:
            outcome_status = "error"

        if span is not None:
            span.set_attribute("outcome.status", str(outcome_status))
            span.set_attribute("duration_ms", duration_ms)
            span.end()

        payload = {
            "event_type": "TOOL_OUTCOME",
            "intent_id": intent_id,
            "agent_name": tool_context.agent_name,
            "tool_name": tool.name,
            "outcome_status": outcome_status,
            "duration_ms": duration_ms,
            "actual_outcome": sanitized_result,
        }
        _record_audit(payload)
        _LOGGER.info("Tool outcome recorded", extra=payload)
        return None

    @override
    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        call_key = f"tool:{tool_context.function_call_id or tool.name}"
        entry = self._in_flight.pop(call_key, None)
        intent_id = entry[0] if entry else uuid.uuid4().hex[:12]
        duration_ms = round((time.monotonic() - entry[1]) * 1000, 2) if entry else 0.0
        span = entry[2] if entry else None

        if span is not None:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()

        payload = {
            "event_type": "TOOL_OUTCOME",
            "intent_id": intent_id,
            "agent_name": tool_context.agent_name,
            "tool_name": tool.name,
            "outcome_status": "exception",
            "duration_ms": duration_ms,
            "actual_outcome": self._redactor.redact_text(str(error)),
        }
        _record_audit(payload)
        _LOGGER.error("Tool exception recorded", extra=payload)
        return None
