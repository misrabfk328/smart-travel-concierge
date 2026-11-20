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

"""Non-blocking background memory extraction and session consolidation tasks."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext

from app.memory.store import get_persistence_store
from app.observability.tracing import trace_span
from app.security.pii import PiiRedactor

_LOGGER = logging.getLogger(__name__)
_REDACTOR = PiiRedactor()
_PENDING_MEMORY_TASKS: set[asyncio.Task[Any]] = set()

_PREFERENCE_EXTRACTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "dietary",
        re.compile(
            r"\b(vegetarian|vegan|gluten[- ]free|halal|kosher|pescatarian)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "seating",
        re.compile(
            r"\b(aisle seat|window seat|exit row|extra legroom)\b", re.IGNORECASE
        ),
    ),
    (
        "home_airport",
        re.compile(
            r"\b(?:fly(?:ing)?\s+out\s+of|home\s+airport(?:\s+is)?|depart(?:ing)?\s+from)\s+([A-Z]{3})\b"
        ),
    ),
    (
        "lodging",
        re.compile(
            r"\b(quiet room|near metro|transit[- ]accessible|boutique hotel|eco[- ]friendly)\b",
            re.IGNORECASE,
        ),
    ),
)


def extract_sanitized_preferences(text: str) -> list[tuple[str, str]]:
    """Extract durable travel preferences from text after scrubbing PII."""
    scrubbed = _REDACTOR.redact_text(text)
    extracted: list[tuple[str, str]] = []
    for category, pattern in _PREFERENCE_EXTRACTORS:
        for match in pattern.finditer(scrubbed):
            value = match.group(1) if match.lastindex else match.group(0)
            extracted.append((category, value.strip()))
    return extracted


async def _consolidate_session_async(
    *,
    user_id: str,
    session_snapshot: Any,
    memory_service: Any,
    recent_texts: list[str],
) -> None:
    """Background coroutine consolidating preferences and ADK long-term memory."""
    with trace_span("concierge.memory.async_consolidation", {"user_id": user_id}):
        store = get_persistence_store()
        saved_count = 0
        for raw_text in recent_texts:
            for category, pref_value in extract_sanitized_preferences(raw_text):
                store.upsert_preference(
                    user_id=user_id,
                    category=category,
                    preference_value=pref_value,
                )
                saved_count += 1

        if memory_service is not None and session_snapshot is not None:
            try:
                await memory_service.add_session_to_memory(session_snapshot)
            except Exception as exc:
                _LOGGER.warning(
                    "Background add_session_to_memory failed: %s",
                    exc,
                    extra={"event_type": "ASYNC_MEMORY_ERROR", "user_id": user_id},
                )

        _LOGGER.info(
            "Background memory consolidation completed",
            extra={
                "event_type": "ASYNC_MEMORY_CONSOLIDATED",
                "user_id": user_id,
                "preferences_saved": saved_count,
            },
        )


def schedule_async_memory_consolidation(callback_context: CallbackContext) -> None:
    """After-agent callback that dispatches memory consolidation off the main turn."""
    inv_ctx = getattr(callback_context, "_invocation_context", None)
    session = getattr(inv_ctx, "session", None)
    memory_service = getattr(inv_ctx, "memory_service", None)
    if session is None:
        return None

    user_id = str(getattr(session, "user_id", None) or "default_traveler")
    recent_texts: list[str] = []
    sanitized_session = session.model_copy(deep=True)
    for event in getattr(sanitized_session, "events", []) or []:
        content = getattr(event, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            if getattr(part, "text", None):
                part.text = _REDACTOR.redact_text(part.text)
                if getattr(event, "author", "") == "user":
                    recent_texts.append(part.text)

    if not recent_texts and memory_service is None:
        return None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    task = loop.create_task(
        _consolidate_session_async(
            user_id=user_id,
            session_snapshot=sanitized_session,
            memory_service=memory_service,
            recent_texts=recent_texts[-4:],
        )
    )
    _PENDING_MEMORY_TASKS.add(task)
    task.add_done_callback(_PENDING_MEMORY_TASKS.discard)
    return None


async def flush_pending_memory_tasks() -> None:
    """Await all queued background memory tasks (useful in tests and shutdown)."""
    if not _PENDING_MEMORY_TASKS:
        return
    await asyncio.gather(*list(_PENDING_MEMORY_TASKS), return_exceptions=True)
