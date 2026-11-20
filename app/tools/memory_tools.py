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

"""Cross-session traveler preference tools backed by persistent SQLite + PII redaction."""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext
from pydantic import ValidationError

from app.memory.store import get_persistence_store
from app.schemas import GuidedErrorResponse, PreferenceQueryInput, PreferenceSaveInput
from app.tools.travel_tools import validation_error_to_guided_response


def retrieve_traveler_preferences(
    query: str,
    user_id: str = "default_traveler",
    top_k: int = 5,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Retrieve persisted traveler preferences relevant to a trip planning query.

    Queries the persistent cross-session preference database using semantic token
    similarity and also reads any session-scoped `user:*` preferences.

    Args:
        query: Search query describing needed context (e.g., 'home airport dietary seating').
        user_id: Unique traveler identifier (automatically falls back to session user).
        top_k: Maximum number of matching preference records to return (1 to 20).

    Returns:
        Dictionary containing matching preferences sorted by relevance score, or
        a structured GuidedErrorResponse if arguments fail validation.
    """
    resolved_user = user_id
    if tool_context is not None and user_id == "default_traveler":
        resolved_user = str(
            tool_context.state.get("user:id")
            or getattr(
                getattr(tool_context, "_invocation_context", None), "user_id", None
            )
            or user_id
        )

    try:
        params = PreferenceQueryInput(
            user_id=resolved_user,
            query=query,
            top_k=top_k,
        )
    except ValidationError as exc:
        return validation_error_to_guided_response("retrieve_traveler_preferences", exc)

    store = get_persistence_store()
    records = store.search_preferences(
        user_id=params.user_id,
        query=params.query,
        top_k=params.top_k,
    )
    return {
        "status": "success",
        "user_id": params.user_id,
        "count": len(records),
        "preferences": [rec.model_dump(mode="json") for rec in records],
    }


def save_traveler_preference(
    category: str,
    preference_value: str,
    user_id: str = "default_traveler",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Persist a durable traveler preference across sessions with automatic PII redaction.

    Scrubs sensitive data (emails, phone numbers, passports, payment cards) before
    writing to the SQLite preference database and updates `user:*` session state.

    Args:
        category: Preference bucket ('home_airport', 'dietary', 'seating', 'lodging',
            'pace', 'budget_style', or 'general').
        preference_value: Preference description to persist (e.g., 'vegetarian meals').
        user_id: Unique traveler identifier (defaults to active session user).

    Returns:
        Dictionary confirming the sanitized persisted preference record, or a
        GuidedErrorResponse explaining valid categories and inputs.
    """
    resolved_user = user_id
    if tool_context is not None and user_id == "default_traveler":
        resolved_user = str(
            tool_context.state.get("user:id")
            or getattr(
                getattr(tool_context, "_invocation_context", None), "user_id", None
            )
            or user_id
        )

    try:
        params = PreferenceSaveInput(
            user_id=resolved_user,
            category=category,  # type: ignore[arg-type]
            preference_value=preference_value,
        )
    except ValidationError as exc:
        return validation_error_to_guided_response("save_traveler_preference", exc)

    if not params.preference_value.strip():
        return GuidedErrorResponse(
            error_code="EMPTY_PREFERENCE",
            message="Cannot save an empty preference value.",
            recovery_instructions="Extract a concrete non-empty preference string before saving.",
            suggested_fixes=[
                "Provide a short value such as 'aisle seat' or 'home_airport SFO'."
            ],
        ).model_dump(mode="json")

    store = get_persistence_store()
    record = store.upsert_preference(
        user_id=params.user_id,
        category=params.category,
        preference_value=params.preference_value,
    )
    if tool_context is not None:
        tool_context.state[f"user:pref_{params.category}"] = record.preference_value

    return {
        "status": "success",
        "user_id": params.user_id,
        "saved_preference": record.model_dump(mode="json"),
    }
