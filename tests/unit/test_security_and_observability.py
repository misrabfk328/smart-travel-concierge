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

"""Unit tests for PII redaction, JSON logging, Secret Manager, compaction, and routing."""

from __future__ import annotations

import json
import logging
from io import StringIO
from unittest.mock import MagicMock

import pytest
from google.genai import types

from app.memory.async_tasks import extract_sanitized_preferences
from app.memory.compaction import (
    estimate_content_tokens,
    sliding_window_compact_contents,
)
from app.observability.logging import PiiScrubbingJsonFormatter
from app.observability.tracing import (
    clear_recorded_spans,
    get_recorded_spans,
    trace_span,
)
from app.routing import ModelRouter, TaskTier
from app.security.pii import PiiRedactor
from app.security.secrets import SecretManagerConfig


def test_pii_redactor_scrubs_sensitive_values() -> None:
    """PiiRedactor removes emails, SSNs, cards, phones, passports, and keys."""
    redactor = PiiRedactor(use_cloud_dlp=False)
    sample = {
        "notes": (
            "Email me at traveler@example.com, SSN 123-45-6789, "
            "passport A1234567, phone 415-555-0199, card 4111-2222-3333-4444."
        ),
        "api_key": "AIzaSyDummyExampleSecretKeyValue1234567890",
    }
    scrubbed = redactor.redact_value(sample)
    assert "traveler@example.com" not in scrubbed["notes"]
    assert "123-45-6789" not in scrubbed["notes"]
    assert "A1234567" not in scrubbed["notes"]
    assert "415-555-0199" not in scrubbed["notes"]
    assert "4111-2222-3333-4444" not in scrubbed["notes"]
    assert scrubbed["api_key"] == "[REDACTED_API_KEY]"


def test_structured_json_logging_includes_trace_ids_and_scrubs_pii() -> None:
    """JSON logs include OTel trace_id/span_id and scrub sensitive strings."""
    clear_recorded_spans()
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        PiiScrubbingJsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    )
    logger = logging.getLogger("test.structured.json")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    with trace_span("unit.test.span", {"email": "secret@example.com"}):
        logger.info(
            "Testing traveler@example.com",
            extra={"event_type": "UNIT_CHECK", "ssn": "123-45-6789"},
        )

    payload = json.loads(stream.getvalue().strip())
    assert payload["event_type"] == "UNIT_CHECK"
    assert "trace_id" in payload and len(payload["trace_id"]) == 32
    assert "span_id" in payload and len(payload["span_id"]) == 16
    assert "traveler@example.com" not in payload["message"]
    assert payload["ssn"] == "[REDACTED_SSN]"

    spans = get_recorded_spans()
    assert any(s.name == "unit.test.span" for s in spans)
    unit_span = next(s for s in spans if s.name == "unit.test.span")
    assert unit_span.attributes["email"] == "[REDACTED_EMAIL]"


def test_secret_manager_config_env_and_client_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SecretManagerConfig resolves env vars first and Secret Manager when enabled."""
    monkeypatch.setenv("TRAVEL_API_KEY", "env-secret-val")
    cfg = SecretManagerConfig(project_id="demo-project")
    assert (
        cfg.get_secret("travel-api-key", env_var="TRAVEL_API_KEY") == "env-secret-val"
    )

    monkeypatch.delenv("TRAVEL_API_KEY")
    monkeypatch.setenv("USE_SECRET_MANAGER", "true")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.payload.data = b"sm-secret-value"
    mock_client.access_secret_version.return_value = mock_response
    cfg._client = mock_client

    val = cfg.get_secret("travel-provider-api-key")
    assert val == "sm-secret-value"
    mock_client.access_secret_version.assert_called_once()


def test_sliding_window_compaction_preserves_head_and_tail() -> None:
    """Context compaction shrinks oversized histories while keeping pairs intact."""
    contents = [
        types.Content(
            role="user", parts=[types.Part.from_text(text="Plan Tokyo trip")]
        ),
        *[
            types.Content(
                role="model" if i % 2 else "user",
                parts=[types.Part.from_text(text=f"Turn {i}: " + ("detail " * 80))],
            )
            for i in range(1, 16)
        ],
    ]
    assert estimate_content_tokens(contents) > 1000
    compacted = sliding_window_compact_contents(
        contents, max_tokens=500, keep_recent_turns=4
    )
    assert len(compacted) == 6  # head + summary note + 4 recent turns
    assert "Plan Tokyo trip" in (compacted[0].parts[0].text or "")
    assert "Compacted conversation summary" in (compacted[1].parts[0].text or "")


def test_strategic_model_routing_selects_distinct_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModelRouter routes fast tasks to Flash and complex planning to Pro."""
    monkeypatch.setenv("FAST_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("PLANNING_MODEL", "gemini-2.5-pro")

    assert ModelRouter.resolve_model_name(TaskTier.FAST_RETRIEVAL) == "gemini-3.8-flash"
    assert ModelRouter.resolve_model_name(TaskTier.SUMMARIZATION) == "gemini-3.8-flash"
    assert ModelRouter.resolve_model_name(TaskTier.PLANNING) == "gemini-2.5-pro"
    assert ModelRouter.resolve_model_name(TaskTier.COORDINATION) == "gemini-2.5-pro"


def test_extract_sanitized_preferences() -> None:
    """Async preference extractor pulls structured preferences without leaking PII."""
    prefs = extract_sanitized_preferences(
        "I am vegetarian, prefer an aisle seat, flying out of SFO. Email alex@example.com."
    )
    assert ("dietary", "vegetarian") in prefs
    assert ("seating", "aisle seat") in prefs
    assert ("home_airport", "SFO") in prefs
