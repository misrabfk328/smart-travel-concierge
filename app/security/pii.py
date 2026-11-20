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

"""Active PII redaction pipeline for logs, audit records, and memory writes.

Uses deterministic regex scrubbing at all times and optionally delegates to
Google Cloud Sensitive Data Protection (Cloud DLP) when enabled.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

_STD_LOGGER = logging.getLogger(__name__)

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "EMAIL_ADDRESS",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "US_SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        "PAYMENT_CARD",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "[REDACTED_CARD]",
    ),
    (
        "PASSPORT_NUMBER",
        re.compile(
            r"\b(?:passport(?:\s*(?:no|num|number|#|:))?\s*[:#]?\s*)([A-Z0-9]{6,12})\b",
            re.IGNORECASE,
        ),
        "passport [REDACTED_PASSPORT]",
    ),
    (
        "PHONE_NUMBER",
        re.compile(
            r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\w)"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        "API_KEY_OR_BEARER",
        re.compile(
            r"\b(?:AIza[0-9A-Za-z\-_]{20,}|sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._\-]{16,})\b"
        ),
        "[REDACTED_SECRET]",
    ),
)

_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "email",
        "phone",
        "phone_number",
        "passport",
        "passport_number",
        "ssn",
        "credit_card",
        "card_number",
        "api_key",
        "token",
        "secret",
        "password",
    }
)


class PiiRedactor:
    """Scrubs sensitive PII and credentials prior to logging or storage."""

    def __init__(
        self,
        *,
        use_cloud_dlp: bool | None = None,
        project_id: str | None = None,
        location: str = "global",
    ) -> None:
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location = location
        if use_cloud_dlp is None:
            use_cloud_dlp = os.environ.get("ENABLE_CLOUD_DLP", "").lower() in (
                "true",
                "1",
                "yes",
            )
        self._use_cloud_dlp = bool(use_cloud_dlp and self._project_id)
        self._dlp_client: Any = None

    def _get_dlp_client(self) -> Any:
        if self._dlp_client is None and self._use_cloud_dlp:
            try:
                from google.cloud import dlp_v2

                self._dlp_client = dlp_v2.DlpServiceClient()
            except Exception as exc:  # pragma: no cover - optional cloud client
                _STD_LOGGER.warning(
                    "Cloud DLP client unavailable, using regex redactor: %s", exc
                )
                self._use_cloud_dlp = False
        return self._dlp_client

    def _redact_with_cloud_dlp(self, text: str) -> str:
        client = self._get_dlp_client()
        if client is None:
            return text
        try:
            parent = f"projects/{self._project_id}/locations/{self._location}"
            inspect_config = {
                "info_types": [
                    {"name": "EMAIL_ADDRESS"},
                    {"name": "PHONE_NUMBER"},
                    {"name": "CREDIT_CARD_NUMBER"},
                    {"name": "PASSPORT"},
                    {"name": "US_SOCIAL_SECURITY_NUMBER"},
                ]
            }
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            }
            response = client.deidentify_content(
                request={
                    "parent": parent,
                    "deidentify_config": deidentify_config,
                    "inspect_config": inspect_config,
                    "item": {"value": text},
                }
            )
            return response.item.value
        except Exception as exc:  # pragma: no cover - fallback guarantees safety
            _STD_LOGGER.warning(
                "Cloud DLP deidentify failed, falling back to regex: %s", exc
            )
            return text

    def redact_text(self, text: str) -> str:
        """Scrub all recognized PII and credential patterns from a string."""
        if not text:
            return text
        scrubbed = self._redact_with_cloud_dlp(text) if self._use_cloud_dlp else text
        for _, pattern, replacement in _PII_PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)
        return scrubbed

    def redact_value(self, value: Any) -> Any:
        """Recursively scrub PII from strings, mappings, and sequences."""
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            redacted: dict[Any, Any] = {}
            for key, item in value.items():
                key_str = str(key).lower()
                if key_str in _SENSITIVE_FIELD_NAMES and item is not None:
                    redacted[key] = f"[REDACTED_{key_str.upper()}]"
                else:
                    redacted[key] = self.redact_value(item)
            return redacted
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_value(item) for item in value)
        return value

    def contains_pii(self, text: str) -> bool:
        """Return True if the text contains any locally detectable PII pattern."""
        if not text:
            return False
        return any(pattern.search(text) for _, pattern, _ in _PII_PATTERNS)


_DEFAULT_REDACTOR = PiiRedactor()


def redact_pii(value: Any) -> Any:
    """Convenience helper using the default process-wide PiiRedactor."""
    return _DEFAULT_REDACTOR.redact_value(value)
