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

"""Structured JSON logging, distributed tracing, and Intent vs. Outcome audit plugin."""

from .audit import IntentOutcomeAuditPlugin, clear_audit_records, get_audit_records
from .logging import PiiScrubbingJsonFormatter, configure_structured_logging
from .tracing import (
    clear_recorded_spans,
    configure_tracing,
    get_recorded_spans,
    trace_span,
)

__all__ = [
    "IntentOutcomeAuditPlugin",
    "PiiScrubbingJsonFormatter",
    "clear_audit_records",
    "clear_recorded_spans",
    "configure_structured_logging",
    "configure_tracing",
    "get_audit_records",
    "get_recorded_spans",
    "trace_span",
]
