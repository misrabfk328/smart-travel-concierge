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

"""Context compaction helpers complementing ADK EventsCompactionConfig."""

from __future__ import annotations

from google.genai import types


def estimate_content_tokens(contents: list[types.Content]) -> int:
    """Approximate token count (~4 chars/token) across conversation contents."""
    total_chars = 0
    for content in contents:
        for part in content.parts or []:
            if part.text:
                total_chars += len(part.text)
            elif part.function_call:
                total_chars += len(str(part.function_call.name or "")) + len(
                    str(part.function_call.args or "")
                )
            elif part.function_response:
                total_chars += len(str(part.function_response.response or ""))
    return max(1, total_chars // 4) if total_chars else 0


def sliding_window_compact_contents(
    contents: list[types.Content],
    *,
    max_tokens: int = 12000,
    keep_recent_turns: int = 8,
) -> list[types.Content]:
    """Compact oldest conversation history when approaching the token budget.

    Designed for ADK `ContextFilterPlugin(custom_filter=...)` alongside
    `EventsCompactionConfig` (`LlmEventSummarizer`). Preserves the first turn
    (trip goal anchor) and the most recent `keep_recent_turns` items while
    summarizing dropped middle turns into a compact breadcrumb message.
    """
    if len(contents) <= keep_recent_turns + 1:
        return contents
    if estimate_content_tokens(contents) <= max_tokens:
        return contents

    split_idx = len(contents) - keep_recent_turns
    while split_idx > 1 and any(
        getattr(p, "function_response", None) is not None
        for p in (contents[split_idx].parts or [])
    ):
        split_idx -= 1
    if split_idx <= 1:
        return contents

    head = contents[:1]
    dropped = contents[1:split_idx]
    tail = contents[split_idx:]

    summary_snippets: list[str] = []
    for item in dropped:
        role = item.role or "unknown"
        texts = [p.text.strip() for p in (item.parts or []) if p.text]
        if texts:
            snippet = " ".join(texts)[:140]
            summary_snippets.append(f"- [{role}] {snippet}")

    compacted_note = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=(
                    "[Compacted conversation summary of earlier turns]\n"
                    + "\n".join(summary_snippets[-6:])
                )
            )
        ],
    )
    return [*head, compacted_note, *tail]
