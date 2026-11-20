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

"""Integration test verifying ADK streaming contract with mocked Gemini transport."""

from unittest.mock import patch

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


async def _fake_generate_content(self, llm_request, stream=False):
    del self, llm_request, stream
    yield LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="I can help plan your trip within budget and remember your travel preferences."
                )
            ],
        ),
        turn_complete=True,
    )


def test_agent_stream() -> None:
    """Integration test verifying valid ADK SSE streaming responses."""
    with patch.object(Gemini, "generate_content_async", _fake_generate_content):
        session_service = InMemorySessionService()
        session = session_service.create_session_sync(
            user_id="test_user", app_name="test"
        )
        runner = Runner(
            agent=root_agent, session_service=session_service, app_name="test"
        )

        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text="What can VoyagerAI help me plan?")],
        )

        events = list(
            runner.run(
                new_message=message,
                user_id="test_user",
                session_id=session.id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

    assert len(events) > 0, "Expected at least one event"
    has_text_content = any(
        event.content
        and event.content.parts
        and any(part.text for part in event.content.parts)
        for event in events
    )
    assert has_text_content, "Expected at least one event with text content"
