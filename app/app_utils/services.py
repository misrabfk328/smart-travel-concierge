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

"""Process-wide ADK session, memory, and artifact services.

Registered under ``shared://`` so ADK web routes, A2A endpoints, and the
reasoning_engine adapter share persistent session and memory backends.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.adk.cli.service_registry import get_service_registry
from google.adk.cli.utils.service_factory import create_session_service_from_options
from google.adk.memory import InMemoryMemoryService, SqliteMemoryService

SESSION_SERVICE_URI = "shared://session"
ARTIFACT_SERVICE_URI = "shared://artifact"
MEMORY_SERVICE_URI = "shared://memory"

_AGENT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _use_in_memory_services() -> bool:
    return os.environ.get("USE_IN_MEMORY_SESSION", "").lower() in (
        "true",
        "1",
        "yes",
    )


def _data_dir() -> Path:
    path = Path(os.environ.get("TRAVEL_DATA_DIR", os.path.join(_AGENT_DIR, "data")))
    path.mkdir(parents=True, exist_ok=True)
    return path


@functools.cache
def get_session_service():
    """Process-wide persistent session service shared across serving surfaces."""
    if uri := os.environ.get("SESSION_SERVICE_URI"):
        return create_session_service_from_options(
            base_dir=_AGENT_DIR, session_service_uri=uri
        )
    if agent_engine_id := os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
        from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

        return VertexAiSessionService(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION"),
            agent_engine_id=agent_engine_id,
        )
    if _use_in_memory_services():
        from google.adk.sessions.in_memory_session_service import InMemorySessionService

        return InMemorySessionService()

    from google.adk.sessions import DatabaseSessionService

    session_db_path = _data_dir() / "travel_sessions.db"
    return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{session_db_path}")


@functools.cache
def get_memory_service():
    """Process-wide persistent memory service (Vertex AI Memory Bank or SQLite FTS5)."""
    if agent_engine_id := os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
        from google.adk.memory import VertexAiMemoryBankService

        return VertexAiMemoryBankService(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION"),
            agent_engine_id=agent_engine_id,
        )
    if _use_in_memory_services():
        return InMemoryMemoryService()

    memory_db_path = _data_dir() / "travel_memory.db"
    return SqliteMemoryService(db_path=memory_db_path)


@functools.cache
def get_artifact_service():
    """Process-wide artifact service: GCS when a bucket is set, else in-memory."""
    if bucket := os.environ.get("LOGS_BUCKET_NAME"):
        return GcsArtifactService(bucket_name=bucket)
    return InMemoryArtifactService()


_registry = get_service_registry()
_registry.register_session_service("shared", lambda uri, **kw: get_session_service())
_registry.register_artifact_service("shared", lambda uri, **kw: get_artifact_service())
_registry.register_memory_service("shared", lambda uri, **kw: get_memory_service())
