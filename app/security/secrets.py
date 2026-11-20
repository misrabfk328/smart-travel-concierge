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

"""Secure secret loading backed by Google Cloud Secret Manager."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

_LOGGER = logging.getLogger(__name__)


class SecretManagerConfig:
    """Fetches secrets from Google Cloud Secret Manager with local env fallback.

    In production, secrets are injected via Agent Runtime `--secrets` or fetched
    on-demand via `SecretManagerServiceClient.access_secret_version`. In local
    development without GCP credentials, it cleanly falls back to environment
    variables so no API keys are ever hardcoded in source code.
    """

    def __init__(self, project_id: str | None = None) -> None:
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google.cloud import secretmanager

            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def get_secret(
        self,
        secret_id: str,
        *,
        env_var: str | None = None,
        version: str = "latest",
        default: str | None = None,
    ) -> str | None:
        """Resolve a secret from environment injection first, then Secret Manager."""
        lookup_env = env_var or secret_id.upper().replace("-", "_")
        if env_value := os.environ.get(lookup_env):
            return env_value

        use_sm = os.environ.get("USE_SECRET_MANAGER", "").lower() in (
            "true",
            "1",
            "yes",
        )
        if not (use_sm and self._project_id):
            return default

        secret_name = (
            f"projects/{self._project_id}/secrets/{secret_id}/versions/{version}"
        )
        try:
            response = self._get_client().access_secret_version(
                request={"name": secret_name}
            )
            return response.payload.data.decode("utf-8")
        except Exception as exc:
            _LOGGER.warning(
                "Failed to fetch secret %s from Secret Manager: %s",
                secret_id,
                exc,
            )
            return default


@lru_cache(maxsize=32)
def get_secret(
    secret_id: str,
    *,
    env_var: str | None = None,
    default: str | None = None,
) -> str | None:
    """Cached secret accessor using SecretManagerConfig."""
    return SecretManagerConfig().get_secret(secret_id, env_var=env_var, default=default)
