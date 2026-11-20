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

# Secret Manager containers for external travel API keys and credentials.
# Secret versions are populated out-of-band (never stored in Terraform state or code).

resource "google_secret_manager_secret" "travel_provider_api_key" {
  project   = var.project_id
  secret_id = "${var.project_name}-travel-provider-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret_iam_member" "agent_travel_secret_access" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.travel_provider_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "principal://${google_vertex_ai_reasoning_engine.app.spec[0].effective_identity}"
}
