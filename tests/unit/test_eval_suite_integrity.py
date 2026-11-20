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

"""Validate golden evaluation datasets and configuration files."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]


def test_golden_eval_dataset_structure() -> None:
    """Ensure the golden travel dataset contains all expected rubric scenarios."""
    dataset_path = _ROOT / "tests" / "eval" / "datasets" / "travel-golden-dataset.json"
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = data.get("eval_cases", [])
    case_ids = {case["eval_case_id"] for case in cases}

    assert {
        "preference_storage_and_pii_safety",
        "tokyo_budget_itinerary_planning",
        "guided_error_recovery_invalid_dates",
        "hitl_booking_confirmation_gate",
        "prompt_injection_guardrail",
    }.issubset(case_ids)

    for case in cases:
        assert case["prompt"]["role"] == "user"
        assert case["prompt"]["parts"][0]["text"].strip()
        assert case["reference"]["response"]["parts"][0]["text"].strip()


def test_eval_config_yaml_valid() -> None:
    """Ensure eval_config.yaml specifies valid metrics and passing threshold."""
    cfg_path = _ROOT / "tests" / "eval" / "eval_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert "custom_response_quality" in cfg["metrics_to_run"]
    assert cfg["thresholds"]["custom_response_quality"] >= 4.0
