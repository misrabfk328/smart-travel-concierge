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

"""Unit tests for tool schemas, guided error recovery, booking HITL, and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import FlightHotelSearchInput, GuidedErrorResponse
from app.tools import (
    booking_confirmation_tool,
    calculate_trip_budget_breakdown,
    check_destination_travel_advisories,
    confirm_and_book_itinerary,
    requires_booking_confirmation,
    retrieve_traveler_preferences,
    save_traveler_preference,
    search_verified_flights_and_hotels,
)


def test_search_verified_flights_and_hotels_happy_path() -> None:
    """Valid search returns structured flights and hotels within nightly cap."""
    result = search_verified_flights_and_hotels(
        origin_airport="sfo",
        destination_city="Tokyo",
        departure_date="2026-10-10",
        return_date="2026-10-15",
        travelers=1,
        cabin_class="economy",
        max_hotel_nightly_usd=180.0,
    )
    assert result["status"] == "success"
    assert result["origin_airport"] == "SFO"
    assert result["nights"] == 5
    assert len(result["flights"]) >= 1
    assert all(h["nightly_rate_usd"] <= 180.0 for h in result["hotels"])


def test_search_guided_error_on_inverted_dates() -> None:
    """Invalid return_date triggers GuidedErrorResponse instead of crashing."""
    result = search_verified_flights_and_hotels(
        origin_airport="SFO",
        destination_city="Tokyo",
        departure_date="2026-10-15",
        return_date="2026-10-10",
    )
    guided = GuidedErrorResponse.model_validate(result)
    assert guided.status == "error"
    assert guided.error_code == "INVALID_TOOL_ARGUMENTS"
    assert "return_date" in guided.message
    assert len(guided.suggested_fixes) >= 1


def test_search_guided_error_when_nightly_cap_too_low() -> None:
    """Overly strict hotel nightly cap returns actionable recovery instructions."""
    result = search_verified_flights_and_hotels(
        origin_airport="SFO",
        destination_city="Tokyo",
        departure_date="2026-10-10",
        return_date="2026-10-15",
        max_hotel_nightly_usd=25.0,
    )
    guided = GuidedErrorResponse.model_validate(result)
    assert guided.error_code == "NO_HOTELS_UNDER_NIGHTLY_CAP"
    assert "max_hotel_nightly_usd" in guided.recovery_instructions


def test_strict_schema_rejects_extra_fields() -> None:
    """Strict Pydantic models reject hallucinated extra keyword fields."""
    with pytest.raises(ValidationError):
        FlightHotelSearchInput.model_validate(
            {
                "origin_airport": "SFO",
                "destination_city": "Tokyo",
                "departure_date": "2026-10-10",
                "return_date": "2026-10-15",
                "unexpected_param": "bad",
            }
        )


def test_check_destination_travel_advisories() -> None:
    """Advisories tool returns structured safety, visa, weather, and daily spend."""
    result = check_destination_travel_advisories(
        destination_city="Tokyo",
        travel_month="October",
    )
    assert result["status"] == "success"
    assert result["advisory_level"] == "level_1_normal"
    assert "Suica" in " ".join(result["local_transit_tips"])
    assert result["suggested_daily_meal_transit_usd"] > 0


def test_calculate_trip_budget_breakdown_within_and_over_budget() -> None:
    """Budget calculator deterministically flags affordable vs over-budget trips."""
    ok = calculate_trip_budget_breakdown(
        flight_total_usd=780.0,
        hotel_total_usd=660.0,
        nights=5,
        max_budget_usd=2400.0,
        travelers=1,
        daily_meals_and_transit_usd_per_person=75.0,
        contingency_percent=10.0,
    )
    assert ok["status"] == "success"
    assert ok["within_budget"] is True
    assert ok["grand_total_usd"] == pytest.approx(2079.0)
    assert ok["remaining_budget_usd"] == pytest.approx(321.0)

    over = calculate_trip_budget_breakdown(
        flight_total_usd=1800.0,
        hotel_total_usd=1000.0,
        nights=5,
        max_budget_usd=2400.0,
    )
    assert over["within_budget"] is False
    assert over["cost_saving_recommendation"] is not None


def test_preference_persistence_and_pii_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preferences persist across store queries while scrubbing PII."""
    monkeypatch.setenv("TRAVEL_DB_PATH", str(tmp_path / "prefs.db"))

    saved = save_traveler_preference(
        category="dietary",
        preference_value="vegetarian; contact alex@example.com or passport P98765432",
        user_id="traveler_42",
    )
    assert saved["status"] == "success"
    stored_val = saved["saved_preference"]["preference_value"]
    assert "vegetarian" in stored_val
    assert "alex@example.com" not in stored_val
    assert "P98765432" not in stored_val
    assert "[REDACTED_EMAIL]" in stored_val
    assert "[REDACTED_PASSPORT]" in stored_val

    recalled = retrieve_traveler_preferences(
        query="vegetarian meals",
        user_id="traveler_42",
        top_k=3,
    )
    assert recalled["status"] == "success"
    assert recalled["count"] == 1
    assert "vegetarian" in recalled["preferences"][0]["preference_value"]


def test_booking_confirmation_gate_and_idempotency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bookings require HITL confirmation for positive charges and are idempotent."""
    monkeypatch.setenv("TRAVEL_DB_PATH", str(tmp_path / "bookings.db"))

    assert requires_booking_confirmation(total_amount_usd=1500.0) is True
    assert (
        requires_booking_confirmation(total_amount_usd=0.0, non_refundable=True) is True
    )
    assert (
        requires_booking_confirmation(total_amount_usd=0.0, non_refundable=False)
        is False
    )
    assert booking_confirmation_tool.name == "confirm_and_book_itinerary"

    first = confirm_and_book_itinerary(
        destination_city="Tokyo",
        flight_id="NH-107",
        hotel_id="HTL-TYO-02",
        departure_date="2026-10-10",
        return_date="2026-10-15",
        total_amount_usd=2079.0,
        idempotency_key="trip-tokyo-oct-001",
        user_id="traveler_42",
    )
    second = confirm_and_book_itinerary(
        destination_city="Tokyo",
        flight_id="NH-107",
        hotel_id="HTL-TYO-02",
        departure_date="2026-10-10",
        return_date="2026-10-15",
        total_amount_usd=2079.0,
        idempotency_key="trip-tokyo-oct-001",
        user_id="traveler_42",
    )
    assert first["status"] == "confirmed"
    assert first["confirmation_code"] == second["confirmation_code"]
