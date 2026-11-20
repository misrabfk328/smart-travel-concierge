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

"""Strict Pydantic v2 input, output, and error schemas for VoyagerAI."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GuidedErrorResponse(BaseModel):
    """Structured error response that instructs the LLM on how to recover."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    error_code: str = Field(description="Machine-readable error category identifier.")
    message: str = Field(
        description="Human-readable summary of why the tool call failed."
    )
    recovery_instructions: str = Field(
        description="Actionable step-by-step instructions for the agent to recover."
    )
    suggested_fixes: list[str] = Field(
        default_factory=list,
        description="Concrete parameter corrections or fallback options.",
    )


class FlightHotelSearchInput(BaseModel):
    """Validated search parameters for flights and lodging inventory."""

    model_config = ConfigDict(extra="forbid")

    origin_airport: str = Field(
        min_length=3,
        max_length=3,
        description="3-letter IATA origin airport code (e.g., 'SFO', 'JFK').",
    )
    destination_city: str = Field(
        min_length=2,
        max_length=80,
        description="Destination city or 3-letter IATA code (e.g., 'Tokyo', 'TYO').",
    )
    departure_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Trip start date in ISO format YYYY-MM-DD.",
    )
    return_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Trip end date in ISO format YYYY-MM-DD.",
    )
    travelers: int = Field(
        default=1,
        ge=1,
        le=9,
        description="Total number of adult travelers (1 to 9).",
    )
    cabin_class: Literal["economy", "premium_economy", "business"] = Field(
        default="economy",
        description="Preferred flight cabin class.",
    )
    max_hotel_nightly_usd: float = Field(
        default=250.0,
        gt=0,
        le=5000.0,
        description="Maximum nightly lodging budget in USD.",
    )

    @field_validator("origin_airport")
    @classmethod
    def validate_iata_uppercase(cls, value: str) -> str:
        """Ensure airport code consists of three alphabetic characters."""
        normalized = value.strip().upper()
        if not normalized.isalpha() or len(normalized) != 3:
            raise ValueError(
                f"origin_airport '{value}' must be a valid 3-letter IATA code such as SFO."
            )
        return normalized

    @model_validator(mode="after")
    def validate_date_order(self) -> FlightHotelSearchInput:
        """Verify return_date is strictly after departure_date."""
        dep = date.fromisoformat(self.departure_date)
        ret = date.fromisoformat(self.return_date)
        if ret <= dep:
            raise ValueError(
                f"return_date ({self.return_date}) must be after departure_date ({self.departure_date})."
            )
        return self


class FlightOption(BaseModel):
    """Structured round-trip flight option."""

    model_config = ConfigDict(extra="forbid")

    flight_id: str
    airline: str
    route: str
    cabin_class: str
    nonstop: bool
    round_trip_usd_per_person: float
    total_flight_usd: float


class HotelOption(BaseModel):
    """Structured hotel lodging option."""

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    name: str
    neighborhood: str
    nightly_rate_usd: float
    nights: int
    total_hotel_usd: float
    amenities: list[str]


class FlightHotelSearchOutput(BaseModel):
    """Validated flight and hotel inventory search results."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    origin_airport: str
    destination_city: str
    nights: int
    flights: list[FlightOption]
    hotels: list[HotelOption]


class DestinationAdvisoryInput(BaseModel):
    """Input schema for checking travel advisories and seasonal guidance."""

    model_config = ConfigDict(extra="forbid")

    destination_city: str = Field(
        min_length=2,
        max_length=80,
        description="Target destination city name (e.g., 'Tokyo', 'Lisbon').",
    )
    travel_month: str = Field(
        min_length=3,
        max_length=20,
        description="Planned travel month or season (e.g., 'October', '2026-10').",
    )


class DestinationAdvisoryOutput(BaseModel):
    """Validated travel safety, visa, weather, and transit advisory output."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    destination_city: str
    advisory_level: Literal["level_1_normal", "level_2_caution", "level_3_reconsider"]
    visa_guidance: str
    weather_summary: str
    local_transit_tips: list[str]
    suggested_daily_meal_transit_usd: float


class BudgetBreakdownInput(BaseModel):
    """Deterministic budget verification request parameters."""

    model_config = ConfigDict(extra="forbid")

    flight_total_usd: float = Field(ge=0, description="Total flight cost in USD.")
    hotel_total_usd: float = Field(ge=0, description="Total lodging cost in USD.")
    nights: int = Field(ge=1, le=60, description="Number of trip nights.")
    travelers: int = Field(default=1, ge=1, le=9, description="Number of travelers.")
    daily_meals_and_transit_usd_per_person: float = Field(
        default=85.0,
        ge=10.0,
        le=2000.0,
        description="Estimated daily food and local transit spend per traveler in USD.",
    )
    contingency_percent: float = Field(
        default=10.0,
        ge=0.0,
        le=30.0,
        description="Percentage contingency buffer added to subtotal.",
    )
    max_budget_usd: float = Field(
        gt=0,
        le=250000.0,
        description="Hard user budget ceiling in USD.",
    )


class BudgetBreakdownOutput(BaseModel):
    """Validated itemized budget calculation and affordability verdict."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    flight_total_usd: float
    hotel_total_usd: float
    daily_living_total_usd: float
    contingency_buffer_usd: float
    grand_total_usd: float
    max_budget_usd: float
    remaining_budget_usd: float
    within_budget: bool
    cost_saving_recommendation: str | None = None


class PreferenceQueryInput(BaseModel):
    """Input schema for retrieving or summarizing traveler preferences."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(
        default="default_traveler",
        min_length=1,
        max_length=120,
        description="Unique traveler identifier.",
    )
    query: str = Field(
        min_length=1,
        max_length=300,
        description="Natural-language query describing the preference context needed.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of matching preferences to return.",
    )


class PreferenceSaveInput(BaseModel):
    """Input schema for storing a durable traveler preference."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(
        default="default_traveler",
        min_length=1,
        max_length=120,
        description="Unique traveler identifier.",
    )
    category: Literal[
        "home_airport",
        "dietary",
        "seating",
        "lodging",
        "pace",
        "budget_style",
        "general",
    ] = Field(description="Structured preference category.")
    preference_value: str = Field(
        min_length=1,
        max_length=300,
        description="Preference statement (PII is automatically scrubbed before saving).",
    )


class PreferenceRecord(BaseModel):
    """Persisted traveler preference item."""

    model_config = ConfigDict(extra="forbid")

    category: str
    preference_value: str
    similarity_score: float
    updated_at: str


class PreferenceSummaryOutput(BaseModel):
    """Structured output returned by preference_memory_agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    user_id: str
    preferences: list[PreferenceRecord] = Field(default_factory=list)
    summary: str


class TripResearchRequest(BaseModel):
    """Input schema for the parallel destination planning workflow."""

    model_config = ConfigDict(extra="forbid")

    origin_airport: str = Field(
        min_length=3,
        max_length=3,
        description="3-letter IATA departure airport code.",
    )
    destination_city: str = Field(
        min_length=2,
        max_length=80,
        description="Target destination city.",
    )
    departure_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Departure date YYYY-MM-DD.",
    )
    return_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Return date YYYY-MM-DD.",
    )
    travelers: int = Field(default=1, ge=1, le=9)
    max_budget_usd: float = Field(gt=0, le=250000.0)
    traveler_preferences: str = Field(
        default="No prior preferences recorded.",
        description="Known dietary, seating, and lodging preferences.",
    )


class ItineraryPlanOutput(BaseModel):
    """Structured output emitted by itinerary_budget_planner."""

    model_config = ConfigDict(extra="forbid")

    destination_city: str
    selected_flight_id: str
    selected_hotel_id: str
    estimated_grand_total_usd: float
    within_budget: bool
    day_by_day_highlights: list[str]
    advisory_notes: str


class BookingRequestInput(BaseModel):
    """Input schema for human-gated itinerary reservation."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="default_traveler", min_length=1, max_length=120)
    destination_city: str = Field(min_length=2, max_length=80)
    flight_id: str = Field(min_length=3, max_length=40)
    hotel_id: str = Field(min_length=3, max_length=40)
    departure_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    return_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    total_amount_usd: float = Field(
        gt=0,
        le=250000.0,
        description="Verified total reservation charge in USD.",
    )
    non_refundable: bool = Field(
        default=False,
        description="Whether the selected fare/hotel rate is non-refundable.",
    )
    idempotency_key: str = Field(
        min_length=4,
        max_length=64,
        description="Deterministic key preventing duplicate bookings on resume.",
    )


class BookingResultOutput(BaseModel):
    """Structured output returned by booking_execution_agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["confirmed", "pending_confirmation", "rejected", "error"]
    confirmation_code: str | None = None
    total_charged_usd: float | None = None
    summary: str
