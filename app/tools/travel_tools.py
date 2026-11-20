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

"""Travel research and deterministic budget calculation tools."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import ValidationError

from app.schemas import (
    BudgetBreakdownInput,
    BudgetBreakdownOutput,
    DestinationAdvisoryInput,
    DestinationAdvisoryOutput,
    FlightHotelSearchInput,
    FlightHotelSearchOutput,
    FlightOption,
    GuidedErrorResponse,
    HotelOption,
)

# Grounded reference inventory for curated destinations, plus dynamic fallback for any city.
_DESTINATION_CATALOG: dict[str, dict[str, Any]] = {
    "tokyo": {
        "airport": "HND",
        "base_flight_usd": 780.0,
        "airlines": [("ANA", "NH-107", True), ("United", "UA-837", True)],
        "hotels": [
            (
                "HTL-TYO-01",
                "Shiba Park Green Hotel",
                "Minato",
                165.0,
                ["vegetarian breakfast", "transit pass desk", "wifi"],
            ),
            (
                "HTL-TYO-02",
                "Asakusa Heritage Boutique",
                "Taito",
                132.0,
                ["vegan cafe nearby", "coin laundry", "wifi"],
            ),
            (
                "HTL-TYO-03",
                "Shinjuku Grand Central",
                "Shinjuku",
                245.0,
                ["airport shuttle", "concierge", "gym"],
            ),
        ],
        "advisory_level": "level_1_normal",
        "visa_guidance": "US/EU/UK passports receive visa-free entry up to 90 days; complete Visit Japan Web QR customs form prior to arrival.",
        "weather_by_month": {
            "october": "Mild autumn temperatures (15-22C / 59-72F) with crisp evenings and light occasional rain.",
            "default": "Four distinct seasons; pack layers and a compact umbrella.",
        },
        "transit_tips": [
            "Load a digital Suica or PASMO IC card in Apple/Google Wallet for trains and buses.",
            "Take the Tokyo Monorail or Keikyu Line from Haneda for sub-$5 airport transfers.",
        ],
        "daily_spend_usd": 75.0,
    },
    "lisbon": {
        "airport": "LIS",
        "base_flight_usd": 640.0,
        "airlines": [("TAP Air Portugal", "TP-236", True), ("Delta", "DL-274", False)],
        "hotels": [
            (
                "HTL-LIS-01",
                "Alfama Terrace Suites",
                "Alfama",
                148.0,
                ["breakfast included", "walkable", "wifi"],
            ),
            (
                "HTL-LIS-02",
                "Principe Real Eco Stay",
                "Principe Real",
                185.0,
                ["plant-based dining nearby", "quiet rooms", "wifi"],
            ),
        ],
        "advisory_level": "level_1_normal",
        "visa_guidance": "Schengen Area rules apply (90 days within 180 days); passport valid 3+ months beyond departure.",
        "weather_by_month": {
            "october": "Warm, sunny shoulder season (16-23C / 61-73F) with low rainfall.",
            "default": "Mild Mediterranean climate; pack comfortable walking shoes for hilly cobblestone streets.",
        },
        "transit_tips": [
            "Use a Viva Viagem rechargeable card for Metro, Tram 28, and Sintra regional trains.",
            "Metro Red Line connects Humberto Delgado Airport directly to central Lisbon in 25 minutes.",
        ],
        "daily_spend_usd": 68.0,
    },
    "paris": {
        "airport": "CDG",
        "base_flight_usd": 690.0,
        "airlines": [("Air France", "AF-083", True), ("JetBlue", "B6-1407", True)],
        "hotels": [
            (
                "HTL-PAR-01",
                "Le Marais Garden Inn",
                "3rd Arrondissement",
                190.0,
                ["metro adjacent", "wifi", "vegetarian bistro nearby"],
            ),
            (
                "HTL-PAR-02",
                "Canal Saint-Martin Boutique",
                "10th Arrondissement",
                152.0,
                ["breakfast", "bike rental", "wifi"],
            ),
        ],
        "advisory_level": "level_1_normal",
        "visa_guidance": "Schengen Area rules apply; keep passport valid at least 3 months past planned return.",
        "weather_by_month": {
            "october": "Cool autumn weather (10-16C / 50-61F); bring a light trench coat and scarf.",
            "default": "Temperate seasonal climate; Metro pass recommended for cross-city transit.",
        },
        "transit_tips": [
            "Use the Navigo Easy card or Bonjour RATP mobile tickets for Metro and RER B.",
            "RER B connects CDG Airport to central Paris in ~35 minutes for ~13 USD.",
        ],
        "daily_spend_usd": 90.0,
    },
    "reykjavik": {
        "airport": "KEF",
        "base_flight_usd": 590.0,
        "airlines": [("Icelandair", "FI-862", True), ("Play", "OG-102", True)],
        "hotels": [
            (
                "HTL-REK-01",
                "Harbor View Nordic Lodge",
                "Vesturbaer",
                175.0,
                ["geothermal heating", "breakfast", "wifi"],
            ),
            (
                "HTL-REK-02",
                "Laugavegur Eco Hostel Suites",
                "Downtown",
                135.0,
                ["shared kitchen", "airport bus stop", "wifi"],
            ),
        ],
        "advisory_level": "level_1_normal",
        "visa_guidance": "Schengen Area rules apply; ensure passport validity of at least 3 months post-departure.",
        "weather_by_month": {
            "october": "Crisp sub-arctic autumn (2-7C / 36-45F) with strong aurora viewing opportunities.",
            "default": "Windproof waterproof outer layer required year-round.",
        },
        "transit_tips": [
            "Book Flybus shuttle from Keflavik Airport (KEF) to BSI terminal in advance.",
            "Klappid app provides single and multi-day Strætó city bus passes.",
        ],
        "daily_spend_usd": 110.0,
    },
}

_CABIN_MULTIPLIER = {
    "economy": 1.0,
    "premium_economy": 1.45,
    "business": 2.6,
}


def validation_error_to_guided_response(
    tool_name: str, exc: ValidationError
) -> dict[str, Any]:
    """Convert Pydantic validation errors into structured LLM recovery instructions."""
    details = "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}"
        for err in exc.errors()
    )
    return GuidedErrorResponse(
        error_code="INVALID_TOOL_ARGUMENTS",
        message=f"Validation failed in `{tool_name}`: {details}",
        recovery_instructions=(
            f"Do not guess missing data or crash. Inspect the invalid fields in `{tool_name}` "
            "and either retry with corrected ISO dates / 3-letter IATA airport codes / positive numbers, "
            "or ask the traveler a concise clarifying question."
        ),
        suggested_fixes=[
            "Use 3-letter uppercase IATA airport codes (e.g., SFO, JFK, HND).",
            "Ensure return_date (YYYY-MM-DD) is strictly later than departure_date (YYYY-MM-DD).",
            "Confirm max_budget_usd and nightly rates are positive numbers.",
        ],
    ).model_dump(mode="json")


def search_verified_flights_and_hotels(
    origin_airport: str,
    destination_city: str,
    departure_date: str,
    return_date: str,
    travelers: int = 1,
    cabin_class: str = "economy",
    max_hotel_nightly_usd: float = 250.0,
) -> dict[str, Any]:
    """Search verified round-trip flights and lodging options within nightly rate limits.

    Validates origin IATA code, trip dates, traveler count, cabin class, and
    maximum nightly hotel cap using strict Pydantic schemas. Returns structured
    recovery instructions if any parameter is malformed or no hotels fit the cap.

    Args:
        origin_airport: 3-letter IATA airport code for departure (e.g., 'SFO').
        destination_city: Destination city name (e.g., 'Tokyo', 'Lisbon').
        departure_date: Outbound travel date in ISO format YYYY-MM-DD.
        return_date: Return travel date in ISO format YYYY-MM-DD (must be after departure).
        travelers: Number of adult travelers (1 to 9).
        cabin_class: Flight cabin class ('economy', 'premium_economy', or 'business').
        max_hotel_nightly_usd: Maximum acceptable hotel nightly rate in USD.

    Returns:
        Dictionary matching FlightHotelSearchOutput on success, or GuidedErrorResponse
        with concrete recovery steps if inputs are invalid or over nightly budget.
    """
    try:
        params = FlightHotelSearchInput(
            origin_airport=origin_airport,
            destination_city=destination_city,
            departure_date=departure_date,
            return_date=return_date,
            travelers=travelers,
            cabin_class=cabin_class,  # type: ignore[arg-type]
            max_hotel_nightly_usd=max_hotel_nightly_usd,
        )
    except ValidationError as exc:
        return validation_error_to_guided_response(
            "search_verified_flights_and_hotels", exc
        )

    dep = date.fromisoformat(params.departure_date)
    ret = date.fromisoformat(params.return_date)
    nights = (ret - dep).days

    city_key = params.destination_city.strip().lower()
    catalog: dict[str, Any] | None = _DESTINATION_CATALOG.get(city_key)
    if catalog is None:
        # Deterministic dynamic fallback for destinations outside the curated catalog.
        dest_code = params.destination_city.strip()[:3].upper()
        catalog = {
            "airport": dest_code,
            "base_flight_usd": 720.0,
            "airlines": [
                ("SkyAlliance", "SA-401", False),
                ("GlobalAir", "GA-210", True),
            ],
            "hotels": [
                (
                    f"HTL-{dest_code}-01",
                    f"{params.destination_city.title()} Central Stay",
                    "City Center",
                    145.0,
                    ["breakfast", "wifi"],
                ),
                (
                    f"HTL-{dest_code}-02",
                    f"{params.destination_city.title()} Boutique Inn",
                    "Old Town",
                    195.0,
                    ["walkable", "wifi"],
                ),
            ],
        }

    multiplier = _CABIN_MULTIPLIER[params.cabin_class]
    flights = [
        FlightOption(
            flight_id=code,
            airline=airline,
            route=f"{params.origin_airport} -> {catalog['airport']} -> {params.origin_airport}",
            cabin_class=params.cabin_class,
            nonstop=nonstop,
            round_trip_usd_per_person=round(catalog["base_flight_usd"] * multiplier, 2),
            total_flight_usd=round(
                catalog["base_flight_usd"] * multiplier * params.travelers, 2
            ),
        )
        for airline, code, nonstop in catalog["airlines"]
    ]

    hotels = [
        HotelOption(
            hotel_id=hotel_id,
            name=name,
            neighborhood=area,
            nightly_rate_usd=rate,
            nights=nights,
            total_hotel_usd=round(rate * nights, 2),
            amenities=amenities,
        )
        for hotel_id, name, area, rate, amenities in catalog["hotels"]
        if rate <= params.max_hotel_nightly_usd
    ]

    if not hotels:
        lowest_rate = min(h[3] for h in catalog["hotels"])
        return GuidedErrorResponse(
            error_code="NO_HOTELS_UNDER_NIGHTLY_CAP",
            message=(
                f"No hotels in {params.destination_city} matched max_hotel_nightly_usd="
                f"${params.max_hotel_nightly_usd:.2f}."
            ),
            recovery_instructions=(
                f"Retry `search_verified_flights_and_hotels` with max_hotel_nightly_usd >= {lowest_rate:.2f}, "
                "or reduce trip duration / switch to an off-peak travel window."
            ),
            suggested_fixes=[
                f"Increase max_hotel_nightly_usd to {lowest_rate:.2f}",
                "Shorten nights or adjust cabin_class to 'economy'",
            ],
        ).model_dump(mode="json")

    return FlightHotelSearchOutput(
        origin_airport=params.origin_airport,
        destination_city=params.destination_city.strip().title(),
        nights=nights,
        flights=flights,
        hotels=hotels,
    ).model_dump(mode="json")


def check_destination_travel_advisories(
    destination_city: str,
    travel_month: str,
) -> dict[str, Any]:
    """Retrieve safety advisories, visa requirements, weather, and transit guidance.

    Args:
        destination_city: Target city name (e.g., 'Tokyo', 'Paris', 'Lisbon').
        travel_month: Planned month or season of travel (e.g., 'October').

    Returns:
        Dictionary matching DestinationAdvisoryOutput with advisory level,
        visa notes, weather expectations, local transit tips, and daily spend estimate.
    """
    try:
        params = DestinationAdvisoryInput(
            destination_city=destination_city,
            travel_month=travel_month,
        )
    except ValidationError as exc:
        return validation_error_to_guided_response(
            "check_destination_travel_advisories", exc
        )

    city_key = params.destination_city.strip().lower()
    month_key = params.travel_month.strip().lower()
    catalog = _DESTINATION_CATALOG.get(city_key)

    if catalog is None:
        return DestinationAdvisoryOutput(
            destination_city=params.destination_city.strip().title(),
            advisory_level="level_1_normal",
            visa_guidance="Verify passport validity (6+ months) and destination entry requirements before booking.",
            weather_summary=f"Typical seasonal variability expected during {params.travel_month}; pack adaptable layers.",
            local_transit_tips=[
                "Prefer official airport rail/bus links or licensed rideshare apps.",
                "Keep digital and offline copies of reservations.",
            ],
            suggested_daily_meal_transit_usd=80.0,
        ).model_dump(mode="json")

    weather_map = catalog["weather_by_month"]
    weather = weather_map.get(month_key, weather_map["default"])
    return DestinationAdvisoryOutput(
        destination_city=params.destination_city.strip().title(),
        advisory_level=catalog["advisory_level"],
        visa_guidance=catalog["visa_guidance"],
        weather_summary=weather,
        local_transit_tips=catalog["transit_tips"],
        suggested_daily_meal_transit_usd=catalog["daily_spend_usd"],
    ).model_dump(mode="json")


def calculate_trip_budget_breakdown(
    flight_total_usd: float,
    hotel_total_usd: float,
    nights: int,
    max_budget_usd: float,
    travelers: int = 1,
    daily_meals_and_transit_usd_per_person: float = 85.0,
    contingency_percent: float = 10.0,
) -> dict[str, Any]:
    """Deterministically calculate total trip cost and verify against hard budget cap.

    Must be used instead of mental arithmetic whenever evaluating whether an
    itinerary fits the traveler's budget ceiling.

    Args:
        flight_total_usd: Total round-trip airfare cost in USD across all travelers.
        hotel_total_usd: Total hotel lodging cost in USD across all nights.
        nights: Number of nights at the destination.
        max_budget_usd: Hard upper budget limit specified by the traveler in USD.
        travelers: Number of travelers included in daily meal/transit calculation.
        daily_meals_and_transit_usd_per_person: Daily food and local transit allowance per person.
        contingency_percent: Percentage buffer (0-30%) added for taxes/incidentals.

    Returns:
        Dictionary matching BudgetBreakdownOutput showing itemized totals,
        remaining budget, boolean `within_budget`, and savings tips if over budget.
    """
    try:
        params = BudgetBreakdownInput(
            flight_total_usd=flight_total_usd,
            hotel_total_usd=hotel_total_usd,
            nights=nights,
            travelers=travelers,
            daily_meals_and_transit_usd_per_person=daily_meals_and_transit_usd_per_person,
            contingency_percent=contingency_percent,
            max_budget_usd=max_budget_usd,
        )
    except ValidationError as exc:
        return validation_error_to_guided_response(
            "calculate_trip_budget_breakdown", exc
        )

    daily_living = round(
        params.daily_meals_and_transit_usd_per_person
        * (params.nights + 1)
        * params.travelers,
        2,
    )
    subtotal = params.flight_total_usd + params.hotel_total_usd + daily_living
    contingency = round(subtotal * (params.contingency_percent / 100.0), 2)
    grand_total = round(subtotal + contingency, 2)
    remaining = round(params.max_budget_usd - grand_total, 2)
    within_budget = grand_total <= params.max_budget_usd

    recommendation: str | None = None
    if not within_budget:
        overage = round(grand_total - params.max_budget_usd, 2)
        recommendation = (
            f"Itinerary exceeds budget by ${overage:.2f}. Select a lower-rate hotel, "
            "shorten the stay by 1 night, or lower cabin class to economy."
        )

    return BudgetBreakdownOutput(
        flight_total_usd=round(params.flight_total_usd, 2),
        hotel_total_usd=round(params.hotel_total_usd, 2),
        daily_living_total_usd=daily_living,
        contingency_buffer_usd=contingency,
        grand_total_usd=grand_total,
        max_budget_usd=round(params.max_budget_usd, 2),
        remaining_budget_usd=remaining,
        within_budget=within_budget,
        cost_saving_recommendation=recommendation,
    ).model_dump(mode="json")
