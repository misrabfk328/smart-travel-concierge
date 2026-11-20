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

"""Human-in-the-loop gated itinerary booking tool."""

from __future__ import annotations

from datetime import date
from typing import Any

from google.adk.tools import FunctionTool, ToolContext
from pydantic import ValidationError

from app.memory.store import get_persistence_store
from app.schemas import BookingRequestInput, BookingResultOutput, GuidedErrorResponse
from app.tools.travel_tools import validation_error_to_guided_response


def requires_booking_confirmation(
    total_amount_usd: float = 0.0,
    non_refundable: bool = False,
    **kwargs: Any,
) -> bool:
    """Human-in-the-loop gate predicate for ADK FunctionTool confirmation.

    Requires explicit human approval before executing any financial charge or
    non-refundable booking.
    """
    del kwargs
    return float(total_amount_usd) > 0.0 or bool(non_refundable)


def confirm_and_book_itinerary(
    destination_city: str,
    flight_id: str,
    hotel_id: str,
    departure_date: str,
    return_date: str,
    total_amount_usd: float,
    idempotency_key: str,
    non_refundable: bool = False,
    user_id: str = "default_traveler",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Finalize and record a flight + hotel itinerary reservation after human confirmation.

    This is a high-stakes financial action gated by ADK's `require_confirmation`
    human-in-the-loop hook. Idempotent on `idempotency_key` so resuming an
    interrupted session never creates duplicate charges.

    Args:
        destination_city: Destination city for the reservation.
        flight_id: Verified flight identifier (e.g., 'NH-107').
        hotel_id: Verified hotel identifier (e.g., 'HTL-TYO-01').
        departure_date: Outbound date in YYYY-MM-DD format.
        return_date: Return date in YYYY-MM-DD format.
        total_amount_usd: Verified total booking charge in USD.
        idempotency_key: Unique deterministic key to prevent duplicate bookings.
        non_refundable: True if the fare or room rate is non-refundable.
        user_id: Unique traveler identifier.

    Returns:
        Dictionary matching BookingResultOutput with confirmation_code, or
        GuidedErrorResponse if dates/inputs are invalid or unconfirmed.
    """
    resolved_user = user_id
    if tool_context is not None and user_id == "default_traveler":
        resolved_user = str(
            tool_context.state.get("user:id")
            or getattr(
                getattr(tool_context, "_invocation_context", None), "user_id", None
            )
            or user_id
        )

    try:
        params = BookingRequestInput(
            user_id=resolved_user,
            destination_city=destination_city,
            flight_id=flight_id,
            hotel_id=hotel_id,
            departure_date=departure_date,
            return_date=return_date,
            total_amount_usd=total_amount_usd,
            non_refundable=non_refundable,
            idempotency_key=idempotency_key,
        )
    except ValidationError as exc:
        return validation_error_to_guided_response("confirm_and_book_itinerary", exc)

    if date.fromisoformat(params.return_date) <= date.fromisoformat(
        params.departure_date
    ):
        return GuidedErrorResponse(
            error_code="INVALID_BOOKING_DATES",
            message="Return date must be strictly after departure date.",
            recovery_instructions="Correct departure_date and return_date before retrying booking.",
            suggested_fixes=["Verify trip dates with the traveler."],
        ).model_dump(mode="json")

    # Enforce budget ceiling if one was stored in session state.
    if tool_context is not None:
        max_budget = tool_context.state.get("max_budget_usd")
        if max_budget is not None and params.total_amount_usd > float(max_budget):
            return GuidedErrorResponse(
                error_code="BOOKING_EXCEEDS_BUDGET_CAP",
                message=(
                    f"Attempted booking total ${params.total_amount_usd:.2f} exceeds "
                    f"the user's max_budget_usd=${float(max_budget):.2f}."
                ),
                recovery_instructions=(
                    "Do not book over budget. Switch to a cheaper hotel/flight or ask "
                    "the traveler to explicitly raise their budget cap."
                ),
                suggested_fixes=["Re-run itinerary planning with a lower-cost hotel."],
            ).model_dump(mode="json")

    store = get_persistence_store()
    record = store.create_or_get_booking(params.model_dump())

    if tool_context is not None:
        tool_context.state["last_booking_code"] = record["confirmation_code"]

    return BookingResultOutput(
        status="confirmed",
        confirmation_code=record["confirmation_code"],
        total_charged_usd=record["total_amount_usd"],
        summary=(
            f"Booked {params.destination_city} ({params.departure_date} to "
            f"{params.return_date}) with flight {params.flight_id} and hotel "
            f"{params.hotel_id}. Confirmation code: {record['confirmation_code']}."
        ),
    ).model_dump(mode="json")


booking_confirmation_tool = FunctionTool(
    func=confirm_and_book_itinerary,
    require_confirmation=requires_booking_confirmation,
)
