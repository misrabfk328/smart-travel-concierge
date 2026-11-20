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

"""VoyagerAI multi-agent coordinator, parallel planning workflow, and ADK App."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import App, EventsCompactionConfig, ResumabilityConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.plugins.auto_tracing_plugin import AutoTracingPlugin
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.workflow import JoinNode, Workflow
from google.genai import types

from app.guardrails import TravelSafetyAndPolicyPlugin
from app.memory.async_tasks import schedule_async_memory_consolidation
from app.memory.compaction import sliding_window_compact_contents
from app.observability import (
    IntentOutcomeAuditPlugin,
    configure_structured_logging,
    configure_tracing,
)
from app.prompts import (
    ADVISORY_WEATHER_RESEARCHER_CONSTITUTION,
    BOOKING_EXECUTION_CONSTITUTION,
    COORDINATOR_CONSTITUTION,
    FLIGHT_HOTEL_RESEARCHER_CONSTITUTION,
    ITINERARY_PLANNER_CONSTITUTION,
    PREFERENCE_MEMORY_CONSTITUTION,
)
from app.routing import DEFAULT_FAST_MODEL, ModelRouter, TaskTier
from app.schemas import (
    BookingRequestInput,
    BookingResultOutput,
    ItineraryPlanOutput,
    PreferenceQueryInput,
    PreferenceSummaryOutput,
    TripResearchRequest,
)
from app.tools import (
    booking_confirmation_tool,
    calculate_trip_budget_breakdown,
    check_destination_travel_advisories,
    retrieve_traveler_preferences,
    save_traveler_preference,
    search_verified_flights_and_hotels,
)

# Preserve explicit MODEL symbol expected by scaffold tooling/conventions.
MODEL = DEFAULT_FAST_MODEL

configure_structured_logging()
configure_tracing()


def create_preference_memory_agent() -> LlmAgent:
    """Create the Flash-tier sub-agent managing persistent user preferences."""
    return LlmAgent(
        name="preference_memory_agent",
        model=ModelRouter.select_model(TaskTier.FAST_RETRIEVAL),
        mode="task",
        input_schema=PreferenceQueryInput,
        output_schema=PreferenceSummaryOutput,
        description=(
            "Retrieves and persists durable traveler preferences (home airport, "
            "dietary rules, seating, lodging style) across sessions."
        ),
        instruction=PREFERENCE_MEMORY_CONSTITUTION,
        tools=[retrieve_traveler_preferences, save_traveler_preference],
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )


def create_destination_planning_workflow() -> Workflow:
    """Create the parallel fan-out / fan-in research and planning ADK Workflow."""
    flight_hotel_researcher = LlmAgent(
        name="flight_hotel_researcher",
        model=ModelRouter.select_model(TaskTier.FAST_RETRIEVAL),
        mode="single_turn",
        description="Searches verified flights and lodging within nightly caps.",
        instruction=FLIGHT_HOTEL_RESEARCHER_CONSTITUTION,
        tools=[search_verified_flights_and_hotels],
        output_key="flight_hotel_options",
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )

    advisory_weather_researcher = LlmAgent(
        name="advisory_weather_researcher",
        model=ModelRouter.select_model(TaskTier.FAST_RETRIEVAL),
        mode="single_turn",
        description="Checks destination travel advisories, visas, and seasonal weather.",
        instruction=ADVISORY_WEATHER_RESEARCHER_CONSTITUTION,
        tools=[check_destination_travel_advisories],
        output_key="advisory_weather_notes",
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )

    merge_research = JoinNode(
        name="merge_destination_research",
        description="Joins concurrent flight/hotel inventory and travel advisory branches.",
    )

    itinerary_budget_planner = LlmAgent(
        name="itinerary_budget_planner",
        model=ModelRouter.select_model(TaskTier.PLANNING),
        mode="single_turn",
        description="Optimizes itinerary selection under hard max_budget_usd cap.",
        instruction=ITINERARY_PLANNER_CONSTITUTION,
        tools=[calculate_trip_budget_breakdown],
        output_schema=ItineraryPlanOutput,
        output_key="planned_itinerary",
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )

    return Workflow(
        name="destination_planning_pipeline",
        description=(
            "Runs parallel flight/hotel research and travel advisory checks, then "
            "synthesizes a budget-verified itinerary plan."
        ),
        input_schema=TripResearchRequest,
        output_schema=ItineraryPlanOutput,
        edges=[
            ("START", (flight_hotel_researcher, advisory_weather_researcher)),
            ((flight_hotel_researcher, advisory_weather_researcher), merge_research),
            (merge_research, itinerary_budget_planner),
        ],
    )


def create_booking_execution_agent() -> LlmAgent:
    """Create the HITL-gated sub-agent responsible for finalizing reservations."""
    return LlmAgent(
        name="booking_execution_agent",
        model=ModelRouter.select_model(TaskTier.BOOKING_FORMAT),
        mode="task",
        input_schema=BookingRequestInput,
        output_schema=BookingResultOutput,
        description=(
            "Finalizes flight and hotel bookings behind a mandatory human-in-the-loop "
            "confirmation gate."
        ),
        instruction=BOOKING_EXECUTION_CONSTITUTION,
        tools=[booking_confirmation_tool],
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )


def build_root_agent() -> LlmAgent:
    """Assemble the top-level VoyagerAI multi-agent coordinator."""
    return LlmAgent(
        # Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml.
        name="smart_travel_concierge",
        model=ModelRouter.select_model(TaskTier.COORDINATION),
        description=(
            "VoyagerAI Smart Travel & Budget Concierge coordinating preferences, "
            "parallel destination research, deterministic budgeting, and gated booking."
        ),
        instruction=COORDINATOR_CONSTITUTION,
        tools=[
            PreloadMemoryTool(),
            save_traveler_preference,
            retrieve_traveler_preferences,
            calculate_trip_budget_breakdown,
            create_destination_planning_workflow(),
        ],
        sub_agents=[
            create_preference_memory_agent(),
            create_booking_execution_agent(),
        ],
        after_agent_callback=schedule_async_memory_consolidation,
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )


def build_app(root: LlmAgent | None = None) -> App:
    """Wrap root_agent in an ADK App with compaction, caching, plugins, and HITL."""
    return App(
        # Must match the agent directory name ('app').
        name="app",
        root_agent=root or build_root_agent(),
        plugins=[
            TravelSafetyAndPolicyPlugin(),
            IntentOutcomeAuditPlugin(),
            ReflectAndRetryToolPlugin(
                max_retries=2,
                throw_exception_if_retry_exceeded=False,
            ),
            ContextFilterPlugin(
                num_invocations_to_keep=10,
                custom_filter=sliding_window_compact_contents,
            ),
            AutoTracingPlugin(),
        ],
        events_compaction_config=EventsCompactionConfig(
            summarizer=LlmEventSummarizer(
                llm=ModelRouter.select_model(TaskTier.SUMMARIZATION)
            ),
            compaction_interval=6,
            overlap_size=2,
        ),
        context_cache_config=ContextCacheConfig(
            min_tokens=2048,
            ttl_seconds=1800,
            cache_intervals=10,
        ),
        resumability_config=ResumabilityConfig(is_resumable=True),
    )


root_agent = build_root_agent()
app = build_app(root_agent)
