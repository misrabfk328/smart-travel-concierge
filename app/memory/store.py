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

"""Persistent SQLite storage for traveler preferences and idempotent bookings."""

from __future__ import annotations

import math
import os
import re
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.schemas import PreferenceRecord
from app.security.pii import PiiRedactor

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall(text.lower()))


def _cosine_similarity(query: str, document: str) -> float:
    """Compute lightweight token cosine similarity for semantic preference recall."""
    q_vec = _tokenize(query)
    d_vec = _tokenize(document)
    if not q_vec or not d_vec:
        return 0.0
    intersection = set(q_vec) & set(d_vec)
    numerator = sum(q_vec[t] * d_vec[t] for t in intersection)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
    d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
    if q_norm == 0.0 or d_norm == 0.0:
        return 0.0
    return round(numerator / (q_norm * d_norm), 4)


def default_db_path() -> Path:
    """Return the configured SQLite database path for persistent state."""
    configured = os.environ.get("TRAVEL_DB_PATH")
    if configured:
        return Path(configured)
    return Path(os.environ.get("TRAVEL_DATA_DIR", "data")) / "travel_concierge.db"


class TravelPersistenceStore:
    """Thread-safe SQLite store for durable preferences and booking records."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        redactor: PiiRedactor | None = None,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else default_db_path()
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._redactor = redactor or PiiRedactor()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traveler_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    preference_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, category, preference_value)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS itinerary_bookings (
                    idempotency_key TEXT PRIMARY KEY,
                    confirmation_code TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    destination_city TEXT NOT NULL,
                    flight_id TEXT NOT NULL,
                    hotel_id TEXT NOT NULL,
                    departure_date TEXT NOT NULL,
                    return_date TEXT NOT NULL,
                    total_amount_usd REAL NOT NULL,
                    non_refundable INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def upsert_preference(
        self,
        *,
        user_id: str,
        category: str,
        preference_value: str,
    ) -> PreferenceRecord:
        """Scrub PII and persist a traveler preference record."""
        clean_value = self._redactor.redact_text(preference_value.strip())
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traveler_preferences (user_id, category, preference_value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, category, preference_value)
                DO UPDATE SET updated_at = excluded.updated_at
                """,
                (user_id, category, clean_value, now),
            )
            conn.commit()
        return PreferenceRecord(
            category=category,
            preference_value=clean_value,
            similarity_score=1.0,
            updated_at=now,
        )

    def search_preferences(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[PreferenceRecord]:
        """Rank stored preferences using semantic token cosine similarity."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT category, preference_value, updated_at
                FROM traveler_preferences
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()

        scored: list[PreferenceRecord] = []
        for row in rows:
            doc = f"{row['category']} {row['preference_value']}"
            score = _cosine_similarity(query, doc)
            # Return even baseline preferences with small floor score so profile recall works.
            effective_score = max(score, 0.15)
            scored.append(
                PreferenceRecord(
                    category=row["category"],
                    preference_value=row["preference_value"],
                    similarity_score=effective_score,
                    updated_at=row["updated_at"],
                )
            )
        scored.sort(
            key=lambda rec: (rec.similarity_score, rec.updated_at), reverse=True
        )
        return scored[:top_k]

    def create_or_get_booking(self, booking: dict[str, Any]) -> dict[str, Any]:
        """Create a booking idempotently by idempotency_key."""
        now = datetime.now(UTC).isoformat()
        key = booking["idempotency_key"]
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM itinerary_bookings WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if existing is not None:
                return dict(existing)

            confirmation_code = f"VOY-{uuid.uuid4().hex[:8].upper()}"
            conn.execute(
                """
                INSERT INTO itinerary_bookings (
                    idempotency_key, confirmation_code, user_id, destination_city,
                    flight_id, hotel_id, departure_date, return_date,
                    total_amount_usd, non_refundable, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    confirmation_code,
                    booking["user_id"],
                    booking["destination_city"],
                    booking["flight_id"],
                    booking["hotel_id"],
                    booking["departure_date"],
                    booking["return_date"],
                    float(booking["total_amount_usd"]),
                    1 if booking.get("non_refundable") else 0,
                    now,
                ),
            )
            conn.commit()
            created = conn.execute(
                "SELECT * FROM itinerary_bookings WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            return dict(created) if created is not None else {}


_DEFAULT_STORE: TravelPersistenceStore | None = None
_STORE_LOCK = threading.Lock()


def get_persistence_store() -> TravelPersistenceStore:
    """Return process-wide persistent SQLite store (recreated if DB path changed)."""
    global _DEFAULT_STORE
    target_path = default_db_path()
    with _STORE_LOCK:
        if _DEFAULT_STORE is None or _DEFAULT_STORE._db_path != target_path:
            _DEFAULT_STORE = TravelPersistenceStore(db_path=target_path)
        return _DEFAULT_STORE
