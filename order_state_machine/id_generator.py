"""Intent ID + Order ID generation for the Order State Machine (Phase 3.4).

Implements two patterns captured in session log Architecture Decisions:

1) Deterministic Intent ID Generation (hash-based)
   - Same intent payload -> same intent_id
   - Format: I-{YYYYMMDD}-{hash_prefix}
   - Hash: SHA256(canonicalized intent payload), prefix = first 12 hex chars

2) Order ID Generation (timestamp + counter, restart-safe)
   - Unique per generated order_id/client_order_id
   - Format: O-{YYYYMMDD}-{run_id_short}-{counter}
   - Counter is thread-safe and can be seeded from persisted mapping count
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import UTC, datetime
from typing import Any

import msgspec

from common.interface import Result, ResultStatus
from order_state_machine.interface import IDGeneratorProtocol
from strategy.interface import TradeIntent

__all__ = ["IDGenerator"]


class IDGenerator(IDGeneratorProtocol):
    """Generate deterministic intent IDs and unique order IDs.

    Args:
        initial_counter: Seed for the monotonic counter (restart-safe).
            Recommended: set to the number of persisted intent->order mappings
            on startup; the next generated order id will use (seed + 1).
        run_id_short_len: Length of the run id prefix in generated order ids.
    """

    def __init__(self, *, initial_counter: int = 0, run_id_short_len: int = 8) -> None:
        if int(initial_counter) < 0:
            raise ValueError("initial_counter must be >= 0")
        if int(run_id_short_len) <= 0:
            raise ValueError("run_id_short_len must be >= 1")

        self._lock = threading.Lock()
        self._counter = int(initial_counter)
        self._run_id_short_len = int(run_id_short_len)

    def generate_intent_id(self, intent: TradeIntent) -> Result[str]:
        """Generate deterministic intent_id from intent payload."""

        canonical = self._canonicalize_intent(intent)
        if canonical.status is not ResultStatus.SUCCESS or canonical.data is None:
            error = canonical.error or RuntimeError("intent canonicalization failed")
            reason = canonical.reason_code or "INTENT_CANONICALIZATION_FAILED"
            return Result.failed(error, reason)

        try:
            hash_hex = hashlib.sha256(canonical.data).hexdigest()
            hash_prefix = hash_hex[:12]
            date_tag = self._get_date_tag(intent.created_at_ns)
            return Result.success(f"I-{date_tag}-{hash_prefix}")
        except Exception as exc:  # noqa: BLE001 - boundary returns Result
            return Result.failed(exc, "INTENT_ID_GENERATION_FAILED")

    def generate_order_id(self, intent_id: str, run_id: str) -> Result[str]:
        """Generate unique order id for broker submission (timestamp+counter)."""

        try:
            if not str(intent_id).strip():
                return Result.failed(ValueError("intent_id is required"), "INVALID_INTENT_ID")
            run_id_short = self._short_run_id(run_id)
            if run_id_short is None:
                return Result.failed(ValueError("run_id is required"), "INVALID_RUN_ID")

            date_tag = self._get_date_tag(None)
            counter = self._next_counter()
            return Result.success(f"O-{date_tag}-{run_id_short}-{counter}")
        except Exception as exc:  # noqa: BLE001 - boundary returns Result
            return Result.failed(exc, "ORDER_ID_GENERATION_FAILED")

    def _next_counter(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def _short_run_id(self, run_id: str) -> str | None:
        value = str(run_id).strip()
        if not value:
            return None
        compact = "".join(ch for ch in value if ch.isalnum())
        if not compact:
            return None
        return compact[: self._run_id_short_len]

    def _canonicalize_intent(self, intent: TradeIntent) -> Result[bytes]:
        """Canonicalize intent payload into stable bytes for hashing.

        Canonicalization rules:
        - Only fields required by the architecture decision participate:
          (symbol, intent_type, quantity, entry_price, created_at_ns)
        - Field names are sorted and serialized deterministically.
        - Enums are serialized via ``.value`` when present.
        """

        try:
            symbol = str(intent.symbol).strip().upper()
            if not symbol:
                return Result.failed(ValueError("intent.symbol is required"), "INVALID_INTENT")

            intent_type: Any = intent.intent_type
            intent_type_value = getattr(intent_type, "value", intent_type)
            intent_type_str = str(intent_type_value)
            if not intent_type_str:
                return Result.failed(ValueError("intent.intent_type is required"), "INVALID_INTENT")

            payload: dict[str, Any] = {
                "created_at_ns": int(intent.created_at_ns),
                "entry_price": None if intent.entry_price is None else float(intent.entry_price),
                "intent_type": intent_type_str,
                "quantity": float(intent.quantity),
                "symbol": symbol,
            }

            # Sort fields for stable ordering, then serialize as a list of pairs.
            ordered_pairs = [(k, payload[k]) for k in sorted(payload.keys())]
            canonical = msgspec.json.encode(ordered_pairs)
            return Result.success(canonical)
        except Exception as exc:  # noqa: BLE001 - boundary returns Result
            return Result.failed(exc, "INTENT_CANONICALIZATION_FAILED")

    def _get_date_tag(self, created_at_ns: int | None) -> str:
        """Return a YYYYMMDD date tag in UTC.

        If created_at_ns is None, uses current time.
        """

        if created_at_ns is None:
            now_ns = time.time_ns()
            dt = datetime.fromtimestamp(now_ns / 1e9, tz=UTC)
        else:
            dt = datetime.fromtimestamp(int(created_at_ns) / 1e9, tz=UTC)
        return dt.strftime("%Y%m%d")
