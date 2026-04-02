from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import msgspec.json
import pytest

from common.interface import ResultStatus
from order_state_machine.id_generator import IDGenerator
from strategy.interface import IntentType, TradeIntent


def _dt_to_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


def test_generate_intent_id_same_intent_same_id() -> None:
    gen = IDGenerator()
    created_at = _dt_to_ns(datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC))
    intent = TradeIntent(
        intent_id="ignored",
        symbol="aapl",
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=created_at,
        entry_price=100.0,
    )
    a = gen.generate_intent_id(intent)
    b = gen.generate_intent_id(intent)
    assert a.status is ResultStatus.SUCCESS and b.status is ResultStatus.SUCCESS
    assert a.data == b.data
    assert a.data and a.data.startswith("I-20250102-")


def test_generate_intent_id_different_intent_different_id() -> None:
    gen = IDGenerator()
    created_at = _dt_to_ns(datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC))
    base = dict(
        intent_id="ignored",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=created_at,
        entry_price=100.0,
    )
    a = gen.generate_intent_id(TradeIntent(**base))
    b = gen.generate_intent_id(TradeIntent(**{**base, "quantity": 11.0}))
    assert a.status is ResultStatus.SUCCESS and b.status is ResultStatus.SUCCESS
    assert a.data != b.data


def test_generate_intent_id_invalid_intent_returns_failed() -> None:
    gen = IDGenerator()
    intent = TradeIntent(
        intent_id="ignored",
        symbol="   ",
        intent_type=IntentType.OPEN_LONG,
        quantity=1.0,
        created_at_ns=1,
        entry_price=None,
    )
    result = gen.generate_intent_id(intent)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_INTENT"


@dataclass(frozen=True, slots=True)
class _BadIntent:
    symbol: str
    intent_type: Any
    quantity: float
    created_at_ns: int
    entry_price: float | None = None


def test_generate_intent_id_empty_intent_type_returns_failed() -> None:
    gen = IDGenerator()
    result = gen.generate_intent_id(_BadIntent(symbol="AAPL", intent_type="", quantity=1.0, created_at_ns=1))  # type: ignore[arg-type]
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_INTENT"


def test_canonicalize_intent_sorts_fields() -> None:
    gen = IDGenerator()
    intent = TradeIntent(
        intent_id="ignored",
        symbol="aapl",
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=123,
        entry_price=None,
    )
    result = gen._canonicalize_intent(intent)
    assert result.status is ResultStatus.SUCCESS and result.data is not None
    pairs = msgspec.json.decode(result.data, type=list[list[Any]])
    keys = [k for k, _ in pairs]
    assert keys == sorted(keys)
    assert keys == ["created_at_ns", "entry_price", "intent_type", "quantity", "symbol"]


def test_get_date_tag_formats_yyyymmdd() -> None:
    gen = IDGenerator()
    created_at = _dt_to_ns(datetime(2024, 2, 29, 12, 0, 0, tzinfo=UTC))
    assert gen._get_date_tag(created_at) == "20240229"


def test_generate_order_id_happy_path_unique_and_format(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = IDGenerator()
    fixed_now = _dt_to_ns(datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC))
    monkeypatch.setattr("order_state_machine.id_generator.time.time_ns", lambda: fixed_now)

    r1 = gen.generate_order_id("I-1", "2025-01-02-run-abcdef")
    r2 = gen.generate_order_id("I-1", "2025-01-02-run-abcdef")
    assert r1.status is ResultStatus.SUCCESS and r2.status is ResultStatus.SUCCESS
    assert r1.data != r2.data

    pattern = re.compile(r"^O-20250102-20250102-1$")
    assert r1.data and pattern.match(r1.data)
    assert r2.data and r2.data.endswith("-2")


def test_generate_order_id_restart_safe_counter_seeding(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = IDGenerator(initial_counter=41)
    fixed_now = _dt_to_ns(datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC))
    monkeypatch.setattr("order_state_machine.id_generator.time.time_ns", lambda: fixed_now)

    r = gen.generate_order_id("I-1", "run-xyz")
    assert r.status is ResultStatus.SUCCESS
    assert r.data and r.data.endswith("-42")


def test_generate_order_id_invalid_inputs_return_failed() -> None:
    gen = IDGenerator()

    r1 = gen.generate_order_id("   ", "run-1")
    assert r1.status is ResultStatus.FAILED
    assert r1.reason_code == "INVALID_INTENT_ID"

    r2 = gen.generate_order_id("I-1", "   ")
    assert r2.status is ResultStatus.FAILED
    assert r2.reason_code == "INVALID_RUN_ID"


def test_id_generator_init_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        IDGenerator(initial_counter=-1)
    with pytest.raises(ValueError):
        IDGenerator(run_id_short_len=0)


def test_generate_order_id_internal_exception_returns_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = IDGenerator()
    monkeypatch.setattr(gen, "_next_counter", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    r = gen.generate_order_id("I-1", "run-1")
    assert r.status is ResultStatus.FAILED
    assert r.reason_code == "ORDER_ID_GENERATION_FAILED"


def test_generate_intent_id_internal_exception_returns_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = IDGenerator()
    created_at = _dt_to_ns(datetime(2025, 1, 2, tzinfo=UTC))
    intent = TradeIntent(
        intent_id="ignored",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=created_at,
        entry_price=100.0,
    )

    class _Boom:
        def __call__(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr("order_state_machine.id_generator.hashlib.sha256", _Boom())
    r = gen.generate_intent_id(intent)
    assert r.status is ResultStatus.FAILED
    assert r.reason_code == "INTENT_ID_GENERATION_FAILED"


def test_generate_order_id_run_id_with_no_alnum_returns_failed() -> None:
    gen = IDGenerator()
    r = gen.generate_order_id("I-1", "---")
    assert r.status is ResultStatus.FAILED
    assert r.reason_code == "INVALID_RUN_ID"


def test_canonicalize_intent_exception_returns_failed() -> None:
    gen = IDGenerator()

    @dataclass(frozen=True, slots=True)
    class _WeirdIntent:
        symbol: str
        intent_type: Any
        quantity: Any
        created_at_ns: int
        entry_price: float | None = None

    out = gen._canonicalize_intent(
        _WeirdIntent(symbol="AAPL", intent_type="OPEN_LONG", quantity="bad", created_at_ns=1)  # type: ignore[arg-type]
    )
    assert out.status is ResultStatus.FAILED
    assert out.reason_code == "INTENT_CANONICALIZATION_FAILED"
