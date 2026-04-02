from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import msgspec.json
import msgspec.structs
import pytest

from common.exceptions import CriticalError, OperationalError, PartialDataError, RecoverableError
from common.interface import Result, ResultStatus
from order_state_machine.interface import IntentOrderMapping, OrderState
from order_state_machine.persistence import JSONLPersistence
from strategy.interface import TradeIntent


def _make_mapping(*, intent: TradeIntent, intent_id: str, state: OrderState, updated_at_ns: int = 1) -> IntentOrderMapping:
    return IntentOrderMapping(
        intent_id=intent_id,
        client_order_id=f"O-{intent_id}",
        broker_order_id=None,
        state=state,
        created_at_ns=1,
        updated_at_ns=updated_at_ns,
        # TradeIntent is type-erased at runtime; store dict to match decode behavior.
        intent_snapshot=msgspec.structs.asdict(intent),
        metadata={},
    )


def test_save_first_and_update_save_roundtrip(temp_persistence_fixture: JSONLPersistence, mock_trade_intent: TradeIntent) -> None:
    m1 = _make_mapping(intent=mock_trade_intent, intent_id="I-1", state=OrderState.PENDING, updated_at_ns=10)
    r1 = temp_persistence_fixture.save(m1)
    assert r1.status is ResultStatus.SUCCESS

    m2 = _make_mapping(intent=mock_trade_intent, intent_id="I-1", state=OrderState.SUBMITTED, updated_at_ns=20)
    r2 = temp_persistence_fixture.save(m2)
    assert r2.status is ResultStatus.SUCCESS

    loaded = temp_persistence_fixture.load("I-1")
    assert loaded.status is ResultStatus.SUCCESS
    assert loaded.data == m2


def test_save_idempotent_same_mapping_returns_already_persisted_reason(
    temp_persistence_fixture: JSONLPersistence,
    mock_trade_intent: TradeIntent,
) -> None:
    m1 = _make_mapping(intent=mock_trade_intent, intent_id="I-1", state=OrderState.PENDING)
    assert temp_persistence_fixture.save(m1).status is ResultStatus.SUCCESS

    again = temp_persistence_fixture.save(m1)
    assert again.status is ResultStatus.SUCCESS
    assert again.reason_code == JSONLPersistence._MAPPING_ALREADY_PERSISTED_REASON


def test_save_is_atomic_uses_os_replace(
    temp_persistence_fixture: JSONLPersistence,
    mock_trade_intent: TradeIntent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def tracking_replace(src: str | bytes | os.PathLike[str] | os.PathLike[bytes], dst: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        calls.append((Path(src), Path(dst)))
        original_replace(src, dst)

    monkeypatch.setattr("order_state_machine.persistence.os.replace", tracking_replace)

    m1 = _make_mapping(intent=mock_trade_intent, intent_id="I-1", state=OrderState.PENDING)
    r1 = temp_persistence_fixture.save(m1)
    assert r1.status is ResultStatus.SUCCESS
    assert calls

    temp_path, final_path = calls[0]
    assert final_path == temp_persistence_fixture.path
    assert temp_path.parent == temp_persistence_fixture.path.parent
    assert temp_path.name.startswith(f".{temp_persistence_fixture.path.name}.")
    assert not temp_path.exists()
    assert final_path.exists()


def test_load_missing_returns_operational_error(temp_persistence_fixture: JSONLPersistence) -> None:
    r = temp_persistence_fixture.load("missing")
    assert r.status is ResultStatus.FAILED
    assert isinstance(r.error, OperationalError)
    assert r.reason_code == JSONLPersistence._MAPPING_NOT_FOUND_REASON


def test_load_all_empty_then_multiple(temp_persistence_fixture: JSONLPersistence, mock_trade_intent: TradeIntent) -> None:
    empty = temp_persistence_fixture.load_all()
    assert empty.status is ResultStatus.SUCCESS
    assert empty.data == []

    m1 = _make_mapping(intent=mock_trade_intent, intent_id="I-1", state=OrderState.PENDING)
    m2 = _make_mapping(intent=mock_trade_intent, intent_id="I-2", state=OrderState.ACCEPTED)
    assert temp_persistence_fixture.save(m1).status is ResultStatus.SUCCESS
    assert temp_persistence_fixture.save(m2).status is ResultStatus.SUCCESS

    loaded = temp_persistence_fixture.load_all()
    assert loaded.status is ResultStatus.SUCCESS
    assert {m.intent_id for m in (loaded.data or [])} == {"I-1", "I-2"}


def test_exists_true_false(temp_persistence_fixture: JSONLPersistence, mock_trade_intent: TradeIntent) -> None:
    assert temp_persistence_fixture.exists("I-1") is False
    m1 = _make_mapping(intent=mock_trade_intent, intent_id="I-1", state=OrderState.PENDING)
    assert temp_persistence_fixture.save(m1).status is ResultStatus.SUCCESS
    assert temp_persistence_fixture.exists("I-1") is True


def test_thread_safety_concurrent_save_and_load(tmp_path: Path, mock_trade_intent: TradeIntent) -> None:
    store = JSONLPersistence(storage_dir=tmp_path)
    mappings = [_make_mapping(intent=mock_trade_intent, intent_id=f"I-{i}", state=OrderState.PENDING) for i in range(20)]

    def save_one(m: IntentOrderMapping) -> Result[None]:
        return store.save(m)

    def load_one(intent_id: str) -> Result[IntentOrderMapping]:
        return store.load(intent_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        save_results = list(pool.map(save_one, mappings))
        assert all(r.status is ResultStatus.SUCCESS for r in save_results)
        load_results = list(pool.map(load_one, [m.intent_id for m in mappings]))
        assert all(r.status is ResultStatus.SUCCESS for r in load_results)


def test_save_permission_error_returns_critical_error(
    tmp_path: Path,
    mock_trade_intent: TradeIntent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JSONLPersistence(storage_dir=tmp_path)
    monkeypatch.setattr("order_state_machine.persistence.os.replace", lambda *_: (_ for _ in ()).throw(PermissionError("denied")))

    r = store.save(_make_mapping(intent=mock_trade_intent, intent_id="I-1", state=OrderState.PENDING))
    assert r.status is ResultStatus.FAILED
    assert isinstance(r.error, CriticalError)
    assert r.reason_code == JSONLPersistence._MAPPING_PERMISSION_DENIED_REASON


def test_save_io_error_returns_recoverable_error_degraded(
    tmp_path: Path,
    mock_trade_intent: TradeIntent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JSONLPersistence(storage_dir=tmp_path)
    monkeypatch.setattr("order_state_machine.persistence.os.replace", lambda *_: (_ for _ in ()).throw(OSError("io")))

    r = store.save(_make_mapping(intent=mock_trade_intent, intent_id="I-1", state=OrderState.PENDING))
    assert r.status is ResultStatus.DEGRADED
    assert isinstance(r.error, RecoverableError)
    assert r.reason_code == JSONLPersistence._MAPPING_IO_FAILED_REASON


def test_exists_fail_closed_returns_true_on_load_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JSONLPersistence(storage_dir=tmp_path)

    def failed_load(_: JSONLPersistence) -> Result[list[IntentOrderMapping]]:
        err = CriticalError("denied", module="test", reason_code="X")
        return Result.failed(err, "X")

    monkeypatch.setattr(JSONLPersistence, "_load_from_file_locked", failed_load)
    assert store.exists("anything") is True


def test_load_all_handles_corrupt_jsonl_lines_degraded(tmp_path: Path, mock_trade_intent: TradeIntent) -> None:
    store = JSONLPersistence(storage_dir=tmp_path)
    good = _make_mapping(intent=mock_trade_intent, intent_id="I-1", state=OrderState.PENDING)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_bytes(msgspec.json.encode(good) + b"\n" + b"{bad json\n")

    r = store.load_all()
    assert r.status is ResultStatus.DEGRADED
    assert isinstance(r.error, PartialDataError)
    assert r.reason_code == JSONLPersistence._MAPPING_DECODE_FAILED_REASON
    assert r.data == [good]


def test_save_idempotent_on_degraded_load_returns_degraded(tmp_path: Path, mock_trade_intent: TradeIntent) -> None:
    good = _make_mapping(intent=mock_trade_intent, intent_id="I-1", state=OrderState.PENDING)
    store = JSONLPersistence(storage_dir=tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_bytes(msgspec.json.encode(good) + b"\n" + b"{bad json\n")

    r = store.save(good)
    assert r.status is ResultStatus.DEGRADED
    assert r.reason_code == JSONLPersistence._MAPPING_DECODE_FAILED_REASON


def test_save_load_and_load_all_return_failed_when_initial_load_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JSONLPersistence(storage_dir=tmp_path)

    def failed_load(_: JSONLPersistence) -> Result[list[IntentOrderMapping]]:
        err = CriticalError("denied", module="test", reason_code="X")
        return Result.failed(err, "X")

    monkeypatch.setattr(JSONLPersistence, "_load_from_file_locked", failed_load)

    save_result = store.save(IntentOrderMapping(intent_id="I", client_order_id="O", state=OrderState.PENDING, created_at_ns=0, updated_at_ns=0, intent_snapshot={}))  # type: ignore[arg-type]
    assert save_result.status is ResultStatus.FAILED

    load_result = store.load("I")
    assert load_result.status is ResultStatus.FAILED

    load_all_result = store.load_all()
    assert load_all_result.status is ResultStatus.FAILED


def test_load_all_permission_error_returns_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JSONLPersistence(storage_dir=tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_bytes(b"{}\n")

    def denied(*_: object, **__: object):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", denied)
    out = store.load_all()
    assert out.status is ResultStatus.FAILED
    assert isinstance(out.error, CriticalError)
    assert out.reason_code == JSONLPersistence._MAPPING_PERMISSION_DENIED_REASON


def test_load_all_os_error_returns_degraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JSONLPersistence(storage_dir=tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_bytes(b"{}\n")

    def ioerr(*_: object, **__: object):
        raise OSError("io")

    monkeypatch.setattr("builtins.open", ioerr)
    out = store.load_all()
    assert out.status is ResultStatus.DEGRADED
    assert isinstance(out.error, RecoverableError)
    assert out.reason_code == JSONLPersistence._MAPPING_IO_FAILED_REASON
