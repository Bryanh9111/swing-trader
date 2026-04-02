"""Tests for execution.fills_persistence.FillsPersistence."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from common.interface import ResultStatus
from execution.fills_persistence import FillsPersistence
from execution.interface import FillDetail


def _make_fill(execution_id: str = "exec-1", price: float = 100.0, qty: float = 10.0) -> FillDetail:
    return FillDetail(
        execution_id=execution_id,
        fill_price=price,
        fill_quantity=qty,
        fill_time_ns=1_000_000_000,
    )


class TestSaveAndLoad:
    def test_save_and_load_basic(self, tmp_path: Path) -> None:
        store = FillsPersistence(tmp_path)
        fill = _make_fill("exec-1")

        result = store.save_fill("order-1", fill)
        assert result.status is ResultStatus.SUCCESS

        load_result = store.load_fills("order-1")
        assert load_result.status is ResultStatus.SUCCESS
        assert load_result.data is not None
        assert len(load_result.data) == 1
        assert load_result.data[0].execution_id == "exec-1"

    def test_save_multiple_fills_same_order(self, tmp_path: Path) -> None:
        store = FillsPersistence(tmp_path)
        store.save_fill("order-1", _make_fill("exec-1"))
        store.save_fill("order-1", _make_fill("exec-2", price=101.0))

        result = store.load_fills("order-1")
        assert result.status is ResultStatus.SUCCESS
        assert len(result.data) == 2

    def test_save_fills_different_orders(self, tmp_path: Path) -> None:
        store = FillsPersistence(tmp_path)
        store.save_fill("order-1", _make_fill("exec-1"))
        store.save_fill("order-2", _make_fill("exec-2"))

        result = store.load_all()
        assert result.status is ResultStatus.SUCCESS
        assert len(result.data) == 2
        assert len(result.data["order-1"]) == 1
        assert len(result.data["order-2"]) == 1

    def test_load_missing_order_returns_empty(self, tmp_path: Path) -> None:
        store = FillsPersistence(tmp_path)

        result = store.load_fills("nonexistent")
        assert result.status is ResultStatus.SUCCESS
        assert result.data == []

    def test_load_all_empty_file(self, tmp_path: Path) -> None:
        store = FillsPersistence(tmp_path)

        result = store.load_all()
        assert result.status is ResultStatus.SUCCESS
        assert result.data == {}


class TestIdempotency:
    def test_duplicate_execution_id_skipped(self, tmp_path: Path) -> None:
        store = FillsPersistence(tmp_path)
        fill = _make_fill("exec-1")

        store.save_fill("order-1", fill)
        store.save_fill("order-1", fill)  # duplicate

        result = store.load_fills("order-1")
        assert len(result.data) == 1

    def test_exists_returns_true_for_saved(self, tmp_path: Path) -> None:
        store = FillsPersistence(tmp_path)
        store.save_fill("order-1", _make_fill("exec-1"))

        assert store.exists("exec-1") is True
        assert store.exists("exec-999") is False


class TestPersistenceAcrossInstances:
    def test_data_survives_new_instance(self, tmp_path: Path) -> None:
        store1 = FillsPersistence(tmp_path)
        store1.save_fill("order-1", _make_fill("exec-1"))
        store1.save_fill("order-1", _make_fill("exec-2", price=105.0))

        # New instance reads from disk
        store2 = FillsPersistence(tmp_path)
        result = store2.load_all()
        assert result.status is ResultStatus.SUCCESS
        assert len(result.data["order-1"]) == 2

    def test_idempotency_across_instances(self, tmp_path: Path) -> None:
        store1 = FillsPersistence(tmp_path)
        store1.save_fill("order-1", _make_fill("exec-1"))

        store2 = FillsPersistence(tmp_path)
        store2.save_fill("order-1", _make_fill("exec-1"))  # duplicate

        result = store2.load_fills("order-1")
        assert len(result.data) == 1


class TestDegradation:
    def test_file_not_exists_returns_empty(self, tmp_path: Path) -> None:
        store = FillsPersistence(tmp_path / "nonexistent_subdir")

        result = store.load_all()
        assert result.status is ResultStatus.SUCCESS
        assert result.data == {}

    def test_corrupt_line_degrades(self, tmp_path: Path) -> None:
        store = FillsPersistence(tmp_path)
        store.save_fill("order-1", _make_fill("exec-1"))

        # Append corrupt line
        with open(store.path, "ab") as f:
            f.write(b"not valid json\n")

        store2 = FillsPersistence(tmp_path)
        result = store2.load_all()
        assert result.status is ResultStatus.DEGRADED
        assert len(result.data["order-1"]) == 1

    def test_exists_fails_closed_on_permission_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        store = FillsPersistence(tmp_path)

        def _raise_perm(*_a, **_kw):
            raise PermissionError("nope")

        monkeypatch.setattr(store, "_load_from_file_locked", _raise_perm)

        # Should fail-closed (return True)
        assert store.exists("anything") is True


class TestThreadSafety:
    def test_concurrent_saves(self, tmp_path: Path) -> None:
        store = FillsPersistence(tmp_path)
        errors: list[Exception] = []

        def writer(order_id: str, exec_ids: list[str]) -> None:
            for eid in exec_ids:
                try:
                    store.save_fill(order_id, _make_fill(eid))
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("order-1", [f"exec-{i}" for i in range(10)])),
            threading.Thread(target=writer, args=("order-2", [f"exec-{10 + i}" for i in range(10)])),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

        result = store.load_all()
        assert result.status is ResultStatus.SUCCESS
        assert len(result.data.get("order-1", [])) == 10
        assert len(result.data.get("order-2", [])) == 10
