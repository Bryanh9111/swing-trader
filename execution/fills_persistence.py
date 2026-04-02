"""JSONL-backed persistence for execution fills.

- Primary storage: append-only JSONL (``fills.jsonl``)
- Index: in-memory ``{client_order_id: [FillDetail, ...]}`` (rebuilt on startup)
- Writes: atomic temp file + ``os.replace`` to avoid partial-line corruption
- Idempotent: duplicate ``execution_id`` values are silently skipped
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import msgspec
import msgspec.json

from common.exceptions import CriticalError, PartialDataError, RecoverableError
from common.interface import Result, ResultStatus
from execution.interface import FillDetail

__all__ = ["FillsPersistence"]


class _FillRecord(msgspec.Struct, frozen=True, kw_only=True):
    """On-disk representation: one fill per JSONL line."""

    client_order_id: str
    fill: FillDetail


class FillsPersistence:
    """Persist execution fills in an append-only JSONL file."""

    _MODULE_NAME = "execution.fills_persistence"
    _DEFAULT_FILENAME = "fills.jsonl"

    _IO_FAILED_REASON = "FILLS_IO_FAILED"
    _PERMISSION_DENIED_REASON = "FILLS_PERMISSION_DENIED"
    _DECODE_FAILED_REASON = "FILLS_DECODE_FAILED"

    def __init__(
        self,
        storage_dir: str | Path,
        *,
        filename: str = _DEFAULT_FILENAME,
    ) -> None:
        self._storage_dir = Path(storage_dir)
        self._path = self._storage_dir / filename

        self._lock = threading.Lock()
        self._loaded = False
        self._fills: dict[str, list[FillDetail]] = {}
        self._seen_execution_ids: set[str] = set()

    @property
    def path(self) -> Path:
        return self._path

    def save_fill(self, client_order_id: str, fill: FillDetail) -> Result[None]:
        """Append a fill to the JSONL log (idempotent by execution_id)."""

        with self._lock:
            load_result = self._ensure_loaded_locked()
            if load_result.status is ResultStatus.FAILED:
                return load_result

            if fill.execution_id in self._seen_execution_ids:
                return Result.success(data=None)

            record = _FillRecord(client_order_id=client_order_id, fill=fill)
            encoded_line = msgspec.json.encode(record) + b"\n"
            append_result = self._append_to_file_locked(encoded_line)
            if append_result.status is not ResultStatus.SUCCESS:
                return append_result

            self._fills.setdefault(client_order_id, []).append(fill)
            self._seen_execution_ids.add(fill.execution_id)

            return Result.success(data=None)

    def load_fills(self, client_order_id: str) -> Result[list[FillDetail]]:
        """Load fills for a specific order from the in-memory index."""

        with self._lock:
            load_result = self._ensure_loaded_locked()
            if load_result.status is ResultStatus.FAILED:
                return Result.failed(
                    error=load_result.error or RuntimeError("Failed to load fills."),
                    reason_code=load_result.reason_code or self._IO_FAILED_REASON,
                )

            fills = list(self._fills.get(client_order_id, []))

            if load_result.status is ResultStatus.DEGRADED:
                return Result.degraded(
                    data=fills,
                    error=load_result.error or RuntimeError("Degraded load state."),
                    reason_code=load_result.reason_code or self._IO_FAILED_REASON,
                )

            return Result.success(fills)

    def load_all(self) -> Result[dict[str, list[FillDetail]]]:
        """Load all fills grouped by client_order_id (startup rebuild)."""

        with self._lock:
            load_result = self._ensure_loaded_locked()
            data = {k: list(v) for k, v in self._fills.items()}

            if load_result.status is ResultStatus.FAILED:
                return Result.failed(
                    error=load_result.error or RuntimeError("Failed to load fills."),
                    reason_code=load_result.reason_code or self._IO_FAILED_REASON,
                )

            if load_result.status is ResultStatus.DEGRADED:
                return Result.degraded(
                    data=data,
                    error=load_result.error or RuntimeError("Degraded load state."),
                    reason_code=load_result.reason_code or self._IO_FAILED_REASON,
                )

            return Result.success(data)

    def exists(self, execution_id: str) -> bool:
        """Return True when a fill with this execution_id has been persisted."""

        with self._lock:
            if not self._loaded:
                try:
                    load_result = self._ensure_loaded_locked()
                except Exception:  # noqa: BLE001 - fail-closed
                    return True
                if load_result.status is ResultStatus.FAILED:
                    return True  # fail-closed
            return execution_id in self._seen_execution_ids

    def _ensure_loaded_locked(self) -> Result[None]:
        if self._loaded:
            return Result.success(data=None)

        load_result = self._load_from_file_locked()
        if load_result.status is ResultStatus.FAILED:
            return Result.failed(
                error=load_result.error or RuntimeError("Failed to load fills."),
                reason_code=load_result.reason_code or self._IO_FAILED_REASON,
            )

        records = load_result.data or []
        fills: dict[str, list[FillDetail]] = {}
        seen: set[str] = set()
        for record in records:
            if record.fill.execution_id in seen:
                continue
            fills.setdefault(record.client_order_id, []).append(record.fill)
            seen.add(record.fill.execution_id)

        self._fills = fills
        self._seen_execution_ids = seen
        self._loaded = True

        if load_result.status is ResultStatus.DEGRADED:
            return Result.degraded(
                data=None,
                error=load_result.error or RuntimeError("Degraded load state."),
                reason_code=load_result.reason_code or self._DECODE_FAILED_REASON,
            )

        return Result.success(data=None)

    def _append_to_file_locked(self, encoded_line: bytes) -> Result[None]:
        temp_path: Path | None = None
        try:
            self._storage_dir.mkdir(parents=True, exist_ok=True)

            file_exists = self._path.exists()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{self._path.name}.",
                dir=str(self._storage_dir),
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                if file_exists:
                    with open(self._path, "rb") as source:
                        shutil.copyfileobj(source, temp_file)
                temp_file.write(encoded_line)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            os.replace(temp_path, self._path)
        except PermissionError as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._PERMISSION_DENIED_REASON,
                details={"path": str(self._path)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)
        except OSError as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._IO_FAILED_REASON,
                details={"path": str(self._path)},
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)

        return Result.success(data=None)

    def _load_from_file_locked(self) -> Result[list[_FillRecord]]:
        if not self._path.exists():
            return Result.success([])

        decoded: list[_FillRecord] = []
        first_decode_error: BaseException | None = None
        decode_error_count = 0

        try:
            with open(self._path, "rb") as file:
                for raw_line in file:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        decoded.append(
                            msgspec.json.decode(line, type=_FillRecord),
                        )
                    except Exception as exc:  # noqa: BLE001
                        decode_error_count += 1
                        if first_decode_error is None:
                            first_decode_error = exc
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._PERMISSION_DENIED_REASON,
                details={"path": str(self._path)},
            )
            return Result.failed(error=error, reason_code=error.reason_code)
        except OSError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._IO_FAILED_REASON,
                details={"path": str(self._path)},
            )
            return Result.degraded(
                data=[],
                error=error,
                reason_code=error.reason_code,
            )

        if decode_error_count:
            error = PartialDataError.from_error(
                first_decode_error or RuntimeError("Unknown decode error."),
                module=self._MODULE_NAME,
                reason_code=self._DECODE_FAILED_REASON,
                details={"path": str(self._path), "decode_errors": decode_error_count},
            )
            return error.to_result(data=decoded, status=ResultStatus.DEGRADED)

        return Result.success(decoded)
