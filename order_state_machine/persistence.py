"""JSONL-backed persistence for intent-to-order mappings (Pattern 5).

- Primary storage: append-only JSONL (``intent_order_mappings.jsonl``)
- Index: in-memory ``{intent_id: IntentOrderMapping}`` (rebuilt on startup)
- Writes: atomic temp file + ``os.replace`` to avoid partial-line corruption
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path

import msgspec.json

from common.exceptions import CriticalError, OperationalError, PartialDataError, RecoverableError
from common.interface import Result, ResultStatus
from order_state_machine.interface import IntentOrderMapping, PersistenceProtocol

__all__ = ["JSONLPersistence"]


class JSONLPersistence(PersistenceProtocol):
    """Persist intent-order mappings in an append-only JSONL file."""

    _MODULE_NAME = "order_state_machine.persistence"

    _DEFAULT_FILENAME = "intent_order_mappings.jsonl"

    _MAPPING_NOT_FOUND_REASON = "MAPPING_NOT_FOUND"
    _MAPPING_DECODE_FAILED_REASON = "MAPPING_DECODE_FAILED"
    _MAPPING_IO_FAILED_REASON = "MAPPING_IO_FAILED"
    _MAPPING_PERMISSION_DENIED_REASON = "MAPPING_PERMISSION_DENIED"
    _MAPPING_ALREADY_PERSISTED_REASON = "MAPPING_ALREADY_PERSISTED"

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
        self._mappings: dict[str, IntentOrderMapping] = {}

    @property
    def path(self) -> Path:
        return self._path

    def save(self, mapping: IntentOrderMapping) -> Result[None]:
        """Append mapping to the JSONL log and update index (idempotent for equals)."""

        with self._lock:
            load_result = self._ensure_loaded_locked()
            if load_result.status is ResultStatus.FAILED:
                return load_result

            existing = self._mappings.get(mapping.intent_id)
            if existing == mapping:
                if load_result.status is ResultStatus.DEGRADED:
                    return Result.degraded(
                        data=None,
                        error=load_result.error or RuntimeError("Degraded load state."),
                        reason_code=load_result.reason_code
                        or self._MAPPING_IO_FAILED_REASON,
                    )
                return Result.success(data=None, reason_code=self._MAPPING_ALREADY_PERSISTED_REASON)

            encoded_line = msgspec.json.encode(mapping) + b"\n"
            append_result = self._append_to_file_locked(encoded_line)
            if append_result.status is not ResultStatus.SUCCESS:
                return append_result

            self._mappings[mapping.intent_id] = mapping

            if load_result.status is ResultStatus.DEGRADED:
                return Result.degraded(
                    data=None,
                    error=load_result.error or RuntimeError("Degraded load state."),
                    reason_code=load_result.reason_code
                    or self._MAPPING_IO_FAILED_REASON,
                )

            return Result.success(data=None)

    def load(self, intent_id: str) -> Result[IntentOrderMapping]:
        """Load mapping by intent ID from the in-memory index."""

        with self._lock:
            load_result = self._ensure_loaded_locked()
            if load_result.status is ResultStatus.FAILED:
                return Result.failed(
                    error=load_result.error or RuntimeError("Failed to load mappings."),
                    reason_code=load_result.reason_code or self._MAPPING_IO_FAILED_REASON,
                )

            mapping = self._mappings.get(intent_id)
            if mapping is None:
                error = OperationalError(
                    f"No mapping found for intent_id={intent_id}.",
                    module=self._MODULE_NAME,
                    reason_code=self._MAPPING_NOT_FOUND_REASON,
                    details={"intent_id": intent_id},
                )
                return Result.failed(error=error, reason_code=error.reason_code)

            if load_result.status is ResultStatus.DEGRADED:
                return Result.degraded(
                    data=mapping,
                    error=load_result.error or RuntimeError("Degraded load state."),
                    reason_code=load_result.reason_code
                    or self._MAPPING_IO_FAILED_REASON,
                )

            return Result.success(mapping)

    def load_all(self) -> Result[list[IntentOrderMapping]]:
        """Load all mappings (current index state)."""

        with self._lock:
            load_result = self._ensure_loaded_locked()
            mappings = list(self._mappings.values())

            if load_result.status is ResultStatus.FAILED:
                return Result.failed(
                    error=load_result.error or RuntimeError("Failed to load mappings."),
                    reason_code=load_result.reason_code or self._MAPPING_IO_FAILED_REASON,
                )

            if load_result.status is ResultStatus.DEGRADED:
                return Result.degraded(
                    data=mappings,
                    error=load_result.error or RuntimeError("Degraded load state."),
                    reason_code=load_result.reason_code
                    or self._MAPPING_IO_FAILED_REASON,
                )

            return Result.success(mappings)

    def exists(self, intent_id: str) -> bool:
        """Return True when a mapping exists for ``intent_id``.

        Note: If the backing file cannot be loaded (e.g., permission errors),
        this method fails closed and returns True to prevent duplicate order
        submissions when idempotency checks cannot be performed reliably.
        """

        with self._lock:
            if not self._loaded:
                load_result = self._ensure_loaded_locked()
                if load_result.status is ResultStatus.FAILED:
                    return True
            return intent_id in self._mappings

    def _ensure_loaded_locked(self) -> Result[None]:
        """Load mappings from disk into the in-memory index once."""

        if self._loaded:
            return Result.success(data=None)

        load_result = self._load_from_file_locked()
        if load_result.status is ResultStatus.FAILED:
            return Result.failed(
                error=load_result.error or RuntimeError("Failed to load mappings."),
                reason_code=load_result.reason_code or self._MAPPING_IO_FAILED_REASON,
            )

        decoded = load_result.data or []
        self._mappings = {mapping.intent_id: mapping for mapping in decoded}
        self._loaded = True

        if load_result.status is ResultStatus.DEGRADED:
            return Result.degraded(
                data=None,
                error=load_result.error or RuntimeError("Degraded load state."),
                reason_code=load_result.reason_code or self._MAPPING_DECODE_FAILED_REASON,
            )

        return Result.success(data=None)

    def _append_to_file_locked(self, encoded_line: bytes) -> Result[None]:
        """Atomically append a JSONL line to the backing file."""

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
                reason_code=self._MAPPING_PERMISSION_DENIED_REASON,
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
                reason_code=self._MAPPING_IO_FAILED_REASON,
                details={"path": str(self._path)},
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)

        return Result.success(data=None)

    def _load_from_file_locked(self) -> Result[list[IntentOrderMapping]]:
        """Load all JSONL entries from disk."""

        if not self._path.exists():
            return Result.success([])

        decoded: list[IntentOrderMapping] = []
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
                            msgspec.json.decode(line, type=IntentOrderMapping),
                        )
                    except Exception as exc:  # noqa: BLE001 - decode failures expected.
                        decode_error_count += 1
                        if first_decode_error is None:
                            first_decode_error = exc
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._MAPPING_PERMISSION_DENIED_REASON,
                details={"path": str(self._path)},
            )
            return Result.failed(error=error, reason_code=error.reason_code)
        except OSError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._MAPPING_IO_FAILED_REASON,
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
                reason_code=self._MAPPING_DECODE_FAILED_REASON,
                details={"path": str(self._path), "decode_errors": decode_error_count},
            )
            return error.to_result(data=decoded, status=ResultStatus.DEGRADED)

        return Result.success(decoded)
