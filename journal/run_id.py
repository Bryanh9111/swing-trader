"""Run identifier utilities for the journaling subsystem.

This module provides a focused helper for generating run identifiers that
encode the execution date, time, run type, and a UUID4 suffix. It also
exposes parsing and git metadata helpers that mirror the conventions used
throughout the Automated Swing Trader project.
"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
import os
from typing import Any
from uuid import UUID, uuid4

from .interface import RunType

__all__ = ["RunIDGenerator"]


_RUN_TYPE_ALIASES: dict[str, str] = {
    "EOD_SCAN": "PRE_MARKET_FULL_SCAN",
    "PRE_CLOSE_RECON": "AFTER_MARKET_RECON",
}

_RUN_ID_PATTERN = re.compile(
    r"^(?P<date>\d{8})-"
    r"(?P<time>\d{6})-"
    r"(?P<run_type>[A-Z0-9_]+)-"
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
)


class RunIDGenerator:
    """Generate, parse, and contextualize run identifiers."""

    @staticmethod
    def generate_run_id(run_type: RunType) -> str:
        """Generate a run identifier for the provided run type.

        Args:
            run_type: Enumerated run entry point.

        Returns:
            A run identifier formatted as ``YYYYMMDD-HHMMSS-<run_type>-<uuid4>``.

        Raises:
            ValueError: If ``run_type`` is not a ``RunType`` instance.
        """

        if not isinstance(run_type, RunType):
            raise ValueError("run_type must be an instance of RunType.")

        timestamp = datetime.now(tz=timezone.utc)
        return (
            f"{timestamp.strftime('%Y%m%d-%H%M%S')}"
            f"-{run_type.value}"
            f"-{uuid4()}"
        )

    @staticmethod
    def parse_run_id(run_id: str) -> dict[str, Any]:
        """Parse and validate a run identifier.

        Args:
            run_id: Run identifier produced by ``generate_run_id``.

        Returns:
            A mapping containing the date, time, resolved ``RunType``, and UUID.

        Raises:
            ValueError: If the run identifier is malformed or references an
                unknown run type.
        """

        match = _RUN_ID_PATTERN.fullmatch(run_id)
        if not match:
            raise ValueError(f"run_id has an invalid format: {run_id!r}")

        payload = match.groupdict()

        try:
            datetime.strptime(
                f"{payload['date']}{payload['time']}",
                "%Y%m%d%H%M%S",
            )
        except ValueError as exc:
            raise ValueError(
                f"run_id contains an invalid timestamp: {run_id!r}",
            ) from exc

        try:
            raw_type = payload["run_type"]
            resolved_type = _RUN_TYPE_ALIASES.get(raw_type, raw_type)
            parsed_run_type = RunType(resolved_type)
        except ValueError as exc:
            raise ValueError(
                f"run_id references an unknown run type: {payload['run_type']!r}",
            ) from exc

        try:
            uuid_value = str(UUID(payload["uuid"]))
        except ValueError as exc:
            raise ValueError(
                f"run_id contains an invalid UUID: {payload['uuid']!r}",
            ) from exc

        return {
            "date": payload["date"],
            "time": payload["time"],
            "run_type": parsed_run_type,
            "uuid": uuid_value,
        }

    @staticmethod
    def get_system_version() -> str:
        """Return the current git commit hash or ``\"unknown\"`` if unavailable.

        Notes:
            This value is used across many snapshots/events. Spawning ``git`` via
            ``subprocess`` can become both a hotspot and a reliability issue when
            called frequently (especially in multi-threaded code paths). We
            therefore:
              1) allow environment overrides,
              2) prefer a fast, subprocess-free read from ``.git``,
              3) fall back to ``git`` with a short timeout.
        """

        return _get_system_version_cached()


@lru_cache(maxsize=1)
def _get_system_version_cached() -> str:
    override = (os.getenv("AST_SYSTEM_VERSION") or os.getenv("SYSTEM_VERSION") or "").strip()
    if override:
        return override

    commit = _read_git_commit_from_dotgit()
    if commit:
        return commit

    # Fallback: subprocess (bounded).
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
            env={
                **os.environ,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
            },
        )
    except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"

    if result.returncode != 0:
        return "unknown"
    commit_hash = (result.stdout or "").strip()
    return commit_hash or "unknown"


def _read_git_commit_from_dotgit() -> str | None:
    """Best-effort commit hash resolution without spawning subprocesses."""

    git_dir = _find_git_dir(start=Path.cwd()) or _find_git_dir(start=Path(__file__).resolve())
    if git_dir is None:
        return None

    head = git_dir / "HEAD"
    try:
        head_text = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not head_text:
        return None

    if not head_text.startswith("ref:"):
        return head_text if _looks_like_commit(head_text) else None

    ref = head_text.split(":", 1)[1].strip()
    if not ref:
        return None

    # Common case: loose ref file exists.
    ref_path = git_dir / ref
    try:
        ref_text = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        ref_text = ""
    if _looks_like_commit(ref_text):
        return ref_text

    # Fallback: packed-refs lookup.
    packed = git_dir / "packed-refs"
    try:
        packed_text = packed.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in packed_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        commit, packed_ref = parts
        if packed_ref == ref and _looks_like_commit(commit):
            return commit
    return None


def _find_git_dir(start: Path) -> Path | None:
    current = start
    for _ in range(25):
        dotgit = current / ".git"
        if dotgit.is_dir():
            return dotgit
        if dotgit.is_file():
            try:
                payload = dotgit.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                payload = ""
            if payload.startswith("gitdir:"):
                candidate = payload.split(":", 1)[1].strip()
                if candidate:
                    resolved = (current / candidate).resolve()
                    if resolved.is_dir():
                        return resolved
        if current.parent == current:
            break
        current = current.parent
    return None


def _looks_like_commit(value: str) -> bool:
    text = value.strip()
    if len(text) != 40:
        return False
    return all(ch in "0123456789abcdef" for ch in text.lower())
