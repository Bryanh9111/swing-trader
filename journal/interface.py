"""Typed journal interfaces for capturing snapshot and run metadata.

This module provides canonical schema definitions for journaling outputs
produced by automated trading workflows. The structures defined here are
designed for deterministic serialization via ``msgspec.Struct`` and mirror
the style established in ``common/interface.py``. They standardize how run
executions are described (``RunMetadata``) and how snapshot payloads should
be versioned (``SnapshotBase``).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from msgspec import Struct


class RunType(str, Enum):
    """Enumerate the discrete workflow entry points supported by the journal.

    The values map directly to scheduled or on-demand runs and are used in
    file naming, logging context, and downstream analytics.

    Example:
        >>> RunType.PRE_MARKET_FULL_SCAN.value
        'PRE_MARKET_FULL_SCAN'
    """

    PRE_MARKET_FULL_SCAN = "PRE_MARKET_FULL_SCAN"
    INTRADAY_CHECK_1030 = "INTRADAY_CHECK_1030"
    INTRADAY_CHECK_1230 = "INTRADAY_CHECK_1230"
    INTRADAY_CHECK_1430 = "INTRADAY_CHECK_1430"
    PRE_CLOSE_CLEANUP = "PRE_CLOSE_CLEANUP"
    AFTER_MARKET_RECON = "AFTER_MARKET_RECON"
    PRE_MARKET_CHECK = "PRE_MARKET_CHECK"


class OperatingMode(str, Enum):
    """Trading system operating modes reflected in the journal payloads.

    These values indicate how orders are handled for a given run and are
    critical for reporting and audit pipelines that differentiate simulated
    executions from real capital deployments.

    Example:
        >>> OperatingMode.DRY_RUN.name
        'DRY_RUN'
    """

    DRY_RUN = "DRY_RUN"
    PAPER = "PAPER"
    LIVE = "LIVE"


class SnapshotBase(Struct, frozen=True, kw_only=True):
    """Base schema embedded by all persisted snapshot payloads.

    Attributes:
        schema_version: Semantic version string describing the snapshot
            payload contract (e.g., ``"1.0.0"``). Consumers rely on this for
            compatibility checks.
        system_version: Git commit hash of the code that generated the
            snapshot. This ties journal records back to the exact binary that
            produced them.
        asof_timestamp: Nanosecond Unix epoch timestamp representing when the
            snapshot view was taken. Used for temporal joins and historical
            replay.

    Example:
        >>> SnapshotBase(
        ...     schema_version="1.0.0",
        ...     system_version="abc1234",
        ...     asof_timestamp=1699999999999999999,
        ... )
        SnapshotBase(schema_version='1.0.0', system_version='abc1234', asof_timestamp=1699999999999999999)
    """

    schema_version: str
    system_version: str
    asof_timestamp: int


class RunMetadata(Struct, frozen=True, kw_only=True):
    """Lifecycle details for a single journalled automation run.

    Attributes:
        run_id: Unique identifier formatted as
            ``YYYYMMDD-HHMMSS-<run_type>-<uuid4>``. This value should be used
            verbatim when correlating logs, artifacts, and downstream events.
        run_type: Enumeration member describing which workflow executed.
        mode: Operating mode indicating whether the run was dry, paper, or
            live.
        system_version: Git commit hash of the executing code, mirroring the
            value stored in ``SnapshotBase.system_version``.
        start_time: Nanosecond Unix epoch timestamp when the run started.
        end_time: Nanosecond Unix epoch timestamp when the run completed, or
            ``None`` if still in progress.
        status: High-level lifecycle label (e.g., ``"running"``, ``"completed"``,
            ``"failed"``). Choice of values is left to the orchestrator but
            should be stable for external consumers.

    Example:
        >>> RunMetadata(
        ...     run_id="20240115-103000-PRE_MARKET_FULL_SCAN-123e4567-e89b-12d3-a456-426614174000",
        ...     run_type=RunType.PRE_MARKET_FULL_SCAN,
        ...     mode=OperatingMode.PAPER,
        ...     system_version="def5678",
        ...     start_time=1705314600000000000,
        ...     end_time=None,
        ...     status="running",
        ... )
        RunMetadata(run_id='20240115-103000-PRE_MARKET_FULL_SCAN-123e4567-e89b-12d3-a456-426614174000', run_type=<RunType.PRE_MARKET_FULL_SCAN: 'PRE_MARKET_FULL_SCAN'>, mode=<OperatingMode.PAPER: 'PAPER'>, system_version='def5678', start_time=1705314600000000000, end_time=None, status='running')
    """

    run_id: str
    run_type: RunType
    mode: OperatingMode
    system_version: str
    start_time: int
    end_time: Optional[int]
    status: str
