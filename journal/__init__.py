"""Public interfaces for the journaling subsystem.

This package exposes the canonical run lifecycle types, identifier helpers,
and artifact management utilities used by Automated Swing Trader workflows.
The exports mirror the conventions from ``common`` so callers can import from
``journal`` directly without drilling into submodules.

Example usage:

.. code-block:: python

    from journal import (
        ArtifactManager,
        OperatingMode,
        RunIDGenerator,
        RunMetadata,
        RunType,
    )

    run_id = RunIDGenerator.generate_run_id(RunType.PRE_MARKET_FULL_SCAN)
    artifacts = ArtifactManager(base_path=\"artifacts\")
    metadata = RunMetadata(
        run_id=run_id,
        run_type=RunType.PRE_MARKET_FULL_SCAN,
        mode=OperatingMode.DRY_RUN,
        system_version=RunIDGenerator.get_system_version(),
        start_time=0,
        end_time=None,
        status=\"running\",
    )
    artifacts.write_metadata(run_id, metadata)
"""

# Interface exports
from .interface import OperatingMode, RunMetadata, RunType, SnapshotBase

# Run identifier utilities
from .run_id import RunIDGenerator

# Artifact management
from .artifacts import ArtifactManager
from .reader import JournalReader
from .replay import ReplayEngine, ReplayMode
from .validator import SnapshotValidator
from .writer import JournalWriter

__all__ = [
    # Interface exports
    "OperatingMode",
    "RunMetadata",
    "RunType",
    "SnapshotBase",
    # Run identifier utilities
    "RunIDGenerator",
    # Artifact management
    "ArtifactManager",
    # High-level journal components
    "JournalWriter",
    "JournalReader",
    "ReplayEngine",
    "ReplayMode",
    "SnapshotValidator",
]

__version__ = "0.1.0"
