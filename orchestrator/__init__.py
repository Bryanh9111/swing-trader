"""Public API exports for the orchestrator subsystem."""

from .eod_scan import EODScanOrchestrator
from .interface import ModuleResult, OrchestratorContext, RunSummary
from .plugins import DataPlugin, RiskGatePlugin, ScannerPlugin, StrategyPlugin, UniversePlugin, register_real_plugins
from .stubs import (
    DataStubPlugin,
    RiskGateStubPlugin,
    ScannerStubPlugin,
    StrategyStubPlugin,
    UniverseStubPlugin,
    register_stub_plugins,
)
from .summary import generate_summary

__all__ = [
    "EODScanOrchestrator",
    "RunSummary",
    "ModuleResult",
    "OrchestratorContext",
    "UniversePlugin",
    "DataPlugin",
    "ScannerPlugin",
    "StrategyPlugin",
    "RiskGatePlugin",
    "register_real_plugins",
    "UniverseStubPlugin",
    "DataStubPlugin",
    "ScannerStubPlugin",
    "StrategyStubPlugin",
    "RiskGateStubPlugin",
    "register_stub_plugins",
    "generate_summary",
]
