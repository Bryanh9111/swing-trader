"""Phase 3.5 Execution layer.

The Execution layer provides a stable boundary between AST's order lifecycle
tracking (``order_state_machine``) and concrete broker integrations. It exposes
schemas + protocols for broker adapters, plus high-level orchestration that
submits/cancels orders, reconciles broker vs. local state, and produces terminal
execution reports.

Core features:
    - IBKR broker adapter boundary (``IBKRAdapter``)
    - Order orchestration (submit/cancel/query via ``OrderManager``)
    - Startup + periodic reconciliation (via the Order State Machine)
    - Execution reporting (``ExecutionReport`` with fill and transition details)

Public API:
    - Interfaces: ``FillDetail``, ``ExecutionReport``,
      ``BrokerConnectionConfig``, ``BrokerAdapterProtocol``
    - Implementations: ``IBKRAdapter``, ``OrderManager``

Example:
    .. code-block:: python

        from pathlib import Path

        from execution import BrokerConnectionConfig, IBKRAdapter, OrderManager
        from order_state_machine import IDGenerator, Persistence, Reconciler, StateMachine

        id_gen = IDGenerator()
        sm = StateMachine()
        store = Persistence(storage_dir=Path("data"))
        reconciler = Reconciler(state_machine=sm)

        config = BrokerConnectionConfig(port=7497, client_id=1, readonly=True)
        adapter = IBKRAdapter(config=config, id_generator=id_gen)
        adapter.connect()

        manager = OrderManager(
            adapter=adapter,
            id_generator=id_gen,
            state_machine=sm,
            persistence=store,
            reconciler=reconciler,
        )
"""

# Interface exports
from .interface import (
    BrokerAdapterProtocol,
    BrokerConnectionConfig,
    ExecutionReport,
    FillDetail,
)

# Implementation exports
from .ibkr_adapter import IBKRAdapter
from .order_manager import OrderManager

__all__ = [
    # Interface exports
    "FillDetail",
    "ExecutionReport",
    "BrokerConnectionConfig",
    "BrokerAdapterProtocol",
    # Implementation exports
    "IBKRAdapter",
    "OrderManager",
]

__version__ = "0.1.0"

