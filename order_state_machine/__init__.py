"""Phase 3.4 Order State Machine.

This package defines the order intent lifecycle infrastructure used by
Automated Swing Trader and exposes a stable, top-level import surface (matching
the Phase 1–3.3 pattern from ``common`` and ``journal``).

Core features:
- Intent ID generation
- Order state machine and transition validation
- Persistence for intent ↔ broker order mappings
- Reconciliation to derive state transitions from broker/local state

Public API:
- Types + protocols: ``OrderState``, ``IntentOrderMapping``,
  ``IntentOrderMappingSet``, ``ReconciliationResult``, ``StateTransition``,
  ``IDGeneratorProtocol``, ``StateMachineProtocol``, ``PersistenceProtocol``,
  ``ReconciliationProtocol``
- Implementations: ``IDGenerator``, ``StateMachine``, ``Persistence``,
  ``Reconciler``

.. code-block:: python

    from pathlib import Path

    from order_state_machine import OrderState, Persistence, Reconciler, StateMachine

    sm = StateMachine()
    assert sm.is_valid_transition(OrderState.NEW, OrderState.SUBMITTED)

    store = Persistence(storage_dir=Path("data"))
    mappings = store.load_all().data or []
    result = Reconciler(state_machine=sm).reconcile_startup(
        local_mappings=mappings,
        broker_orders=[],
    )
"""

# Interface exports
from .interface import (
    IDGeneratorProtocol,
    IntentOrderMapping,
    IntentOrderMappingSet,
    OrderState,
    PersistenceProtocol,
    ReconciliationProtocol,
    ReconciliationResult,
    StateMachineProtocol,
    StateTransition,
)

# Implementation exports
from .id_generator import IDGenerator
from .persistence import JSONLPersistence as Persistence
from .reconciliation import Reconciler
from .state_machine import StateMachine

__all__ = [
    # Interface exports
    "OrderState",
    "IntentOrderMapping",
    "IntentOrderMappingSet",
    "ReconciliationResult",
    "StateTransition",
    "IDGeneratorProtocol",
    "StateMachineProtocol",
    "PersistenceProtocol",
    "ReconciliationProtocol",
    # Implementation exports
    "IDGenerator",
    "StateMachine",
    "Persistence",
    "Reconciler",
]

__version__ = "0.1.0"
