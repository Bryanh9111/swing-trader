"""Interface contracts for AST common infrastructure components.

This module captures the interface boundaries described in
``docs/patterns/common-infrastructure-patterns.md`` so that downstream
implementations can adhere to a consistent set of behaviours for config
loading, structured logging, pub/sub eventing, and result propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

try:  # pragma: no cover - structlog is optional at interface definition time.
    from structlog.typing import BindableLogger as BoundLogger
except Exception:  # noqa: BLE001 - fallback definition for typing only.
    class BoundLogger(Protocol):
        """Fallback protocol matching the subset of structlog BoundLogger API."""

        def bind(self, **new_values: Any) -> "BoundLogger":
            ...

        def debug(self, event: str, **kwargs: Any) -> None:
            ...

        def info(self, event: str, **kwargs: Any) -> None:
            ...

        def warning(self, event: str, **kwargs: Any) -> None:
            ...

        def error(self, event: str, **kwargs: Any) -> None:
            ...

        def exception(self, event: str, **kwargs: Any) -> None:
            ...


@runtime_checkable
class Config(Protocol):
    """Marker protocol for typed, immutable configuration value objects.

    Implementations typically derive from ``msgspec.Struct`` or frozen
    ``dataclasses.dataclass`` definitions to ensure type safety, hashability,
    and deterministic serialization semantics.
    """


ConfigT = TypeVar("ConfigT", bound=Config)


class ConfigLoader(Protocol[ConfigT]):
    """Interface for loading layered configuration objects.

    Implementations must layer configuration sources in the following order:

    1. ``base.yaml`` – baseline defaults shared across environments.
    2. ``{env}.yaml`` – environment-specific overrides (e.g., ``prod.yaml``).
    3. ``local.yaml`` – developer overrides ignored in source control.
    4. ``os.environ`` – final overrides for secrets and runtime toggles.

    The loader should merge each layer immutably, validate the resulting data,
    and return an instance of the target ``Config`` type.
    """

    def load_config(self, env: str) -> ConfigT:
        """Load and validate the configuration for the supplied environment."""


class LoggerFactory(Protocol):
    """Interface for constructing structured loggers with contextual binding."""

    def get_logger(self, module: str, run_id: str) -> BoundLogger:
        """Return a logger bound to ``run_id`` and ``module`` context.

        Implementations must support binding additional context (e.g.,
        ``event_type``) on demand and allow switching between JSON and console
        renderers without altering the caller contract.
        """


DomainEventHandler = Callable[["DomainEvent"], None]


class EventBus(Protocol):
    """Interface for publishing and subscribing to domain events."""

    def publish(self, topic: str, event: "DomainEvent") -> None:
        """Publish a domain event to the specified topic."""

    def subscribe(self, pattern: str, handler: DomainEventHandler) -> None:
        """Register a handler for topics matching ``pattern``.

        Wildcard patterns must support ``*`` for multi-character matching and
        ``?`` for single-character matching in keeping with the documented bus
        topology conventions.
        """


class ResultStatus(str, Enum):
    """Discrete lifecycle outcomes for service invocations."""

    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"


T = TypeVar("T")


@dataclass(slots=True)
class Result(Generic[T]):
    """Structured result wrapper for command/query operations."""

    status: ResultStatus
    data: Optional[T] = None
    error: Optional[BaseException] = None
    reason_code: Optional[str] = None

    @staticmethod
    def success(data: T, reason_code: str | None = "OK") -> "Result[T]":
        """Create a success result with the supplied payload."""
        return Result(status=ResultStatus.SUCCESS, data=data, reason_code=reason_code)

    @staticmethod
    def degraded(
        data: Optional[T],
        error: BaseException,
        reason_code: str,
    ) -> "Result[T]":
        """Create a degraded result capturing partial progress or fallback data."""
        return Result(
            status=ResultStatus.DEGRADED,
            data=data,
            error=error,
            reason_code=reason_code,
        )

    @staticmethod
    def failed(error: BaseException, reason_code: str) -> "Result[T]":
        """Create a failed result when an operation could not be completed."""
        return Result(status=ResultStatus.FAILED, error=error, reason_code=reason_code)


try:  # pragma: no cover - prefer msgspec.Struct when available.
    from msgspec import Struct as _MsgspecStruct

    class DomainEvent(_MsgspecStruct, kw_only=True):
        """Typed representation of domain events published on the event bus."""

        event_id: str
        event_type: str
        run_id: str
        module: str
        timestamp_ns: int
        data: Optional[dict[str, Any]] = None

except Exception:  # noqa: BLE001 - simplify dependency management.

    @dataclass(slots=True, kw_only=True)  # pragma: no cover - fallback only.
    class DomainEvent:
        """Typed representation of domain events published on the event bus."""

        event_id: str
        event_type: str
        run_id: str
        module: str
        timestamp_ns: int
        data: Optional[dict[str, Any]] = None
