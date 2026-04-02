"""Structured exception hierarchy for the Automated Swing Trader (AST).

This module implements the error taxonomy described in
``docs/patterns/common-infrastructure-patterns.md`` and expands on the
guidance captured in ``CLAUDE.md``.  The hierarchy distinguishes between
recoverable errors that should transition a component into a ``DEGRADED``
state, critical faults that force a ``FAILED`` state, and operational
failures where the caller decides whether a retry or fallback is viable.

Each exception captures a standardized ``reason_code``, emitting module, and
optional structured ``details`` payload to aid observability.  Errors can be
created from existing exceptions to preserve causal chains and easily convert
into the :class:`common.interface.Result` type used throughout the system.

Example:

.. code-block:: python

    from common.exceptions import BrokerConnectionError
    from common.interface import ResultStatus

    try:
        broker.submit_order(order)
    except TimeoutError as original:
        err = BrokerConnectionError.from_error(
            original,
            module="execution.broker",
            reason_code="BROKER_TIMEOUT",
            details={"order_id": order.id},
        )
        result = err.to_result(
            data=None,
            status=ResultStatus.DEGRADED,
        )
        logger.warning("Order submission degraded", error=err.to_dict())
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Final, Self, TypeVar

from .interface import Result, ResultStatus

ErrorDetails = Mapping[str, Any]
T = TypeVar("T")


class ASTError(Exception):
    """Base class for all AST-specific exceptions.

    The base class enforces structured error reporting by requiring a
    ``reason_code`` and the emitting ``module``.  Optional ``details`` can be
    supplied to capture structured metadata (e.g., entity identifiers or guard
    inputs), and ``original_error`` may be provided to retain causal context
    when wrapping lower-level exceptions.

    Subclasses should define ``default_reason_code`` and, if appropriate,
    override ``default_status`` to align with the component lifecycle semantics
    (``DEGRADED`` vs. ``FAILED``).  When creating new error types, prefer
    adding a specialized subclass rather than reusing the base directly so that
    log aggregation and metrics can bucket the failure accurately.
    """

    default_reason_code: ClassVar[str] = "AST_ERROR"
    default_status: ClassVar[ResultStatus] = ResultStatus.FAILED

    def __init__(
        self,
        message: str,
        *,
        module: str,
        reason_code: str | None = None,
        details: ErrorDetails | None = None,
        original_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message: Final[str] = message
        self.module: Final[str] = module
        self.reason_code: Final[str] = reason_code or self.default_reason_code
        self.details: Final[dict[str, Any]] = dict(details or {})
        self.original_error: Final[BaseException | None] = original_error
        if original_error is not None:
            self.__cause__ = original_error

    # NOTE: __repr__ intentionally mirrors the constructor signature to give
    # structured observability when rendered in logs or tracebacks.
    def __repr__(self) -> str:  # noqa: D401 - simple data representation.
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"module={self.module!r}, "
            f"reason_code={self.reason_code!r}, "
            f"details={self.details!r}, "
            f"original_error={self.original_error!r})"
        )

    def __str__(self) -> str:
        base = (
            f"[{self.reason_code}] {self.__class__.__name__} "
            f"(module={self.module}) {self.message}"
        )
        if self.details:
            base = f"{base} details={self.details}"
        if self.original_error is not None:
            base = f"{base} caused_by={self.original_error!r}"
        return base

    @classmethod
    def from_error(
        cls,
        error: BaseException,
        *,
        module: str,
        message: str | None = None,
        reason_code: str | None = None,
        details: ErrorDetails | None = None,
    ) -> Self:
        """Create a structured wrapper around an existing exception.

        Parameters
        ----------
        error:
            The underlying exception to wrap.
        module:
            Name of the module (typically ``package.component``) where the
            error surfaced.
        message:
            Optional human-readable annotation; defaults to ``str(error)`` when
            omitted.
        reason_code:
            Optional override for the class-level ``default_reason_code``.
        details:
            Optional structured context for diagnostics.
        """
        inferred_message = message or str(error) or cls.__name__
        return cls(
            inferred_message,
            module=module,
            reason_code=reason_code,
            details=details,
            original_error=error,
        )

    def to_result(
        self,
        *,
        data: T | None = None,
        status: ResultStatus | None = None,
    ) -> Result[T | None]:
        """Convert the error into a :class:`common.interface.Result`.

        The default result status follows ``default_status`` but can be
        overridden when the caller needs finer control, such as promoting an
        ``OperationalError`` to ``DEGRADED`` after applying a local retry.
        """
        final_status = status or self.default_status
        if final_status is ResultStatus.DEGRADED:
            return Result.degraded(data, self, self.reason_code)
        if final_status is ResultStatus.FAILED:
            return Result.failed(self, self.reason_code)
        if final_status is ResultStatus.SUCCESS:
            msg = (
                "Attempted to convert an error into a SUCCESS Result. "
                "Callers should create Result.success(...) explicitly."
            )
            raise ValueError(msg)
        raise ValueError(f"Unsupported result status: {final_status!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a structured dictionary representing the error context."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "module": self.module,
            "reason_code": self.reason_code,
            "details": dict(self.details),
            "original_error": repr(self.original_error) if self.original_error else None,
        }


class RecoverableError(ASTError):
    """Base class for recoverable errors leading to ``DEGRADED`` state.

    Raise subclasses of this error when a component can continue operating with
    reduced functionality or partial data. Typical handling involves logging
    at ``warning`` level, triggering backoff, and returning a degraded
    :class:`Result`.
    """

    default_reason_code: ClassVar[str] = "RECOVERABLE_ERROR"
    default_status: ClassVar[ResultStatus] = ResultStatus.DEGRADED


class DataSourceUnavailableError(RecoverableError):
    """Raised when a primary or failover data source cannot be reached."""

    default_reason_code: ClassVar[str] = "DATA_SOURCE_UNAVAILABLE"


class BrokerConnectionError(RecoverableError):
    """Raised when broker connectivity fails due to transient network issues."""

    default_reason_code: ClassVar[str] = "BROKER_CONNECTION_ERROR"


class RateLimitExceededError(RecoverableError):
    """Raised when an external service throttles requests."""

    default_reason_code: ClassVar[str] = "RATE_LIMIT_EXCEEDED"


class PartialDataError(RecoverableError):
    """Raised when only a subset of expected data could be retrieved safely."""

    default_reason_code: ClassVar[str] = "PARTIAL_DATA"


class CriticalError(ASTError):
    """Base class for unrecoverable faults triggering a ``FAILED`` state.

    Use subclasses when the system cannot proceed without human intervention,
    such as corrupted configuration, invalid state transitions, or security
    violations.  These errors typically emit ``error`` logs and stop the
    offending component.
    """

    default_reason_code: ClassVar[str] = "CRITICAL_ERROR"
    default_status: ClassVar[ResultStatus] = ResultStatus.FAILED


class ConfigurationError(CriticalError):
    """Raised when configuration values are missing, malformed, or insecure."""

    default_reason_code: ClassVar[str] = "CONFIGURATION_ERROR"


class ValidationError(CriticalError):
    """Raised when data validation or schema checks fail irrecoverably."""

    default_reason_code: ClassVar[str] = "VALIDATION_ERROR"


class SystemStateError(CriticalError):
    """Raised when a component attempts an illegal state transition."""

    default_reason_code: ClassVar[str] = "SYSTEM_STATE_ERROR"


class SecurityError(CriticalError):
    """Raised when authentication, authorization, or integrity checks fail."""

    default_reason_code: ClassVar[str] = "SECURITY_ERROR"


class OperationalError(ASTError):
    """Base class for operational failures where recoverability is contextual.

    These errors represent business-level failures such as rejected orders or
    insufficient capital. Callers decide whether to downgrade or fail based on
    runtime conditions.  Use :meth:`ASTError.to_result` with an explicit status
    to reflect the outcome.
    """

    default_reason_code: ClassVar[str] = "OPERATIONAL_ERROR"
    default_status: ClassVar[ResultStatus] = ResultStatus.FAILED


class OrderRejectedError(OperationalError):
    """Raised when an order submission is rejected by the broker or exchange."""

    default_reason_code: ClassVar[str] = "ORDER_REJECTED"


class InsufficientFundsError(OperationalError):
    """Raised when there is not enough capital to complete an operation."""

    default_reason_code: ClassVar[str] = "INSUFFICIENT_FUNDS"


class TimeoutError(OperationalError):  # noqa: A001 - intentional override.
    """Raised when an operation exceeds the allocated time budget."""

    default_reason_code: ClassVar[str] = "OPERATION_TIMEOUT"


__all__ = [
    "ASTError",
    "RecoverableError",
    "DataSourceUnavailableError",
    "BrokerConnectionError",
    "RateLimitExceededError",
    "PartialDataError",
    "CriticalError",
    "ConfigurationError",
    "ValidationError",
    "SystemStateError",
    "SecurityError",
    "OperationalError",
    "OrderRejectedError",
    "InsufficientFundsError",
    "TimeoutError",
    "ErrorDetails",
]
