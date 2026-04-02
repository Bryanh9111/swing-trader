from __future__ import annotations

import pytest

from common.exceptions import (
    ASTError,
    BrokerConnectionError,
    ConfigurationError,
    CriticalError,
    DataSourceUnavailableError,
    InsufficientFundsError,
    OperationalError,
    OrderRejectedError,
    PartialDataError,
    RateLimitExceededError,
    RecoverableError,
    SecurityError,
    SystemStateError,
    TimeoutError,
    ValidationError,
)
from common.interface import ResultStatus


@pytest.mark.parametrize(
    ("exc_cls", "expected_status", "expected_reason"),
    [
        (ASTError, ResultStatus.FAILED, "AST_ERROR"),
        (RecoverableError, ResultStatus.DEGRADED, "RECOVERABLE_ERROR"),
        (DataSourceUnavailableError, ResultStatus.DEGRADED, "DATA_SOURCE_UNAVAILABLE"),
        (BrokerConnectionError, ResultStatus.DEGRADED, "BROKER_CONNECTION_ERROR"),
        (RateLimitExceededError, ResultStatus.DEGRADED, "RATE_LIMIT_EXCEEDED"),
        (PartialDataError, ResultStatus.DEGRADED, "PARTIAL_DATA"),
        (CriticalError, ResultStatus.FAILED, "CRITICAL_ERROR"),
        (ConfigurationError, ResultStatus.FAILED, "CONFIGURATION_ERROR"),
        (ValidationError, ResultStatus.FAILED, "VALIDATION_ERROR"),
        (SystemStateError, ResultStatus.FAILED, "SYSTEM_STATE_ERROR"),
        (SecurityError, ResultStatus.FAILED, "SECURITY_ERROR"),
        (OperationalError, ResultStatus.FAILED, "OPERATIONAL_ERROR"),
        (OrderRejectedError, ResultStatus.FAILED, "ORDER_REJECTED"),
        (InsufficientFundsError, ResultStatus.FAILED, "INSUFFICIENT_FUNDS"),
        (TimeoutError, ResultStatus.FAILED, "OPERATION_TIMEOUT"),
    ],
)
def test_exception_creation_defaults(exc_cls: type[ASTError], expected_status: ResultStatus, expected_reason: str) -> None:
    error = exc_cls("test message", module="tests.unit")

    assert isinstance(error, exc_cls)
    assert error.reason_code == expected_reason
    assert error.default_status is expected_status
    assert expected_reason in repr(error)
    assert f"module={error.module!r}" in repr(error)


def test_from_error_preserves_chain_and_overrides() -> None:
    original = ValueError("boom")
    error = ConfigurationError.from_error(
        original,
        module="tests.config",
        message="override message",
        reason_code="CONFIG_OVERRIDE",
        details={"key": "value"},
    )

    assert error.message == "override message"
    assert error.original_error is original
    assert error.__cause__ is original
    assert error.reason_code == "CONFIG_OVERRIDE"
    assert error.details == {"key": "value"}
    assert "caused_by" in str(error)


def test_to_result_defaults_to_degraded_for_recoverable() -> None:
    error = DataSourceUnavailableError("partial outage", module="tests.ds")
    result = error.to_result(data={"fallback": True})

    assert result.status is ResultStatus.DEGRADED
    assert result.data == {"fallback": True}
    assert result.error is error
    assert result.reason_code == error.reason_code


def test_to_result_can_force_failure() -> None:
    error = RecoverableError("explicit failure", module="tests.recover")
    result = error.to_result(status=ResultStatus.FAILED)

    assert result.status is ResultStatus.FAILED
    assert result.error is error
    assert result.data is None


def test_to_result_success_not_allowed() -> None:
    error = ASTError("cannot succeed", module="tests.core")

    with pytest.raises(ValueError):
        error.to_result(status=ResultStatus.SUCCESS)


def test_to_result_unsupported_status_raises() -> None:
    error = ASTError("bad status", module="tests.core")

    with pytest.raises(ValueError):
        error.to_result(status="UNSUPPORTED")  # type: ignore[arg-type]


def test_to_dict_serialises_context() -> None:
    original = KeyError("missing")
    error = OrderRejectedError(
        "Rejected by broker",
        module="execution.broker",
        details={"order_id": "abc"},
        original_error=original,
    )

    payload = error.to_dict()
    assert payload["error_type"] == "OrderRejectedError"
    assert payload["module"] == "execution.broker"
    assert payload["details"] == {"order_id": "abc"}
    assert payload["original_error"] == repr(original)


def test_operational_error_result_status_control() -> None:
    error = OperationalError("retryable", module="tests.ops")
    result = error.to_result(status=ResultStatus.DEGRADED, data="partial")

    assert result.status is ResultStatus.DEGRADED
    assert result.data == "partial"
    assert result.reason_code == error.reason_code


def test_str_representation_includes_details() -> None:
    error = TimeoutError(
        "Reached timeout",
        module="tests.timeout",
        details={"duration": 30},
        original_error=RuntimeError("slow call"),
    )

    rendered = str(error)
    assert "[OPERATION_TIMEOUT]" in rendered
    assert "duration" in rendered
    assert "slow call" in rendered

