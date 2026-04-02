from __future__ import annotations

import logging
import logging.handlers
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import structlog

from common import logging as logging_module
from common.logging import (
    LoggingConfigurationError,
    StructlogLoggerFactory,
    _build_renderer,
    _parse_level,
    init_logging,
    update_component_log_levels,
)


@pytest.fixture
def logging_env(monkeypatch: pytest.MonkeyPatch) -> None:
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = list(root_logger.handlers)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    monkeypatch.setattr(logging_module, "_LOGGING_INITIALISED", False)
    monkeypatch.setattr(logging_module, "_COMPONENT_LEVELS", {})
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()

    yield

    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)

    for handler in original_handlers:
        root_logger.addHandler(handler)

    root_logger.setLevel(original_level)
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


def _make_system_config(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "version": "0.0.1",
        "log_level": "INFO",
        "log_output": str(tmp_path / "logs" / "system.log"),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_parse_level_supports_numeric_and_strings() -> None:
    assert _parse_level(10) == 10
    assert _parse_level("info") == logging.INFO

    with pytest.raises(LoggingConfigurationError):
        _parse_level("not-a-level")


def test_build_renderer_variants() -> None:
    assert isinstance(_build_renderer("json"), structlog.processors.JSONRenderer)
    assert isinstance(_build_renderer("console"), structlog.dev.ConsoleRenderer)

    with pytest.raises(LoggingConfigurationError):
        _build_renderer("yaml")


def test_init_logging_json_file_handler(tmp_path: Path, logging_env: None) -> None:
    config = _make_system_config(
        tmp_path,
        log_format="json",
        log_console_enabled=False,
        log_file_max_bytes=2048,
        log_file_backup_count=1,
        log_component_levels={"common.events": "DEBUG"},
    )

    init_logging(config, force=True)

    root_logger = logging.getLogger()
    assert len(root_logger.handlers) == 1
    handler = root_logger.handlers[0]
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert Path(config.log_output).exists()
    assert logging.getLogger("common.events").level == logging.DEBUG

    logger = structlog.get_logger("common.events")
    logger.info("structured log", extra_field="value")
    handler.flush()

    log_lines = Path(config.log_output).read_text(encoding="utf-8").strip().splitlines()
    assert log_lines, "Expected log file to contain rendered entries"
    rendered = json.loads(log_lines[-1])
    assert rendered["event"] == "structured log"
    assert rendered["extra_field"] == "value"


def test_init_logging_console_renderer(tmp_path: Path, logging_env: None) -> None:
    config = _make_system_config(
        tmp_path,
        log_output="",
        log_format="console",
        log_console_enabled=True,
    )

    init_logging(config, force=True)

    root_logger = logging.getLogger()
    console_handlers = [handler for handler in root_logger.handlers if isinstance(handler, logging.StreamHandler)]
    assert console_handlers, "Console handler should be configured"
    formatter = console_handlers[0].formatter
    assert formatter is not None

    console_handler = console_handlers[0]
    stream = io.StringIO()
    console_handler.stream = stream

    logger = structlog.get_logger("console-test")
    logger.warning("console log", extra_field="value")
    console_handler.flush()

    output = stream.getvalue()
    assert "console log" in output
    assert "extra_field" in output


def test_init_logging_force_reconfigures(tmp_path: Path, logging_env: None) -> None:
    first_config = _make_system_config(
        tmp_path,
        log_output=str(tmp_path / "logs" / "first.log"),
        log_console_enabled=False,
    )
    second_config = _make_system_config(
        tmp_path,
        log_level="WARNING",
        log_output=str(tmp_path / "logs" / "second.log"),
        log_console_enabled=False,
    )

    init_logging(first_config, force=True)
    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO
    first_handlers = list(root_logger.handlers)
    init_logging(second_config)
    assert root_logger.level == logging.INFO
    assert list(root_logger.handlers) == first_handlers

    init_logging(second_config, force=True)
    assert root_logger.level == logging.WARNING
    assert Path(second_config.log_output).exists()


def test_update_component_log_levels_runtime(tmp_path: Path, logging_env: None) -> None:
    config = _make_system_config(
        tmp_path,
        log_console_enabled=False,
    )
    init_logging(config, force=True)

    update_component_log_levels({"custom.component": "ERROR"})

    assert logging.getLogger("custom.component").level == logging.ERROR


def test_structlog_logger_factory_binds_context(tmp_path: Path, logging_env: None) -> None:
    config = _make_system_config(tmp_path, log_console_enabled=False)
    init_logging(config, force=True)

    factory = StructlogLoggerFactory(default_event_type="trade_event")
    logger = factory.get_logger("module.component", "run-123")

    bound_logger = logger.bind(extra="value")
    assert logger._context["module"] == "module.component"  # type: ignore[attr-defined]
    assert logger._context["run_id"] == "run-123"  # type: ignore[attr-defined]
    assert logger._context["event_type"] == "trade_event"  # type: ignore[attr-defined]
    assert bound_logger._context["extra"] == "value"  # type: ignore[attr-defined]


def test_init_logging_thread_safe(tmp_path: Path, logging_env: None) -> None:
    config = _make_system_config(
        tmp_path,
        log_output=str(tmp_path / "logs" / "threaded.log"),
        log_console_enabled=False,
    )

    def _init() -> None:
        init_logging(config, force=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: _init(), range(8)))

    assert logging.getLogger().handlers


def test_file_rotation_creates_backups(tmp_path: Path, logging_env: None) -> None:
    log_path = tmp_path / "logs" / "rotating.log"
    config = _make_system_config(
        tmp_path,
        log_output=str(log_path),
        log_console_enabled=False,
        log_file_max_bytes=300,
        log_file_backup_count=2,
    )

    init_logging(config, force=True)

    logger = logging.getLogger("rotating")
    for _ in range(40):
        logger.info("x" * 200)

    for handler in logging.getLogger().handlers:
        handler.flush()

    backup_path = log_path.with_suffix(log_path.suffix + ".1")
    assert log_path.exists()
    assert backup_path.exists()


def test_invalid_log_level_raises(tmp_path: Path, logging_env: None) -> None:
    config = _make_system_config(tmp_path, log_level="INVALID", log_output="")

    with pytest.raises(LoggingConfigurationError):
        init_logging(config, force=True)
