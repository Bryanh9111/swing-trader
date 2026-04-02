"""Structured logging utilities built on top of structlog.

This module provides a single entrypoint ``init_logging`` that configures
structlog and the standard library logging subsystem according to the AST
configuration schema, followed by a ``LoggerFactory`` implementation that
returns contextualised bound loggers for application components.

The design follows the guidance captured in
``docs/patterns/common-infrastructure-patterns.md`` and the protocol
expectations defined in ``common.interface``.
"""

from __future__ import annotations

import logging
from logging import Handler
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Mapping

import structlog
from structlog.stdlib import BoundLogger as StructlogBoundLogger
from structlog.stdlib import ProcessorFormatter
from structlog.typing import Processor

from .config import SystemConfig
from .interface import BoundLogger, LoggerFactory

__all__ = ["init_logging", "StructlogLoggerFactory", "update_component_log_levels"]

_LOGGING_LOCK = Lock()
_LOGGING_INITIALISED = False
_COMPONENT_LEVELS: dict[str, int] = {}


class LoggingConfigurationError(RuntimeError):
    """Raised when logging initialisation receives invalid configuration."""


def _parse_level(level: str | int) -> int:
    """Convert level names or numeric values into a valid ``logging`` level."""
    if isinstance(level, int):
        return level

    candidate = level.upper()
    numeric_level = logging.getLevelName(candidate)
    if isinstance(numeric_level, str):
        raise LoggingConfigurationError(f"Unknown log level '{level}'.")
    return int(numeric_level)


def _build_renderer(log_format: str) -> Processor:
    """Return the structlog renderer processor for the desired output format."""
    normalised = log_format.strip().lower()
    if normalised == "json":
        return structlog.processors.JSONRenderer()
    if normalised == "console":
        return structlog.dev.ConsoleRenderer()
    raise LoggingConfigurationError(
        f"Unsupported log_format '{log_format}'. Expected 'json' or 'console'.",
    )


def _shared_processors() -> list[Processor]:
    """Processors applied before rendering for both stdlib and structlog flows."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]


def _configure_structlog() -> None:
    """Configure the structlog global state for stdlib integration."""
    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=StructlogBoundLogger,
        cache_logger_on_first_use=True,
    )


def _configure_handlers(
    base_level: int,
    renderer: Processor,
    file_path: str | None,
    max_bytes: int,
    backup_count: int,
    enable_console: bool,
) -> None:
    """Attach logging handlers (console and rotating file) to the root logger."""
    processor_formatter = ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=_shared_processors(),
    )

    handlers: list[Handler] = []

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(processor_formatter)
        handlers.append(console_handler)

    if file_path:
        path = Path(file_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(processor_formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)
        try:
            existing_handler.close()
        except Exception:  # noqa: BLE001 - best-effort handler cleanup.
            pass
    root_logger.setLevel(base_level)
    root_logger.handlers = handlers


def _configure_component_levels(overrides: Mapping[str, int]) -> None:
    """Apply per-component log level overrides to stdlib loggers."""
    global _COMPONENT_LEVELS
    _COMPONENT_LEVELS = dict(overrides)

    for logger_name, level in overrides.items():
        logging.getLogger(logger_name).setLevel(level)


def init_logging(system_config: SystemConfig, *, force: bool = False) -> None:
    """Initialise structlog and stdlib logging according to ``SystemConfig``.

    Parameters
    ----------
    system_config:
        The ``SystemConfig`` portion of the AST ``BaseConfig``. Only a subset
        of fields is required:

        * ``log_level`` – base log level applied to the root logger.
        * ``log_output`` – path to the rotating file output. When omitted or an
          empty string, file output is disabled.
        * ``log_format`` – optional format selector (``"json"`` | ``"console"``).
        * ``log_component_levels`` – optional mapping of logger name to level.
        * ``log_file_max_bytes`` – optional max file size in bytes before
          rotation (defaults to 10 MB).
        * ``log_file_backup_count`` – optional number of retained rotated files.
        * ``log_console_enabled`` – optional toggle for console output
          (defaults to ``True``).

    force:
        When set to ``True`` the logging stack is reconfigured even if it has
        already been initialised in this process. Intended for test harnesses.
    """
    global _LOGGING_INITIALISED

    with _LOGGING_LOCK:
        if _LOGGING_INITIALISED and not force:
            return

        base_level = _parse_level(system_config.log_level)
        renderer = _build_renderer(getattr(system_config, "log_format", "json"))

        log_output = getattr(system_config, "log_output", "") or None
        max_bytes = int(getattr(system_config, "log_file_max_bytes", 10 * 1024 * 1024))
        backup_count = int(getattr(system_config, "log_file_backup_count", 5))
        enable_console = bool(getattr(system_config, "log_console_enabled", True))

        component_levels_raw = getattr(system_config, "log_component_levels", {}) or {}
        component_levels = {
            name: _parse_level(level)
            for name, level in dict(component_levels_raw).items()
        }

        structlog.reset_defaults()
        _configure_structlog()
        _configure_handlers(
            base_level=base_level,
            renderer=renderer,
            file_path=log_output,
            max_bytes=max_bytes,
            backup_count=backup_count,
            enable_console=enable_console,
        )
        _configure_component_levels(component_levels)

        structlog.contextvars.clear_contextvars()
        _LOGGING_INITIALISED = True


class StructlogLoggerFactory(LoggerFactory):
    """Create structlog bound loggers with ``run_id`` and ``module`` context."""

    def __init__(self, *, default_event_type: str | None = None) -> None:
        self._default_event_type = default_event_type

    def get_logger(self, module: str, run_id: str) -> BoundLogger:
        """Return a structlog bound logger for the supplied component details."""
        logger = structlog.get_logger(module)
        bound = logger.bind(module=module, run_id=run_id)
        if self._default_event_type is not None:
            bound = bound.bind(event_type=self._default_event_type)
        return bound


def update_component_log_levels(overrides: Mapping[str, str | int]) -> None:
    """Dynamically override component log levels after initialisation."""
    parsed_overrides = {name: _parse_level(level) for name, level in overrides.items()}
    _configure_component_levels(parsed_overrides)
