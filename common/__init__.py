"""Common infrastructure components for AST.

This package provides foundational infrastructure used across all AST modules:
- Configuration loading with layered YAML + environment variable support
- Structured logging with context binding
- Domain event publishing and subscription
- Typed result wrappers for graceful degradation
- Structured exception hierarchy

Example usage:

.. code-block:: python

    from common import (
        YAMLConfigLoader,
        init_logging,
        StructlogLoggerFactory,
        InMemoryEventBus,
        Result,
        DomainEvent,
    )

    # Load configuration
    loader = YAMLConfigLoader(config_dir=Path("config"))
    config = loader.load_config(env="dev")

    # Initialize logging
    init_logging(config.system)

    # Get logger
    factory = StructlogLoggerFactory()
    logger = factory.get_logger(module="my_module", run_id="run_001")

    # Create event bus
    bus = InMemoryEventBus()
    bus.subscribe("events.*", lambda e: logger.info("Event received", event=e))

    # Publish event
    event = DomainEvent(
        event_id="evt_001",
        event_type="test",
        run_id="run_001",
        module="my_module",
        timestamp_ns=0,
        data={"key": "value"}
    )
    bus.publish("events.test", event)
"""

# Configuration
from .config import (
    BaseConfig,
    ConfigLoaderError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    SystemConfig,
    YAMLConfigLoader,
)

# Exceptions
from .exceptions import (
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

# Events
from .events import EventPersistence, InMemoryEventBus

# Interface types
from .interface import (
    BoundLogger,
    Config,
    ConfigLoader,
    DomainEvent,
    DomainEventHandler,
    EventBus,
    LoggerFactory,
    Result,
    ResultStatus,
)

# Logging
from .logging import (
    LoggingConfigurationError,
    StructlogLoggerFactory,
    init_logging,
    update_component_log_levels,
)

__all__ = [
    # Configuration
    "BaseConfig",
    "ConfigLoaderError",
    "ConfigFileNotFoundError",
    "ConfigValidationError",
    "SystemConfig",
    "YAMLConfigLoader",
    # Exceptions
    "ASTError",
    "BrokerConnectionError",
    "ConfigurationError",
    "CriticalError",
    "DataSourceUnavailableError",
    "InsufficientFundsError",
    "OperationalError",
    "OrderRejectedError",
    "PartialDataError",
    "RateLimitExceededError",
    "RecoverableError",
    "SecurityError",
    "SystemStateError",
    "TimeoutError",
    "ValidationError",
    # Events
    "EventPersistence",
    "InMemoryEventBus",
    # Interface types
    "BoundLogger",
    "Config",
    "ConfigLoader",
    "DomainEvent",
    "DomainEventHandler",
    "EventBus",
    "LoggerFactory",
    "Result",
    "ResultStatus",
    # Logging
    "LoggingConfigurationError",
    "StructlogLoggerFactory",
    "init_logging",
    "update_component_log_levels",
]

__version__ = "0.1.0"
