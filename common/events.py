"""In-process event bus implementations and supporting contracts.

This module provides a baseline ``InMemoryEventBus`` implementation that
conforms to the :class:`~common.interface.EventBus` protocol defined in
``common.interface``.  The bus offers simple publish/subscribe semantics with
support for hierarchical topics, glob-style wildcard matching, and optional
integration points for attaching durable persistence backends.
"""

from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any, Dict, List, Protocol

import fnmatch
import logging

from .interface import BoundLogger, DomainEvent, DomainEventHandler, EventBus

__all__ = ["EventPersistence", "InMemoryEventBus"]


class EventPersistence(Protocol):
    """Protocol for durable event storage backends.

    Implementations may persist events to append-only files, Redis Streams,
    databases, or any other medium that supports replay.  The protocol is kept
    intentionally small so that callers can inject lightweight adapters without
    coupling the in-memory bus to a specific persistence technology.
    """

    def append(self, topic: str, event: DomainEvent) -> None:
        """Persist the published ``event`` for the supplied ``topic``."""


class InMemoryEventBus(EventBus):
    """Thread-safe, in-process publish/subscribe event bus.

    The bus keeps a registry of subscription patterns in memory, resolving
    matching handlers for each publication using glob semantics (`*` matches
    zero or more characters, `?` matches exactly one character).  Handler
    callbacks execute synchronously.  Failures are logged and do not prevent
    delivery to other subscribers.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | BoundLogger | None = None,
        persistence: EventPersistence | None = None,
    ) -> None:
        """Create a new in-memory bus instance.

        Parameters
        ----------
        logger:
            Optional logger used to report diagnostics. Accepts either a stdlib
            ``logging.Logger`` or a structlog ``BoundLogger``.
        persistence:
            Optional backend that will receive every published event. Failures
            raised by the persistence layer are logged and suppressed to ensure
            core pub/sub delivery remains best-effort in-process.
        """
        self._logger = logger or logging.getLogger(__name__)
        self._persistence = persistence
        self._lock = RLock()
        self._subscriptions: Dict[str, List[DomainEventHandler]] = defaultdict(list)

    def publish(self, topic: str, event: DomainEvent) -> None:
        """Publish ``event`` to ``topic`` and fan out to matching subscribers."""
        if not topic:
            raise ValueError("topic must be a non-empty string")

        handlers: List[DomainEventHandler] = []
        with self._lock:
            for pattern, subscribers in self._subscriptions.items():
                if fnmatch.fnmatchcase(topic, pattern):
                    handlers.extend(subscribers)

        if not handlers:
            self._log_debug("No subscribers matched event", topic=topic)

        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - best effort fan-out
                self._log_exception(
                    "Event handler failed",
                    topic=topic,
                    handler=getattr(handler, "__name__", repr(handler)),
                )

        if self._persistence is not None:
            try:
                self._persistence.append(topic, event)
            except Exception:  # noqa: BLE001 - persistence must not break pub/sub
                self._log_exception(
                    "Event persistence failed",
                    topic=topic,
                    persistence=type(self._persistence).__name__,
                )

    def subscribe(self, pattern: str, handler: DomainEventHandler) -> None:
        """Register ``handler`` for topics matching ``pattern``.

        Patterns honour ``fnmatch`` semantics where ``*`` matches any number of
        characters and ``?`` matches exactly one character across topic
        segments (e.g., ``events.scanner.*`` or ``events.?.detected``).
        """
        if not pattern:
            raise ValueError("pattern must be a non-empty string")
        if handler is None:
            raise ValueError("handler must be provided")

        with self._lock:
            self._subscriptions[pattern].append(handler)

    def _log_debug(self, message: str, **kwargs: Any) -> None:
        if hasattr(self._logger, "bind"):
            self._logger.debug(message, **kwargs)
        else:
            if kwargs:
                formatted = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
                message = f"{message} | {formatted}"
            self._logger.debug(message)

    def _log_exception(self, message: str, **kwargs: Any) -> None:
        if hasattr(self._logger, "bind"):
            self._logger.exception(message, **kwargs)
        else:
            if kwargs:
                formatted = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
                message = f"{message} | {formatted}"
            self._logger.exception(message)
