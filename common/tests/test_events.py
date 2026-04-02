from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, List

import logging
import pytest

from common.events import EventPersistence, InMemoryEventBus
from common.interface import DomainEvent


@pytest.fixture
def sample_event() -> DomainEvent:
    return DomainEvent(
        event_id="evt-1",
        event_type="test-event",
        run_id="run-1",
        module="tests.events",
        timestamp_ns=1,
        data={"value": 42},
    )


def test_publish_subscribe_delivers_event(sample_event: DomainEvent) -> None:
    bus = InMemoryEventBus()
    received: list[DomainEvent] = []

    def handler(event: DomainEvent) -> None:
        received.append(event)

    bus.subscribe("events.detected", handler)
    bus.publish("events.detected", sample_event)

    assert received == [sample_event]


def test_wildcard_matching_supports_star_and_question(sample_event: DomainEvent) -> None:
    bus = InMemoryEventBus()
    received: list[str] = []

    bus.subscribe("scanner.*", lambda event: received.append(f"star:{event.event_type}"))
    bus.subscribe("scanner.?", lambda event: received.append(f"single:{event.event_type}"))

    bus.publish("scanner.A", sample_event)
    bus.publish("scanner.signal", sample_event)

    assert received.count("single:test-event") == 1
    assert received.count("star:test-event") == 2


def test_subscriber_error_isolated(caplog: pytest.LogCaptureFixture, sample_event: DomainEvent) -> None:
    caplog.set_level(logging.ERROR)
    bus = InMemoryEventBus()
    calls: list[str] = []

    def bad_handler(_: DomainEvent) -> None:
        raise RuntimeError("handler failure")

    def good_handler(event: DomainEvent) -> None:
        calls.append(event.event_id)

    bus.subscribe("events.test", bad_handler)
    bus.subscribe("events.test", good_handler)
    bus.publish("events.test", sample_event)

    assert calls == [sample_event.event_id]
    assert any("Event handler failed" in record.message for record in caplog.records)


def test_event_persistence_called(sample_event: DomainEvent) -> None:
    class Recorder(EventPersistence):
        def __init__(self) -> None:
            self.calls: list[tuple[str, DomainEvent]] = []

        def append(self, topic: str, event: DomainEvent) -> None:
            self.calls.append((topic, event))

    recorder = Recorder()
    bus = InMemoryEventBus(persistence=recorder)
    bus.subscribe("events.persist", lambda _: None)
    bus.publish("events.persist", sample_event)

    assert recorder.calls == [("events.persist", sample_event)]


def test_persistence_failure_logged(caplog: pytest.LogCaptureFixture, sample_event: DomainEvent) -> None:
    caplog.set_level(logging.ERROR)

    class FailingPersistence(EventPersistence):
        def append(self, topic: str, event: DomainEvent) -> None:
            raise RuntimeError("persistence down")

    bus = InMemoryEventBus(persistence=FailingPersistence())
    bus.publish("events.persist", sample_event)

    assert any("Event persistence failed" in record.message for record in caplog.records)


def test_publish_no_subscribers_logs_debug(caplog: pytest.LogCaptureFixture, sample_event: DomainEvent) -> None:
    caplog.set_level(logging.DEBUG)
    bus = InMemoryEventBus()
    bus.publish("events.empty", sample_event)

    assert any("No subscribers matched event" in record.message for record in caplog.records)


def test_subscribe_validates_inputs(sample_event: DomainEvent) -> None:
    bus = InMemoryEventBus()

    with pytest.raises(ValueError):
        bus.subscribe("", lambda _: None)

    with pytest.raises(ValueError):
        bus.subscribe("events.invalid", None)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        bus.publish("", sample_event)


def test_multiple_subscribers_receive(sample_event: DomainEvent) -> None:
    bus = InMemoryEventBus()
    received: List[str] = []

    bus.subscribe("events.multi", lambda event: received.append(f"first:{event.event_id}"))
    bus.subscribe("events.multi", lambda event: received.append(f"second:{event.event_id}"))

    bus.publish("events.multi", sample_event)

    assert received == [f"first:{sample_event.event_id}", f"second:{sample_event.event_id}"]


def test_thread_safety_for_publish(sample_event: DomainEvent) -> None:
    bus = InMemoryEventBus()
    lock = Lock()
    counter = Counter()

    def handler(event: DomainEvent) -> None:
        with lock:
            counter.update([event.event_id])

    bus.subscribe("events.thread", handler)

    def _publish(_: int) -> None:
        bus.publish("events.thread", sample_event)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_publish, range(50)))

    assert counter[sample_event.event_id] == 50


def test_structlog_logger_branches(sample_event: DomainEvent) -> None:
    class FakeBoundLogger:
        def __init__(self) -> None:
            self.debug_calls: list[tuple[str, dict[str, Any]]] = []
            self.exception_calls: list[tuple[str, dict[str, Any]]] = []

        def bind(self, **_: Any) -> "FakeBoundLogger":
            return self

        def debug(self, message: str, **kwargs: Any) -> None:
            self.debug_calls.append((message, kwargs))

        def exception(self, message: str, **kwargs: Any) -> None:
            self.exception_calls.append((message, kwargs))

    logger = FakeBoundLogger()
    bus = InMemoryEventBus(logger=logger)

    bus.publish("events.miss", sample_event)
    assert logger.debug_calls

    def raising(_: DomainEvent) -> None:
        raise RuntimeError("boom")

    bus.subscribe("events.fail", raising)
    bus.publish("events.fail", sample_event)
    assert logger.exception_calls
