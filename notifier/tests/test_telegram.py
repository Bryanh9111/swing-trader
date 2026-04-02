"""Tests for TelegramNotifier — send, rate limit, dedup, silent hours, P0 alerts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.interface import ResultStatus
from notifier.telegram import TelegramNotifier
from reporting.run_report import (
    FillSummary,
    ReportActions,
    ReportMarket,
    ReportMeta,
    ReportPortfolio,
    ReportRisk,
    ReportTrades,
    RunReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def notifier():
    """A notifier with fake credentials, enabled."""
    return TelegramNotifier(bot_token="fake-token", chat_id="fake-chat", enabled=True)


def _make_report(
    *,
    status: str = "completed",
    fills: list[FillSummary] | None = None,
    intents: int = 2,
    cash: float | None = 48200.0,
) -> RunReport:
    meta = ReportMeta(
        run_id="run-1",
        timestamp_ns=1_707_700_800_000_000_000,
        mode="PAPER",
        run_type="PRE_MARKET_FULL_SCAN",
        status=status,
        duration_ms=12300.0,
        git_sha="abc1234",
    )
    return RunReport(
        meta=meta,
        market=ReportMarket(regime="TREND", regime_confidence=0.85),
        actions=ReportActions(
            intents_generated=intents,
            orders_submitted=2,
            fills_entry_count=len(fills) if fills else 1,
        ),
        portfolio=ReportPortfolio(cash=cash, gross_exposure=51800.0),
        trades=ReportTrades(fills=fills or [FillSummary(symbol="AAPL", side="BUY", qty=50, price=185.20, reason="entry")]),
        risk=ReportRisk(),
    )


# ---------------------------------------------------------------------------
# Test: send()
# ---------------------------------------------------------------------------


class TestSend:
    @patch("notifier.telegram.urllib.request.urlopen")
    def test_send_posts_correct_params(self, mock_urlopen, notifier: TelegramNotifier):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=b'{"ok":true}')
        mock_urlopen.return_value = mock_resp

        result = notifier.send("Hello test")
        assert result.status is ResultStatus.SUCCESS

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "fake-token" in req.full_url
        body = json.loads(req.data)
        assert body["chat_id"] == "fake-chat"
        assert body["text"] == "Hello test"
        assert "parse_mode" not in body  # default is plain text

    def test_disabled_returns_success(self):
        n = TelegramNotifier(bot_token="t", chat_id="c", enabled=False)
        result = n.send("test")
        assert result.status is ResultStatus.SUCCESS

    def test_missing_token_not_enabled(self):
        n = TelegramNotifier(bot_token="", chat_id="c", enabled=True)
        assert not n.enabled
        result = n.send("test")
        assert result.status is ResultStatus.SUCCESS


class TestRateLimit:
    def test_rate_limit_exceeded(self, notifier: TelegramNotifier):
        notifier._rate_limit = 2
        # Simulate 2 recent sends
        now = time.monotonic()
        notifier._send_times.extend([now, now])

        result = notifier.send("over limit")
        assert result.status is ResultStatus.DEGRADED
        assert result.reason_code == "RATE_LIMITED"


class TestSilentHours:
    @patch("notifier.telegram.datetime")
    def test_silent_hours_skips(self, mock_dt):
        mock_now = MagicMock()
        mock_now.hour = 3
        mock_dt.now.return_value = mock_now
        # We need to let fromtimestamp work normally for other calls
        mock_dt.side_effect = None

        n = TelegramNotifier(bot_token="t", chat_id="c", silent_hours="2-6")
        # Manually set _silent_hours since the mock might interfere
        n._silent_hours = (2, 6)

        # Override _in_silent_hours directly for a clean test
        n._in_silent_hours = lambda: True

        result = n.send("during silent")
        assert result.status is ResultStatus.SUCCESS


class TestDedup:
    @patch("notifier.telegram.urllib.request.urlopen")
    def test_dedup_within_ttl(self, mock_urlopen, notifier: TelegramNotifier):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=b'{"ok":true}')
        mock_urlopen.return_value = mock_resp

        report = _make_report()
        # First call should send
        r1 = notifier.send_fills_summary(report)
        assert r1.status is ResultStatus.SUCCESS
        assert mock_urlopen.call_count == 1

        # Second call within TTL should be deduped
        r2 = notifier.send_fills_summary(report)
        assert r2.status is ResultStatus.SUCCESS
        assert mock_urlopen.call_count == 1  # NOT called again


# ---------------------------------------------------------------------------
# Test: send_run_summary()
# ---------------------------------------------------------------------------


class TestSendRunSummary:
    @patch("notifier.telegram.urllib.request.urlopen")
    def test_format_valid(self, mock_urlopen, notifier: TelegramNotifier):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=b'{"ok":true}')
        mock_urlopen.return_value = mock_resp

        report = _make_report()
        result = notifier.send_run_summary(report)
        assert result.status is ResultStatus.SUCCESS

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        text = body["text"]
        assert "[AST][PAPER]" in text
        assert "PRE_MARKET_FULL_SCAN" in text
        assert "COMPLETED" in text
        assert len(text) <= 4096

    @patch("notifier.telegram.urllib.request.urlopen")
    def test_includes_fills(self, mock_urlopen, notifier: TelegramNotifier):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=b'{"ok":true}')
        mock_urlopen.return_value = mock_resp

        fills = [FillSummary(symbol="MSFT", side="SELL", qty=30, price=410.5, reason="exit", pnl=250.0)]
        report = _make_report(fills=fills)
        notifier.send_run_summary(report)

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert "MSFT" in body["text"]


# ---------------------------------------------------------------------------
# Test: send_p0_alert()
# ---------------------------------------------------------------------------


class TestSendP0Alert:
    @patch("notifier.telegram.urllib.request.urlopen")
    def test_p0_contains_error_and_log(self, mock_urlopen, notifier: TelegramNotifier):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=b'{"ok":true}')
        mock_urlopen.return_value = mock_resp

        result = notifier.send_p0_alert(
            run_id="run-fail",
            stage="orchestrator",
            error="Connection refused",
            log_tail="Traceback: ...",
        )
        assert result.status is ResultStatus.SUCCESS

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        text = body["text"]
        assert "[P0]" in text
        assert "Connection refused" in text
        assert "Traceback: ..." in text
