"""Telegram push notifier for AST run events.

Zero new dependencies — uses only urllib.request + json from stdlib.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import UTC, datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any

from common.interface import Result, ResultStatus

if TYPE_CHECKING:
    from reporting.run_report import RunReport

__all__ = ["TelegramNotifier"]

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_MESSAGE_LEN = 4096


class TelegramNotifier:
    """Send concise Telegram messages for P0/P1 events.

    Gracefully degrades when token/chat_id are missing or rate-limited.
    """

    def __init__(
        self,
        *,
        bot_token: str | None = None,
        chat_id: str | None = None,
        enabled: bool = True,
        rate_limit_per_min: int = 20,
        silent_hours: str | None = None,
    ) -> None:
        self._token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        env_enabled = os.getenv("TELEGRAM_ENABLED", "").lower()
        if env_enabled in ("0", "false", "no"):
            enabled = False
        self.enabled = enabled and bool(self._token) and bool(self._chat_id)
        self._rate_limit = rate_limit_per_min
        self._send_times: deque[float] = deque()
        self._silent_hours = self._parse_silent_hours(silent_hours)
        self._dedup_cache: dict[str, float] = {}
        self._dedup_ttl = 600.0  # 10 minutes

    # -- public API ---------------------------------------------------------

    def send(self, text: str, *, parse_mode: str = "") -> Result[None]:
        if not self.enabled:
            return Result.success(None)

        if self._in_silent_hours():
            return Result.success(None)

        if not self._check_rate_limit():
            return Result(
                status=ResultStatus.DEGRADED,
                data=None,
                reason_code="RATE_LIMITED",
            )

        text = text[:_MAX_MESSAGE_LEN]
        body: dict[str, str] = {"chat_id": self._chat_id, "text": text}
        if parse_mode:
            body["parse_mode"] = parse_mode
        payload = json.dumps(body).encode("utf-8")

        url = _TELEGRAM_API.format(token=self._token)
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                resp.read()
            self._send_times.append(time.monotonic())
            return Result.success(None)
        except (urllib.error.URLError, OSError) as exc:
            return Result(
                status=ResultStatus.DEGRADED,
                data=None,
                error=exc,
                reason_code="SEND_FAILED",
            )

    def send_run_summary(self, report: RunReport) -> Result[None]:
        m = report.meta
        _ET = timezone(timedelta(hours=-5), "ET")
        dt = datetime.fromtimestamp(m.timestamp_ns / 1e9, tz=UTC).astimezone(_ET) if m.timestamp_ns else None
        date_str = dt.strftime("%Y-%m-%d") if dt else "N/A"
        time_str = dt.strftime("%H:%M ET") if dt else "N/A"

        status_icon = "✅" if m.status in ("completed", "success") else "⚠️" if m.status == "degraded" else "❌"
        duration_s = m.duration_ms / 1000.0

        a = report.actions
        p = report.portfolio
        mk = report.market
        sp = report.scan_pipeline

        lines = [
            f"[AST][{m.mode}] {m.run_type} {date_str} {time_str}",
            "━━━━━━━━━━━━━━",
            f"{status_icon} Status: {m.status.upper()} ({duration_s:.1f}s)",
        ]
        if m.degraded_modules:
            for dm in m.degraded_modules:
                lines.append(f"   └ {dm}")

        # Market regime
        regime_str = mk.regime.upper()
        if mk.regime_confidence is not None:
            regime_str += f" ({mk.regime_confidence:.0%})"
        lines.append(f"🌐 Regime: {regime_str}")

        # Universe / Scanner pipeline
        if sp.universe_filtered is not None or sp.scanner_detected is not None:
            # Use universe_filtered as the canonical count; scanner_scanned includes
            # injected regime symbols (SPY/QQQ/VIXY/XLK/SOXX) which inflates the number.
            uni_str = str(sp.universe_filtered) if sp.universe_filtered is not None else "?"
            detected_str = str(sp.scanner_detected) if sp.scanner_detected is not None else "0"
            lines.append(f"🔍 Universe: {uni_str} | Candidates: {detected_str}")

        # Actions
        approved = len(a.intent_details)
        if approved:
            intent_str = f"{approved}/{a.intents_generated}"
        else:
            intent_str = str(a.intents_generated)
        lines.append(f"📊 Intents: {intent_str} | Orders: {a.orders_submitted} | Fills: {a.fills_entry_count + a.fills_exit_count}")
        # Full scan: show intent details (symbol + entry/SL/TP prices)
        for d in a.intent_details:
            ep = f"@${d.entry_price:,.2f}" if d.entry_price is not None else "@MKT"
            sl = f" SL${d.stop_loss_price:,.2f}" if d.stop_loss_price is not None else ""
            tp = f" TP${d.take_profit_price:,.2f}" if d.take_profit_price is not None else ""
            lines.append(f"   └ {d.symbol} {d.qty} shares {ep}{sl}{tp}")
        # Execution-only runs: show unique entry symbols (compact)
        if not approved and a.order_details:
            entry_syms = sorted({od.symbol for od in a.order_details if od.intent_type in ("OPEN_LONG", "OPEN_SHORT")})
            if entry_syms:
                lines.append(f"   └ Symbols: {', '.join(entry_syms)}")
        if a.orders_cancelled:
            lines.append(f"🗑 Cancelled: {a.orders_cancelled}")

        # Portfolio
        nlv_str = f"${p.net_liquidation:,.0f}" if p.net_liquidation is not None else "N/A"
        equity_str = f"${p.configured_equity:,.0f}" if p.configured_equity is not None else "N/A"
        exposure_str = f"${p.gross_exposure:,.0f}" if p.gross_exposure is not None else "N/A"
        if p.configured_equity is not None:
            cost = p.cost_basis if p.cost_basis is not None else 0.0
            algo_bp = p.configured_equity - cost
            usd_cash = p.currency_balances.get("USD", {}).get("cash") if p.currency_balances else None
            if usd_cash is not None:
                algo_bp = min(algo_bp, usd_cash)
            bp_str = f"${algo_bp:,.0f}"
        else:
            bp_str = "N/A"
        pos_count = p.open_positions_count if p.open_positions_count is not None else 0
        lines.append(f"🏦 IBKR NLV: {nlv_str}")
        if p.currency_balances:
            bal_parts = []
            for cur in sorted(p.currency_balances):
                cb = p.currency_balances[cur]
                cash_val = cb.get("cash", 0)
                bal_parts.append(f"{cur} ${cash_val:,.0f}")
            if bal_parts:
                lines.append(f"   💱 {' | '.join(bal_parts)}")
        lines.append(f"💰 Equity: {equity_str} | BP: {bp_str} | Pos: {exposure_str} ({pos_count})")

        # PnL %
        if p.configured_equity is not None and p.cost_basis is not None and p.gross_exposure is not None:
            unrealized = p.gross_exposure - p.cost_basis
            pnl_pct = unrealized / p.configured_equity * 100
            sign = "+" if pnl_pct >= 0 else ""
            lines.append(f"📈 PnL: {sign}{pnl_pct:.2f}% (${unrealized:,.0f})")

        # Positions detail
        for pos in p.positions:
            pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
            lines.append(
                f"   📌 {pos.symbol} {pos.qty:.0f} shares"
                f" avg${pos.avg_cost:,.2f} → ${pos.current_price:,.2f}"
                f" ({pnl_sign}${pos.unrealized_pnl:,.0f})"
            )
            bracket_parts: list[str] = []
            if pos.sl is not None and pos.sl.price > 0:
                bracket_parts.append(f"SL @${pos.sl.price:,.2f} x{pos.sl.qty:.0f}")
            if pos.tp is not None and pos.tp.price > 0:
                bracket_parts.append(f"TP @${pos.tp.price:,.2f} x{pos.tp.qty:.0f}")
            if bracket_parts:
                lines.append(f"      {' | '.join(bracket_parts)}")

        # Fills (max 5)
        fills = report.trades.fills[:5] if report.trades and report.trades.fills else []
        for f in fills:
            pnl_str = f" PnL ${f.pnl:,.0f}" if f.pnl is not None else ""
            lines.append(f"📝 {f.side} {f.symbol} {f.qty}@${f.price:,.2f}{pnl_str}")

        # Risk gate blocks with reasons
        if a.riskgate_blocks_count:
            lines.append(f"🚫 Blocked: {a.riskgate_blocks_count}")
            for reason, count in sorted(a.top_block_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"   └ {reason}: {count}")

        lines.append("━━━━━━━━━━━━━━")

        return self.send("\n".join(lines))

    def send_fills_summary(self, report: RunReport) -> Result[None]:
        fills = report.trades.fills if report.trades else []
        if not fills:
            return Result.success(None)

        # Dedup: only send if we haven't sent these fills recently
        dedup_key = ":".join(f"{f.symbol}:{f.side}" for f in fills)
        if self._is_deduped(dedup_key):
            return Result.success(None)

        lines = [f"📋 Fills Detail ({len(fills)} total)"]
        for f in fills:
            pnl_str = f" → ${f.pnl:,.2f}" if f.pnl is not None else ""
            lines.append(f"  {f.side} {f.symbol} {f.qty}@${f.price:,.2f}{pnl_str}")

        return self.send("\n".join(lines))

    def send_p0_alert(
        self,
        run_id: str,
        stage: str,
        error: str,
        log_tail: str = "",
    ) -> Result[None]:
        lines = [
            "[AST][P0] 🚨 RUN FAILED",
            "━━━━━━━━━━━━━━",
            f"Run: {run_id}",
            f"Stage: {stage}",
            f"Error: {error[:500]}",
        ]
        if log_tail:
            lines.append(f"\n```\n{log_tail[:800]}\n```")
        lines.append("━━━━━━━━━━━━━━")
        return self.send("\n".join(lines))

    # -- private helpers ----------------------------------------------------

    def _check_rate_limit(self) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        while self._send_times and self._send_times[0] < cutoff:
            self._send_times.popleft()
        return len(self._send_times) < self._rate_limit

    def _in_silent_hours(self) -> bool:
        if self._silent_hours is None:
            return False
        start_h, end_h = self._silent_hours
        current_h = datetime.now(tz=UTC).hour
        if start_h <= end_h:
            return start_h <= current_h < end_h
        return current_h >= start_h or current_h < end_h

    def _is_deduped(self, key: str) -> bool:
        now = time.monotonic()
        # Prune expired entries
        expired = [k for k, t in self._dedup_cache.items() if now - t > self._dedup_ttl]
        for k in expired:
            del self._dedup_cache[k]

        if key in self._dedup_cache:
            return True
        self._dedup_cache[key] = now
        return False

    @staticmethod
    def _parse_silent_hours(spec: str | None) -> tuple[int, int] | None:
        if not spec:
            return None
        try:
            parts = spec.split("-")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass
        return None
