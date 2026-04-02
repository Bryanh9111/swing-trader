"""RunReport data structure, generator, and Markdown renderer.

All data is extracted read-only from orchestrator outputs — no side effects
on strategy, capital, or order logic.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgspec

from common.interface import Result, ResultStatus

__all__ = ["FillSummary", "ReportScanPipeline", "RunReport", "RunReportGenerator", "render_markdown"]


# ---------------------------------------------------------------------------
# Sub-structs
# ---------------------------------------------------------------------------


class ReportMeta(msgspec.Struct, frozen=True, kw_only=True):
    run_id: str
    timestamp_ns: int
    mode: str
    run_type: str
    status: str
    duration_ms: float
    git_sha: str = ""
    degraded_modules: list[str] = msgspec.field(default_factory=list)


class ReportMarket(msgspec.Struct, frozen=True, kw_only=True):
    regime: str = "unknown"
    regime_confidence: float | None = None


class IntentDetail(msgspec.Struct, frozen=True, kw_only=True):
    symbol: str = ""
    qty: int = 0
    entry_price: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None


class OrderDetail(msgspec.Struct, frozen=True, kw_only=True):
    symbol: str = ""
    intent_type: str = ""
    ordered_qty: float = 0.0
    filled_qty: float = 0.0
    price: float = 0.0
    avg_fill_price: float = 0.0
    state: str = ""


class ReportActions(msgspec.Struct, frozen=True, kw_only=True):
    intents_generated: int = 0
    orders_submitted: int = 0
    orders_cancelled: int = 0
    fills_entry_count: int = 0
    fills_exit_count: int = 0
    riskgate_blocks_count: int = 0
    top_block_reasons: dict[str, int] = msgspec.field(default_factory=dict)
    intent_details: list[IntentDetail] = msgspec.field(default_factory=list)
    order_details: list[OrderDetail] = msgspec.field(default_factory=list)


class FillSummary(msgspec.Struct, frozen=True, kw_only=True):
    symbol: str = ""
    side: str = ""
    qty: int = 0
    price: float = 0.0
    reason: str = ""
    pnl: float | None = None


class BracketLeg(msgspec.Struct, frozen=True, kw_only=True):
    price: float = 0.0
    qty: float = 0.0
    status: str = ""


class PositionDetail(msgspec.Struct, frozen=True, kw_only=True):
    symbol: str = ""
    qty: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    sl: BracketLeg | None = None
    tp: BracketLeg | None = None


class ReportPortfolio(msgspec.Struct, frozen=True, kw_only=True):
    cash: float | None = None
    gross_exposure: float | None = None
    net_liquidation: float | None = None
    configured_equity: float | None = None
    buying_power: float | None = None
    cost_basis: float | None = None
    open_positions_count: int | None = None
    positions: list[PositionDetail] = msgspec.field(default_factory=list)
    largest_positions: list[dict[str, Any]] = msgspec.field(default_factory=list)
    realized_pnl_today: float | None = None
    unrealized_pnl: float | None = None
    currency_balances: dict[str, dict[str, float]] = msgspec.field(default_factory=dict)


class ReportTrades(msgspec.Struct, frozen=True, kw_only=True):
    fills: list[FillSummary] = msgspec.field(default_factory=list)
    rejections: list[dict[str, Any]] = msgspec.field(default_factory=list)
    warnings: list[str] = msgspec.field(default_factory=list)


class ReportScanPipeline(msgspec.Struct, frozen=True, kw_only=True):
    universe_total: int | None = None
    universe_filtered: int | None = None
    scanner_scanned: int | None = None
    scanner_detected: int | None = None


class ReportRisk(msgspec.Struct, frozen=True, kw_only=True):
    sl_count_5d: int | None = None
    sl_count_10d: int | None = None
    consecutive_losses: int | None = None
    max_position_utilization: float | None = None
    cash_utilization: float | None = None


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


class RunReport(msgspec.Struct, frozen=True, kw_only=True):
    meta: ReportMeta
    market: ReportMarket = msgspec.field(default_factory=lambda: ReportMarket())
    actions: ReportActions = msgspec.field(default_factory=lambda: ReportActions())
    scan_pipeline: ReportScanPipeline = msgspec.field(default_factory=lambda: ReportScanPipeline())
    portfolio: ReportPortfolio = msgspec.field(default_factory=lambda: ReportPortfolio())
    trades: ReportTrades = msgspec.field(default_factory=lambda: ReportTrades())
    risk: ReportRisk = msgspec.field(default_factory=lambda: ReportRisk())


# ---------------------------------------------------------------------------
# Slot mapping
# ---------------------------------------------------------------------------

_SLOT_MAP: dict[str, str] = {
    "PRE_MARKET_FULL_SCAN": "0900",
    "INTRADAY_CHECK_1030": "1030",
    "INTRADAY_CHECK_1430": "1430",
    "PRE_CLOSE_CLEANUP": "1555",
    "AFTER_MARKET_RECON": "1640",
    "PRE_MARKET_CHECK": "pre_market",
}


def _enum_to_str(val: Any) -> str:
    """Convert an enum value to its string representation, or return as-is if already str."""
    from enum import Enum

    if isinstance(val, Enum):
        return str(val.value) or "UNKNOWN"
    if isinstance(val, str):
        return val or "UNKNOWN"
    return str(getattr(val, "value", None) or "UNKNOWN")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class RunReportGenerator:
    """Extract a RunReport from orchestrator run data (pure read-only)."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        run_id: str,
        metadata: Any,
        module_results: list[Any],
        outputs: dict[str, Any],
        events: list[Any],
    ) -> RunReport:
        meta = self._extract_meta(run_id, metadata, outputs, module_results)
        market = self._extract_market(outputs)
        actions = self._extract_actions(outputs)
        scan_pipeline = self._extract_scan_pipeline(outputs)
        portfolio = self._extract_portfolio(outputs)
        trades = self._extract_trades(outputs)
        risk = ReportRisk()  # V1: all None, rolling window comes later
        return RunReport(
            meta=meta,
            market=market,
            actions=actions,
            scan_pipeline=scan_pipeline,
            portfolio=portfolio,
            trades=trades,
            risk=risk,
        )

    def save(self, report: RunReport) -> Result[Path]:
        """Write summary.json + summary.md under base_dir/<date>/run_<slot>/."""
        try:
            ts = report.meta.timestamp_ns
            dt = datetime.fromtimestamp(ts / 1e9, tz=UTC)
            date_str = dt.strftime("%Y-%m-%d")
            slot = _SLOT_MAP.get(report.meta.run_type, "unknown")
            out_dir = self._base_dir / date_str / f"run_{slot}"
            out_dir.mkdir(parents=True, exist_ok=True)

            json_path = out_dir / "summary.json"
            json_bytes = msgspec.json.encode(report)
            # Pretty-print via stdlib for human readability
            parsed = json.loads(json_bytes)
            json_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))

            md_path = out_dir / "summary.md"
            md_path.write_text(render_markdown(report))

            return Result.success(out_dir)
        except Exception as exc:  # noqa: BLE001
            return Result.failed(exc, "REPORT_SAVE_FAILED")

    # -- private extractors -------------------------------------------------

    @staticmethod
    def _extract_meta(run_id: str, metadata: Any, outputs: dict[str, Any], module_results: list[Any] | None = None) -> ReportMeta:
        start_ns = getattr(metadata, "start_time", 0) or 0
        end_ns = getattr(metadata, "end_time", 0) or 0
        duration_ms = (end_ns - start_ns) / 1e6 if end_ns > start_ns else 0.0
        degraded: list[str] = []
        for mr in (module_results or []):
            status_val = getattr(mr, "status", None)
            if status_val is not None and _enum_to_str(status_val).lower() == "degraded":
                name = getattr(mr, "module_name", "?")
                err = getattr(mr, "error", None)
                reason = getattr(err, "reason_code", None) or (str(err)[:80] if err else "unknown")
                degraded.append(f"{name}: {reason}")
        return ReportMeta(
            run_id=run_id,
            timestamp_ns=end_ns or int(time.time_ns()),
            mode=_enum_to_str(getattr(metadata, "mode", "UNKNOWN")),
            run_type=_enum_to_str(getattr(metadata, "run_type", "UNKNOWN")),
            status=_enum_to_str(getattr(metadata, "status", "unknown") or "unknown"),
            duration_ms=duration_ms,
            git_sha=getattr(metadata, "system_version", "") or "",
            degraded_modules=degraded,
        )

    def _extract_market(self, outputs: dict[str, Any]) -> ReportMarket:
        regime_data = outputs.get("market_regime.json", {})
        if not isinstance(regime_data, dict):
            regime_data = {}
        # Orchestrator stores as "detected_regime"; fallback to "regime" for compat
        regime_val = regime_data.get("detected_regime") or regime_data.get("regime") or "unknown"
        if regime_val == "unknown":
            regime_data = self._load_today_regime_fallback()
            regime_val = regime_data.get("detected_regime") or regime_data.get("regime") or "unknown"
        if not isinstance(regime_data, dict):
            return ReportMarket()
        return ReportMarket(
            regime=regime_val,
            regime_confidence=regime_data.get("confidence"),
        )

    def _load_today_regime_fallback(self) -> dict[str, Any]:
        """Try to load regime from today's PRE_MARKET_FULL_SCAN report (run_0900)."""
        try:
            today_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
            summary_path = self._base_dir / today_str / "run_0900" / "summary.json"
            if summary_path.exists():
                data = json.loads(summary_path.read_text())
                market = data.get("market", {})
                if isinstance(market, dict) and market.get("regime", "unknown") != "unknown":
                    return {"regime": market["regime"], "confidence": market.get("regime_confidence")}
        except Exception:  # noqa: BLE001
            pass
        return {}

    @staticmethod
    def _extract_actions(outputs: dict[str, Any]) -> ReportActions:
        # --- intent count (all strategy-generated) ---
        intents_data = outputs.get("intents.json", {})
        all_groups: list[Any] = []
        if isinstance(intents_data, dict):
            all_groups = intents_data.get("intent_groups", [])
            if not all_groups:
                nested = intents_data.get("intents", {})
                if isinstance(nested, dict):
                    all_groups = nested.get("intent_groups", [])
        intents_count = len(all_groups)

        # --- intent details (only risk-gate-approved, from execution_reports) ---
        exec_data = outputs.get("execution_reports.json", {})
        approved_groups: list[Any] = []
        if isinstance(exec_data, dict):
            embedded = exec_data.get("intents", {})
            if isinstance(embedded, dict):
                approved_groups = embedded.get("intent_groups", [])
        # Fallback: if no execution_reports, use all_groups (backtest mode)
        if not approved_groups:
            approved_groups = all_groups

        details: list[IntentDetail] = []
        for group in approved_groups:
            if not isinstance(group, dict):
                continue
            symbol = group.get("symbol", "")
            intents_list = group.get("intents", [])
            entry_price: float | None = None
            sl_price: float | None = None
            tp_price: float | None = None
            qty = 0
            has_open = False
            for intent in intents_list:
                if not isinstance(intent, dict):
                    continue
                itype = intent.get("intent_type", "")
                if itype in ("OPEN_LONG", "OPEN_SHORT"):
                    has_open = True
                    qty = int(intent.get("quantity", 0))
                    entry_price = intent.get("entry_price")
                elif itype == "STOP_LOSS":
                    sl_price = intent.get("stop_loss_price")
                elif itype == "TAKE_PROFIT":
                    tp_price = intent.get("take_profit_price")
            if has_open:
                details.append(IntentDetail(
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                ))

        # --- Risk decisions ---
        risk_data = outputs.get("risk_decisions.json", {})
        if not isinstance(risk_data, dict):
            # Also check embedded in execution_reports
            risk_data = exec_data.get("risk_decisions", {}) if isinstance(exec_data, dict) else {}
        blocks_count = 0
        block_reasons: dict[str, int] = {}
        if isinstance(risk_data, dict):
            decisions = risk_data.get("decisions", [])
            if isinstance(decisions, list):
                for d in decisions:
                    if not isinstance(d, dict):
                        continue
                    # Support both "action"/"BLOCK" and "decision_type"/"BLOCK"
                    action = d.get("action") or d.get("decision_type", "")
                    if action == "BLOCK":
                        blocks_count += 1
                        codes = d.get("reason_codes", [])
                        reason = codes[0] if codes else d.get("reason", "unknown")
                        block_reasons[reason] = block_reasons.get(reason, 0) + 1

        # Execution reports (exec_data already loaded above)
        reports_list = exec_data.get("reports", []) if isinstance(exec_data, dict) else []
        orders_submitted = 0
        orders_cancelled = 0
        fills_entry = 0
        fills_exit = 0
        if isinstance(reports_list, list):
            for rpt in reports_list:
                if not isinstance(rpt, dict):
                    continue
                status = (rpt.get("final_state") or rpt.get("status") or "").upper()
                if status in ("SUBMITTED", "FILLED", "PARTIAL"):
                    orders_submitted += 1
                if status == "CANCELLED":
                    orders_cancelled += 1
                filled_qty = rpt.get("filled_quantity") or rpt.get("filled_qty") or 0
                if filled_qty and float(filled_qty) > 0:
                    intent_type = (rpt.get("intent_id") or "").upper()
                    reason = (rpt.get("reason") or "").lower()
                    if "OPEN_LONG" in intent_type or "OPEN_SHORT" in intent_type or reason == "entry":
                        fills_entry += 1
                    elif "STOP_LOSS" in intent_type or "TAKE_PROFIT" in intent_type or reason == "exit":
                        fills_exit += 1

        # Fallback: execution-only runs embed cumulative stats via order_stats.
        od_list: list[Any] = []
        if not intents_count and not orders_submitted:
            stats = exec_data.get("order_stats") if isinstance(exec_data, dict) else None
            if isinstance(stats, dict):
                intents_count = stats.get("intents_generated", 0)
                orders_submitted = stats.get("orders_submitted", 0)
                orders_cancelled = stats.get("orders_cancelled", 0)
                fills_entry = stats.get("fills_entry_count", 0)
                fills_exit = stats.get("fills_exit_count", 0)
                od_list = stats.get("order_details", [])

        from reporting.run_report import OrderDetail  # local to avoid circular at module level

        order_details_out: list[OrderDetail] = []
        if isinstance(od_list, list):
            for od in od_list:
                if not isinstance(od, dict):
                    continue
                order_details_out.append(OrderDetail(
                    symbol=od.get("symbol", ""),
                    intent_type=od.get("intent_type", ""),
                    ordered_qty=float(od.get("ordered_qty", 0) or 0),
                    filled_qty=float(od.get("filled_qty", 0) or 0),
                    price=float(od.get("price", 0) or 0),
                    avg_fill_price=float(od.get("avg_fill_price", 0) or 0),
                    state=od.get("state", ""),
                ))

        return ReportActions(
            intents_generated=intents_count,
            orders_submitted=orders_submitted,
            orders_cancelled=orders_cancelled,
            fills_entry_count=fills_entry,
            fills_exit_count=fills_exit,
            riskgate_blocks_count=blocks_count,
            intent_details=details,
            top_block_reasons=block_reasons,
            order_details=order_details_out,
        )

    @staticmethod
    def _extract_scan_pipeline(outputs: dict[str, Any]) -> ReportScanPipeline:
        uni = outputs.get("universe.json", {})
        uni = uni if isinstance(uni, dict) else {}
        cand = outputs.get("candidates.json", {})
        cand = cand if isinstance(cand, dict) else {}
        return ReportScanPipeline(
            universe_total=uni.get("total_candidates"),
            universe_filtered=uni.get("total_filtered"),
            scanner_scanned=cand.get("total_scanned"),
            scanner_detected=cand.get("total_detected"),
        )

    @staticmethod
    def _extract_portfolio(outputs: dict[str, Any]) -> ReportPortfolio:
        exec_data = outputs.get("execution_reports.json", {})
        if not isinstance(exec_data, dict):
            return ReportPortfolio()
        acct = exec_data.get("account_summary", {})
        if not isinstance(acct, dict):
            acct = {}
        configured_equity = outputs.get("_configured_equity")

        bracket_orders = acct.get("bracket_orders", {})
        if not isinstance(bracket_orders, dict):
            bracket_orders = {}

        pos_details: list[PositionDetail] = []
        for raw_pos in acct.get("positions", []):
            if not isinstance(raw_pos, dict):
                continue
            qty = abs(raw_pos.get("qty") or raw_pos.get("position") or 0.0)
            if qty <= 0:
                continue
            mv = raw_pos.get("market_value") or 0.0
            current_price = mv / qty if qty else 0.0
            symbol = raw_pos.get("symbol", "")

            sl_leg: BracketLeg | None = None
            tp_leg: BracketLeg | None = None
            sym_brackets = bracket_orders.get(symbol, {})
            if isinstance(sym_brackets, dict):
                sl_raw = sym_brackets.get("sl")
                if isinstance(sl_raw, dict):
                    sl_leg = BracketLeg(
                        price=float(sl_raw.get("price", 0) or 0),
                        qty=float(sl_raw.get("qty", 0) or 0),
                        status=str(sl_raw.get("status", "")),
                    )
                tp_raw = sym_brackets.get("tp")
                if isinstance(tp_raw, dict):
                    tp_leg = BracketLeg(
                        price=float(tp_raw.get("price", 0) or 0),
                        qty=float(tp_raw.get("qty", 0) or 0),
                        status=str(tp_raw.get("status", "")),
                    )

            pos_details.append(PositionDetail(
                symbol=symbol,
                qty=qty,
                avg_cost=raw_pos.get("avg_cost", 0.0),
                current_price=current_price,
                unrealized_pnl=raw_pos.get("unrealized_pnl", 0.0),
                sl=sl_leg,
                tp=tp_leg,
            ))

        raw_cb = acct.get("currency_balances")
        currency_balances: dict[str, dict[str, float]] = {}
        if isinstance(raw_cb, dict):
            currency_balances = raw_cb

        return ReportPortfolio(
            cash=acct.get("cash"),
            gross_exposure=acct.get("gross_exposure"),
            net_liquidation=acct.get("net_liquidation"),
            configured_equity=float(configured_equity) if configured_equity is not None else None,
            buying_power=acct.get("buying_power"),
            cost_basis=acct.get("cost_basis"),
            open_positions_count=acct.get("open_positions_count"),
            positions=pos_details,
            realized_pnl_today=acct.get("realized_pnl_today"),
            unrealized_pnl=acct.get("unrealized_pnl"),
            currency_balances=currency_balances,
        )

    @staticmethod
    def _extract_trades(outputs: dict[str, Any]) -> ReportTrades:
        exec_data = outputs.get("execution_reports.json", {})
        reports_list = exec_data.get("reports", []) if isinstance(exec_data, dict) else []
        fills: list[FillSummary] = []
        warnings: list[str] = []
        if isinstance(reports_list, list):
            for rpt in reports_list:
                if not isinstance(rpt, dict):
                    continue
                filled_qty = rpt.get("filled_qty", 0)
                if filled_qty and filled_qty > 0:
                    fills.append(
                        FillSummary(
                            symbol=rpt.get("symbol", ""),
                            side=rpt.get("side", ""),
                            qty=int(filled_qty),
                            price=float(rpt.get("avg_fill_price", 0.0) or 0.0),
                            reason=rpt.get("reason", ""),
                            pnl=rpt.get("realized_pnl"),
                        )
                    )
                warn = rpt.get("warning")
                if warn:
                    warnings.append(str(warn))
        return ReportTrades(fills=fills, warnings=warnings)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_markdown(report: RunReport) -> str:
    """Render a RunReport as human-readable Markdown (pure string ops, no jinja2)."""
    m = report.meta
    dt = datetime.fromtimestamp(m.timestamp_ns / 1e9, tz=UTC) if m.timestamp_ns else None
    date_str = dt.strftime("%Y-%m-%d") if dt else "N/A"
    time_str = dt.strftime("%H:%M:%S UTC") if dt else "N/A"
    slot = _SLOT_MAP.get(m.run_type, m.run_type)

    lines: list[str] = []
    lines.append(f"# AST Run Report: {m.run_type} {date_str} {slot}")
    lines.append("")

    # Meta
    lines.append("## Meta")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Run ID | `{m.run_id}` |")
    lines.append(f"| Mode | {m.mode} |")
    lines.append(f"| Run Type | {m.run_type} |")
    lines.append(f"| Status | {m.status} |")
    lines.append(f"| Time | {time_str} |")
    lines.append(f"| Duration | {m.duration_ms:.1f} ms |")
    if m.git_sha:
        lines.append(f"| Git SHA | `{m.git_sha[:8]}` |")
    if m.degraded_modules:
        lines.append(f"| Degraded | {', '.join(m.degraded_modules)} |")
    lines.append("")

    # Market
    mk = report.market
    lines.append("## Market")
    lines.append(f"- Regime: **{mk.regime}**")
    if mk.regime_confidence is not None:
        lines.append(f"- Confidence: {mk.regime_confidence:.2f}")
    lines.append("")

    # Scan Pipeline
    sp = report.scan_pipeline
    lines.append("## Scan Pipeline")
    if sp.universe_filtered is not None:
        lines.append(f"- Universe filtered: {sp.universe_filtered} (from {sp.universe_total or '?'})")
    if sp.scanner_scanned is not None:
        lines.append(f"- Scanner scanned: {sp.scanner_scanned}")
    if sp.scanner_detected is not None:
        lines.append(f"- Candidates detected: {sp.scanner_detected}")
    if all(v is None for v in [sp.universe_filtered, sp.scanner_scanned, sp.scanner_detected]):
        lines.append("(no scan pipeline data)")
    lines.append("")

    # Actions
    a = report.actions
    lines.append("## Actions")
    lines.append(f"- Intents generated: {a.intents_generated}")
    lines.append(f"- Orders submitted: {a.orders_submitted}")
    lines.append(f"- Orders cancelled: {a.orders_cancelled}")
    lines.append(f"- Fills (entry): {a.fills_entry_count}")
    lines.append(f"- Fills (exit): {a.fills_exit_count}")
    lines.append(f"- Risk gate blocks: {a.riskgate_blocks_count}")
    if a.top_block_reasons:
        lines.append("- Block reasons:")
        for reason, count in sorted(a.top_block_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"  - {reason}: {count}")
    lines.append("")

    # Portfolio
    p = report.portfolio
    lines.append("## Portfolio")
    if p.net_liquidation is not None:
        lines.append(f"- IBKR Net Liquidation: ${p.net_liquidation:,.2f}")
    if p.configured_equity is not None:
        lines.append(f"- Algo Equity (configured): ${p.configured_equity:,.2f}")
    if p.gross_exposure is not None:
        lines.append(f"- Gross Exposure: ${p.gross_exposure:,.2f}")
    if p.buying_power is not None:
        lines.append(f"- Buying Power: ${p.buying_power:,.2f}")
    if p.cash is not None:
        lines.append(f"- Cash: ${p.cash:,.2f}")
    if p.open_positions_count is not None:
        lines.append(f"- Open Positions: {p.open_positions_count}")
    if p.realized_pnl_today is not None:
        lines.append(f"- Realized PnL Today: ${p.realized_pnl_today:,.2f}")
    if p.unrealized_pnl is not None:
        lines.append(f"- Unrealized PnL: ${p.unrealized_pnl:,.2f}")
    if all(v is None for v in [p.net_liquidation, p.gross_exposure, p.open_positions_count]):
        lines.append("- (no portfolio data available)")
    lines.append("")

    # Fills
    lines.append("## Fills")
    fills = report.trades.fills
    if fills:
        lines.append("| Symbol | Side | Qty | Price | Reason | PnL |")
        lines.append("|--------|------|-----|-------|--------|-----|")
        for f in fills:
            pnl_str = f"${f.pnl:,.2f}" if f.pnl is not None else "-"
            lines.append(f"| {f.symbol} | {f.side} | {f.qty} | ${f.price:,.2f} | {f.reason} | {pnl_str} |")
    else:
        lines.append("(no fills)")
    if report.trades.warnings:
        lines.append("")
        lines.append("**Warnings:**")
        for w in report.trades.warnings:
            lines.append(f"- {w}")
    lines.append("")

    # Risk diagnostics
    r = report.risk
    lines.append("## Risk Diagnostics")
    if r.sl_count_5d is not None:
        lines.append(f"- SL count (5d): {r.sl_count_5d}")
    if r.sl_count_10d is not None:
        lines.append(f"- SL count (10d): {r.sl_count_10d}")
    if r.consecutive_losses is not None:
        lines.append(f"- Consecutive losses: {r.consecutive_losses}")
    if r.max_position_utilization is not None:
        lines.append(f"- Max position utilization: {r.max_position_utilization:.1%}")
    if r.cash_utilization is not None:
        lines.append(f"- Cash utilization: {r.cash_utilization:.1%}")
    if all(
        v is None
        for v in [r.sl_count_5d, r.sl_count_10d, r.consecutive_losses, r.max_position_utilization, r.cash_utilization]
    ):
        lines.append("(V1: rolling window not yet implemented)")
    lines.append("")

    return "\n".join(lines)
