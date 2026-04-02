# Automated Swing Trader (AST)

A modular, production-grade framework for building automated swing trading systems on U.S. equity markets, integrated with Interactive Brokers.

## What This Is

AST provides the **infrastructure** for automated trading — not the alpha. The framework ships with a simple MA Crossover demo strategy to exercise the full pipeline. You bring your own pattern detectors, sizing logic, and parameters.

**Architecture**: Modular monolith with strict module boundaries, plugin lifecycle, event-driven logging, and deterministic replay.

**Core design principle**: When in doubt, do nothing. Capital preservation outweighs opportunity cost.

## Architecture

```
                          +------------------+
                          |   Orchestrator   |  Pipeline coordinator
                          |  (eod_scan.py)   |  Plugin lifecycle management
                          +--------+---------+
                                   |
          +------------------------+------------------------+
          |                        |                        |
          v                        v                        v
  +-------+-------+    +-----------+---------+    +---------+---------+
  | Universe      |    | Data Layer          |    | Market Regime     |
  | NYSE/NASDAQ   |--->| Polygon + S3 hybrid |--->| Bull/Bear/Choppy  |
  | equity filter |    | SQLite cache        |    | ADX + VIX detect  |
  +-------+-------+    +-----------+---------+    +---------+---------+
          |                        |                        |
          v                        v                        v
  +-------+-------+    +-----------+---------+    +---------+---------+
  | Scanner       |    | Event Guard         |    | Indicators        |
  | Pattern detect|--->| Earnings/splits     |    | ATR, RSI, MACD    |
  | Filter chain  |    | exclusion windows   |    | Bollinger, KDJ    |
  +-------+-------+    +-----------+---------+    +---------+---------+
          |                        |
          v                        v
  +-------+------------------------+---------+
  |            Strategy Engine               |
  |  Candidate -> IntentGroup (OTO bracket)  |
  |  Position sizing + ATR-based pricing     |
  +--------------------+---------------------+
                       |
                       v
  +--------------------+---------------------+
  |              Risk Gate                   |
  |  Leverage / Drawdown / Concentration     |
  |  Safe Mode: ACTIVE -> REDUCING -> HALTED |
  +--------------------+---------------------+
                       |
                       v
  +--------------------+---------------------+
  |             Execution Layer              |
  |  IBKR adapter (ib_insync)               |
  |  OCA bracket groups, partial fill align  |
  |  Order State Machine (intent->fill)      |
  +--------------------+---------------------+
                       |
          +------------+------------+
          |            |            |
          v            v            v
  +-------+--+  +-----+----+  +---+--------+
  | Journal   |  | Reporting|  | Notifier   |
  | Snapshot  |  | JSON + MD|  | Telegram   |
  | Replay    |  | per-run  |  | P0/P1      |
  +-----------+  +----------+  +------------+
```

### Module Responsibilities

| Module | Input | Output | Role |
|--------|-------|--------|------|
| **Universe** | Polygon API | `UniverseSnapshot` | Filter tradable equities by exchange, price, liquidity, market cap |
| **Data** | Universe symbols | `PriceSeriesSnapshot` | Fetch OHLCV bars via S3 (historical) + Polygon (current year), SQLite cache |
| **Scanner** | Price bars | `CandidateSet` | Detect patterns (platform consolidation, trend continuation), multi-window scoring |
| **Event Guard** | Symbols | `TradeConstraints` | Block trading during earnings/splits/dividends windows (Polygon + YFinance dual-source) |
| **Strategy** | Candidates + Constraints | `OrderIntentSet` | Convert candidates to bracket intents (entry + SL + TP), position sizing |
| **Risk Gate** | Intents + Portfolio | `RiskDecisionSet` | Approve/block/downgrade intents; Safe Mode circuit breaker |
| **Execution** | Approved intents | `ExecutionReport` | Submit orders to IBKR, manage OCA groups, track partial fills |
| **Order State Machine** | Intents + Broker state | `IntentOrderMappingSet` | Track intent -> order -> fill lifecycle, broker reconciliation |
| **Journal** | All snapshots | Persisted artifacts | Immutable run lineage with schema versioning for deterministic replay |
| **Reporting** | Run data | JSON + Markdown | Per-run structured reports |
| **Notifier** | Run summary | Telegram message | P0 (crash/fail) and P1 (run summary) alerts |

### Cross-Cutting Concerns

| Component | Location | Purpose |
|-----------|----------|---------|
| `Result[T]` | `common/interface.py` | All module boundaries return SUCCESS/DEGRADED/FAILED -- no exceptions for business logic |
| `SnapshotBase` | `journal/interface.py` | Versioned immutable data (schema_version + system_version + asof_timestamp) |
| `PluginBase` | `plugins/interface.py` | Lifecycle contract: init -> validate_config -> execute -> cleanup |
| `EventBus` | `common/events.py` | In-memory pub/sub domain events with wildcard topic matching |
| `Config` | `common/config.py` | Layered YAML: base -> env -> local -> environment variables |

## Daily Schedule (Daemon Mode)

```
09:00  PRE_MARKET_FULL_SCAN    — full pipeline scan + order generation
10:30  INTRADAY_CHECK_1030     — execution-only reconciliation
12:30  INTRADAY_CHECK_1230     — execution-only reconciliation
14:30  INTRADAY_CHECK_1430     — execution-only + safe reducing
15:55  PRE_CLOSE_CLEANUP       — cancel entries + verify stops
16:40  AFTER_MARKET_RECON      — end-of-day reconciliation + report
```

## Quick Start

### Prerequisites

- Python 3.11+
- IBKR TWS or Gateway (paper trading account)
- Polygon.io API key (Basic plan sufficient)

### Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
cp config/secrets.env.example config/secrets.env
# Edit config/secrets.env with your POLYGON_API_KEY
```

### Run

```bash
# Dry run (no broker, reports only)
python scripts/run_paper_eod_scan.py --dry-run

# Paper trading (requires IBKR on port 7497)
python scripts/run_paper_eod_scan.py --paper

# Backtest
python scripts/run_backtest.py --start 2023-01-01 --end 2023-12-31
```

### Daemon Mode

```bash
# Long-running process with 6-segment daily schedule
nohup python scripts/ast_daemon.py --paper >> logs/cron.log 2>&1 &
```

## Key Design Patterns

### Result[T] -- No Exceptions for Business Logic

```python
# Every module boundary returns Result[T] instead of raising exceptions
result = scanner.detect(bars, symbol="AAPL", config=config)
if result.status is ResultStatus.SUCCESS:
    candidates = result.data
elif result.status is ResultStatus.DEGRADED:
    # Partial data available, proceed with caution
    candidates = result.data  # may be incomplete
    log.warning(result.reason_code)
elif result.status is ResultStatus.FAILED:
    # No data, handle gracefully
    log.error(result.error)
```

### SnapshotBase -- Versioned Immutable Data

```python
class CandidateSet(SnapshotBase, frozen=True, kw_only=True):
    # Every persisted payload inherits these fields:
    #   schema_version: str    "3.2.0"
    #   system_version: str    git commit hash
    #   asof_timestamp: int    nanoseconds
    candidates: list[PlatformCandidate]
    total_scanned: int
    total_detected: int
```

### Plugin Lifecycle

```python
# All modules follow the same lifecycle contract
plugin.init(context)           # Allocate resources
plugin.validate_config(config) # Fail fast on bad config
plugin.execute(payload)        # Do the work, return Result[T]
plugin.cleanup()               # Release resources
```

### Bracket Order Pattern

```python
# Strategy generates atomic bracket groups (entry + protective stops)
IntentGroup:
  entry:       OPEN_LONG  $50.00 x 100 shares  (contingency: OTO)
  stop_loss:   STOP_LOSS  $47.50              (contingency: OUO, reduce_only)
  take_profit: TAKE_PROFIT $55.00              (contingency: OUO, reduce_only)
# OTO = One-Triggers-Other, OUO = One-Updates-Other
```

### Safe Mode Circuit Breaker

```
ACTIVE          -- Normal operation, all order types allowed
  | portfolio violation (drawdown, concentration, daily loss)
  v
SAFE_REDUCING   -- Reduce-only mode, no new entries allowed
  | critical failure (broker disconnect, data corruption)
  v
HALTED          -- No orders of any kind
```

## Backtest Engine

The built-in backtest engine (`backtest/`) supports date-driven simulation with realistic execution modeling:

- **Daily loop**: check exits -> execute entries -> run EOD scan -> generate next-day intents
- **Exit strategies**: Stop Loss, Take Profit, Time Stop, Trailing Stop (R-based activation + ATR-adaptive trail), Staged Take Profit (partial exits at configurable thresholds), Weak Exit (position health-based), Time Stop V2 (hold/extend decision point)
- **Position sizing**: Fixed percent, fixed risk, quality-scaled, volatility-scaled, adaptive (with capital allocation context)
- **Market regime integration**: Bull/Bear/Choppy detection with regime-specific config overlays (scanner thresholds, strategy parameters, risk limits)
- **Capital allocation**: Dynamic tier sizing, max exposure caps, cash reserves, commission optimization
- **Position rotation**: Replace underperforming positions with higher-scoring candidates
- **Reporting**: Per-trade CSV export, summary statistics (Sharpe, max drawdown, win rate, profit factor)

```bash
python scripts/run_backtest.py --start 2023-01-01 --end 2023-12-31 --capital 10000
```

## Extending the Framework

### Add Your Own Pattern Detector

```python
# scanner/patterns/your_pattern.py
from scanner.patterns.interface import TrendPatternConfig, TrendPatternResult

class YourPatternConfig(TrendPatternConfig, frozen=True, kw_only=True):
    your_param: float = 0.5

class YourPatternDetector:
    def __init__(self, config=None):
        self.config = config or YourPatternConfig()

    @property
    def pattern_name(self) -> str:
        return "your_pattern"

    def detect(self, symbol, bars, current_date, **kwargs):
        # Your detection logic here
        ...
```

See `scanner/patterns/ma_crossover.py` for a complete working example.

### Configuration

Layered config: `config.yaml` → `config.{env}.yaml` → `config.local.yaml` → environment variables.

Secrets in `config/secrets.env` (git-ignored). See `config/secrets.env.example`.

## Operating Modes

- **DRY_RUN** — Full pipeline, no broker connection, reports only
- **PAPER** — Real orders on IBKR paper account
- **LIVE** — Real trading (use at your own risk)

## Testing

```bash
pytest --tb=short -q
```

## Project Structure

```
common/              # Result[T], Config, EventBus, logging
universe/            # Equity universe builder (Polygon filters)
data/                # Price data (Polygon + S3 hybrid, SQLite cache)
scanner/             # Pattern detection framework + filters + regime
  patterns/          # Pluggable pattern detectors (demo: MA Crossover)
  filters/           # Filter chain (ATR, liquidity, market cap, etc.)
  market_regime/     # Bull/Bear/Choppy regime detection
  gates/             # Sector regime + relative strength gates
event_guard/         # Event-driven trading constraints
strategy/            # Intent generation (sizing + pricing)
risk_gate/           # Portfolio/symbol/operational risk checks
execution/           # IBKR adapter + order management
order_state_machine/ # Intent → order → fill lifecycle
journal/             # Run lineage + deterministic replay
portfolio/           # Capital allocation + position management
indicators/          # ATR, RSI, MACD, Bollinger, etc.
backtest/            # Date-driven trade simulator
reporting/           # JSON + Markdown run reports
notifier/            # Telegram alerts
orchestrator/        # Pipeline coordination + plugin lifecycle
plugins/             # Plugin registry + lifecycle management
scripts/             # Entry points (daemon, paper trading, backtest)
config/              # YAML configuration + regime overlays
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Disclaimer

For educational and research purposes only. Trading involves risk of loss. Use at your own risk. The authors are not responsible for any financial losses incurred through use of this software.
