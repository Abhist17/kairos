# Kairos Engine

**Real-Time Options Trade Timing Intelligence Engine**

> *Kairos* (Καιρός) — the ancient Greek word for "the supreme moment"

## What This Is

Kairos doesn't predict market direction. It answers a harder question:

**"Even if my direction is correct, is THIS exact moment mathematically efficient to enter THIS specific option contract?"**

## Architecture
Market Data → Regime → Structure → Compression → IV State
→ Options Efficiency → Flow/Pressure → Thesis → State Machine → Entry Window

### 8 Engines

| Engine | Purpose |
|--------|---------|
| **Regime** | Classify market state (trend/compression/chaotic/exhaustion/mean-reversion) |
| **Compression** | Detect volatility contraction, velocity, and half-life |
| **Structure** | Find magnetic price zones from PDH/PDL, VWAP, volume nodes, swings |
| **IV State** | Classify implied volatility (compressed/normal/expanding/overexpanded/collapsing) |
| **Options Efficiency** | Delta acceleration, gamma/theta ratio, theta survival, move feasibility |
| **Flow/Pressure** | Level tests, pullback shrinkage, aggression, liquidity depletion |
| **Thesis** | Score primary vs counter thesis, measure separation |
| **State Machine** | 6-gate sequential validation → ENTRY_WINDOW_OPEN |

### State Machine Gates
STRUCTURE → COMPRESSION → PRESSURE → OPTION_EFF → FLOW_CONF → ENTRY
●            ●            ●           ○            ○          ○

ALL gates must be READY. This is NOT `score > 80 = BUY`.

### Additional Analysis

- **OI Center of Gravity** — tracks OI concentration movement (ESTIMATED)
- **Gamma Gravity Map** — estimates gamma exposure across strikes (ESTIMATED)
- **Setup Memory** — stores every setup with forward outcomes for future ML

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy pydantic ruff pytest fastapi uvicorn websockets

# Test
pytest tests/ -v

# Run demo
python main.py

# Dashboard
python dashboard.py

# Dashboard with real data
python dashboard.py --csv path/to/nifty_1min.csv

# Backtest with analytics
python -m backtest.run_backtest

# API
uvicorn api.main:app --reload --port 8000
```

## API Endpoints

- `POST /analyze` — Full pipeline analysis from OHLCV arrays
- `POST /oi-gravity` — OI Center of Gravity
- `POST /gamma-map` — Gamma exposure estimation
- `GET /health` — Status check
- `WS /ws/stream` — WebSocket streaming

## What This Is NOT

- Not a trading journal or P&L tracker
- Not a TradingView clone
- Not "AI predicts stocks"
- Not a broker execution system
- Does NOT claim profitability

## Philosophy

Every threshold is documented with WHY, not just WHAT. Dealer gamma is always labelled ESTIMATED because exact positioning is not publicly observable. ML comes AFTER collecting labelled data, not before.

## Project Status

- [x] Regime Engine V1
- [x] Compression Engine V1
- [x] Structure Engine V1
- [x] IV State Engine V1
- [x] Options Efficiency Engine V1
- [x] Flow/Pressure Engine V1
- [x] Thesis Engine V1
- [x] Entry State Machine V1
- [x] Pipeline Orchestrator
- [x] Setup Memory (SQLite)
- [x] Backtest Engine + Analytics
- [x] OI Center of Gravity
- [x] Gamma Gravity Map
- [x] CSV Data Loader
- [x] REST API + WebSocket
- [x] Terminal Dashboard
- [ ] Real broker data connector
- [ ] Vanna / Charm flow
- [ ] ML on labelled setup data