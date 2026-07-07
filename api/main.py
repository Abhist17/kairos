"""
Kairos Engine — API

FastAPI endpoint that accepts candle data and returns
the full MarketState analysis.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from engine.pipeline.market_pipeline import MarketPipeline
from data.models.market_state import MarketState

app = FastAPI(title="Kairos Engine", version="0.1.0")
pipeline = MarketPipeline()


class CandleInput(BaseModel):
    closes: list[float]
    highs: list[float]
    lows: list[float]
    opens: list[float]
    volumes: list[float]
    iv_series: list[float] | None = None
    symbol: str = "NIFTY"


@app.post("/analyze", response_model=MarketState)
def analyze(data: CandleInput):
    c = np.array(data.closes)
    h = np.array(data.highs)
    lo = np.array(data.lows)
    o = np.array(data.opens)
    v = np.array(data.volumes)
    iv = np.array(data.iv_series) if data.iv_series else None

    state = pipeline.process(c, h, lo, o, v, iv, symbol=data.symbol)
    return state


@app.get("/health")
def health():
    return {"status": "ok", "min_candles": pipeline.min_candles}
