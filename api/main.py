"""
Kairos Engine — API with WebSocket streaming
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import numpy as np
import json
import asyncio

from engine.pipeline.market_pipeline import MarketPipeline
from engine.options.oi_gravity import OIGravityTracker
from engine.options.gamma_map import GammaGravityMap
from data.models.market_state import MarketState

app = FastAPI(title="Kairos Engine", version="0.2.0")
pipeline = MarketPipeline()
oi_tracker = OIGravityTracker()
gamma_map = GammaGravityMap()


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
    return pipeline.process(c, h, lo, o, v, iv, symbol=data.symbol)


@app.post("/oi-gravity")
def oi_gravity(spot: float = 25000.0):
    result = oi_tracker.analyze(spot)
    return {
        "call_gravity": result.call_gravity,
        "put_gravity": result.put_gravity,
        "combined_gravity": result.combined_gravity,
        "max_pain": result.max_pain,
        "pcr": result.pcr_overall,
        "velocity": result.gravity_velocity,
    }


@app.post("/gamma-map")
def gamma_exposure(spot: float = 25000.0, iv: float = 0.15):
    result = gamma_map.estimate(spot, iv=iv)
    return {
        "total_gex": result.total_gamma_exposure,
        "flip_strike": result.gamma_flip_strike,
        "max_gamma_strike": result.max_gamma_strike,
        "regime": result.regime_note,
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0", "min_candles": pipeline.min_candles}


@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    """
    WebSocket streaming: client sends candle arrays as JSON,
    server responds with full MarketState after each update.
    """
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            c = np.array(data["closes"])
            h = np.array(data["highs"])
            lo = np.array(data["lows"])
            o = np.array(data["opens"])
            v = np.array(data["volumes"])
            iv = np.array(data["iv_series"]) if "iv_series" in data else None

            state = pipeline.process(c, h, lo, o, v, iv, symbol=data.get("symbol", "NIFTY"))
            await ws.send_text(state.model_dump_json())
    except WebSocketDisconnect:
        pass