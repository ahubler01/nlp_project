"""FinLens FastAPI backend."""
import sys
from pathlib import Path

# ensure backend/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

import data_store  # noqa: F401 — triggers parquet loading
from services.topics import list_topics
from services.timeline import weekly_intensity, headlines_for_week
from services.graph import get_graph
from services.chat import chat
from services.tickers import get_ticker, list_tickers

app = FastAPI(title="FinLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    # precompute cointegration graph (cached after first run)
    get_graph()


# ── health ──────────────────────────────────────────────────────
@app.get("/health")
def health():
    from config import (
        SUBSET_NEWS_PATH, TOPIC_PROBS_PATH, FINBERT_PATH,
        XGB_PREDS_PATH, STOCK_PRICE_PATH,
    )
    return {
        "status": "ok",
        "artefacts_present": {
            "subset_news": SUBSET_NEWS_PATH.exists(),
            "topic_probs": TOPIC_PROBS_PATH.exists(),
            "finbert": FINBERT_PATH.exists(),
            "xgb_preds": XGB_PREDS_PATH.exists(),
            "stock_price": STOCK_PRICE_PATH.exists(),
        },
    }


# ── topics ──────────────────────────────────────────────────────
@app.get("/topics")
def get_topics():
    return list_topics()


# ── timeline (Topic Explorer) ──────────────────────────────────
@app.get("/timeline")
def get_timeline(topic_id: int = Query(...)):
    return weekly_intensity(topic_id)


@app.get("/timeline/headlines")
def get_headlines(topic_id: int = Query(...), week: str = Query(...)):
    return headlines_for_week(topic_id, week)


# ── graph ───────────────────────────────────────────────────────
@app.get("/graph")
def graph():
    return get_graph()


# ── chat ────────────────────────────────────────────────────────
@app.post("/chat")
async def chat_endpoint(body: dict):
    topic_id = body.get("topic_id", 0)
    query = body.get("query", "")
    return chat(topic_id, query)


# ── tickers ─────────────────────────────────────────────────────
@app.get("/ticker")
def ticker(symbol: str = Query(...)):
    return get_ticker(symbol)


@app.get("/tickers")
def tickers():
    return list_tickers()
