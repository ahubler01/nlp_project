"""FinLens FastAPI app. Serves a closed corpus of precomputed artefacts;
the only live inference path is MiniLM encoding for user-defined topics."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import config
from backend import loaders
from backend.data_store import (
    fixed_seasonality,
    fixed_timeline_grid,
    load_corpus,
    topic_ticker_matrix,
)
from backend.services import articles as articles_svc
from backend.services import phase_detector
from backend.services import stocks as stocks_svc
from backend.services import timeline as timeline_svc
from backend.services import topics as topics_svc

log = logging.getLogger("finlens")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    load_corpus()
    # warm the three cache tables (build if absent)
    fixed_timeline_grid()
    topic_ticker_matrix()
    fixed_seasonality()
    yield


app = FastAPI(title="FinLens", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- models -------------------------------------------------------------
class TopicCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=120)
    description: str | None = None


# --- endpoints ----------------------------------------------------------
@app.get("/health")
def health() -> dict:
    paths = {k: v.exists() for k, v in config.MODEL_PATHS.items()}
    corpus = load_corpus()
    return {
        "status": "ok",
        "models_loaded": loaders.models_loaded_status(),
        "artefacts_present": paths,
        "corpus": {
            "rows": len(corpus.news),
            "tickers": len(corpus.pool),
            "date_min": corpus.date_min,
            "date_max": corpus.date_max,
            "span_weeks": corpus.corpus_span_weeks,
            "seasonality_enabled": corpus.corpus_span_weeks >= config.SEASONALITY_MIN_SPAN_WEEKS,
        },
    }


@app.get("/topics")
def get_topics() -> dict:
    fixed = [
        {"id": t["id"], "label": t["label"], "description": t["description"],
         "kind": t["kind"], "color": t["color"]}
        for t in config.FIXED_TOPICS
    ]
    return {"topics": fixed + topics_svc.list_user_topics()}


@app.post("/topics", status_code=201)
def post_topic(body: TopicCreate) -> dict:
    try:
        return topics_svc.create_user_topic(body.label, body.description or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/topics/{topic_id}", status_code=204)
def delete_topic(topic_id: str):
    if topic_id in config.FIXED_TOPIC_BY_ID:
        raise HTTPException(status_code=400, detail="Cannot delete a fixed topic")
    if not topics_svc.delete_user_topic(topic_id):
        raise HTTPException(status_code=404, detail="Topic not found")
    return None


@app.get("/timeline")
def get_timeline(
    topic_id: str,
    weeks: int = Query(config.DEFAULT_WEEKS, ge=4, le=104),
) -> list[dict]:
    return timeline_svc.timeline(topic_id, weeks)


@app.get("/phase")
def get_phase(topic_id: str, weeks: int = config.DEFAULT_WEEKS) -> dict:
    tl = timeline_svc.timeline(topic_id, weeks)
    curve = [p["intensity"] for p in tl]
    return phase_detector.detect_phase(curve, current_week=len(curve) - 1)


@app.get("/stocks")
def get_stocks(
    topic_id: str,
    top_n: int = Query(20, ge=1, le=50),
) -> list[dict]:
    return stocks_svc.per_ticker(topic_id, top_n=top_n)


@app.get("/matrix")
def get_matrix(tickers: str = Query(..., description="comma-separated ticker list")) -> dict:
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return stocks_svc.topic_matrix(ticker_list)


@app.get("/articles")
def get_articles(
    topic_id: str,
    iso_week: str = Query(..., description="e.g. 2023-W47"),
    top_n: int = Query(5, ge=1, le=20),
) -> list[dict]:
    return articles_svc.drill_down(topic_id, iso_week, top_n=top_n)


@app.get("/seasonality")
def get_seasonality(topic_id: str) -> list[dict]:
    return timeline_svc.seasonality(topic_id)


@app.get("/price")
def get_price(
    ticker: str,
    weeks: int = Query(config.DEFAULT_WEEKS, ge=1, le=260),
) -> list[dict]:
    return stocks_svc.price_series(ticker.upper(), weeks=weeks)


@app.get("/universe")
def get_universe() -> dict:
    corpus = load_corpus()
    return {"tickers": corpus.pool}
