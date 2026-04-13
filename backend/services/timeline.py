"""Weekly intensity curves for fixed and user-defined topics."""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import polars as pl

from backend import config
from backend.data_store import Corpus, load_corpus, fixed_timeline_grid
from backend.services import topics as topics_service

log = logging.getLogger(__name__)


def _iso_week_start(iso_year: int, iso_week: int) -> date:
    """Monday of the given ISO week."""
    return date.fromisocalendar(iso_year, iso_week, 1)


def _timeline_from_grid(topic_id: str, weeks: int) -> list[dict]:
    corpus = load_corpus()
    grid = fixed_timeline_grid()  # columns: topic_id, iso_year, iso_week, intensity, article_count
    sel = grid.filter(pl.col("topic_id") == topic_id)
    if sel.height == 0:
        return []
    window_keys = corpus.window_week_keys(weeks)
    win = pl.DataFrame(
        {"iso_year": [k[0] for k in window_keys],
         "iso_week": [k[1] for k in window_keys]}
    )
    sel = win.join(sel.drop("topic_id"), on=["iso_year", "iso_week"], how="left")
    sel = sel.with_columns([
        pl.col("intensity").fill_null(0.0),
        pl.col("article_count").fill_null(0),
    ])
    out = []
    for row in sel.iter_rows(named=True):
        ws = _iso_week_start(row["iso_year"], row["iso_week"])
        out.append({
            "iso_week": f"{row['iso_year']}-W{row['iso_week']:02d}",
            "week_start": ws.isoformat(),
            "intensity": float(row["intensity"]),
            "article_count": int(row["article_count"]),
        })
    return out


def _timeline_user(topic: dict, weeks: int) -> list[dict]:
    """Compute weekly intensity for a user topic via MiniLM cosine similarity.
    Does NOT persist the per-article distances — only the topic embedding is cached."""
    corpus = load_corpus()
    emb = topic["embedding"].astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-12)

    # article embeddings are already L2-normalised at training time (sentence-transformers default).
    # but keep a guard: just dot product.
    sims = (corpus.embeddings @ emb).astype(np.float32)  # (N,)
    sims = (sims + 1.0) / 2.0  # -> [0, 1] per brief instructions

    # align to ids via embedding_ids
    df = pl.DataFrame({
        "id": corpus.embedding_ids.astype(np.uint32),
        "sim": sims,
    })
    joined = df.join(
        corpus.news.select(["id", "iso_year", "iso_week"]),
        on="id", how="inner",
    )
    weekly = (
        joined.group_by(["iso_year", "iso_week"])
        .agg([
            pl.col("sim").mean().alias("intensity"),
            pl.len().alias("article_count"),
        ])
    )

    window_keys = corpus.window_week_keys(weeks)
    win = pl.DataFrame(
        {"iso_year": [k[0] for k in window_keys],
         "iso_week": [k[1] for k in window_keys]}
    )
    weekly = win.join(weekly, on=["iso_year", "iso_week"], how="left").with_columns([
        pl.col("intensity").fill_null(0.0),
        pl.col("article_count").fill_null(0),
    ])

    out = []
    for row in weekly.iter_rows(named=True):
        ws = _iso_week_start(row["iso_year"], row["iso_week"])
        out.append({
            "iso_week": f"{row['iso_year']}-W{row['iso_week']:02d}",
            "week_start": ws.isoformat(),
            "intensity": float(row["intensity"]),
            "article_count": int(row["article_count"]),
        })
    return out


def timeline(topic_id: str, weeks: int = config.DEFAULT_WEEKS) -> list[dict]:
    if topic_id in config.FIXED_TOPIC_BY_ID:
        return _timeline_from_grid(topic_id, weeks)
    topic = topics_service.get_user_topic(topic_id)
    if topic is None:
        return []
    return _timeline_user(topic, weeks)


def seasonality(topic_id: str) -> list[dict]:
    """Aggregate intensity by ISO week-of-year across the whole corpus.
    Return [] when the corpus span is under the configured minimum."""
    corpus = load_corpus()
    if corpus.corpus_span_weeks < config.SEASONALITY_MIN_SPAN_WEEKS:
        return []

    if topic_id in config.FIXED_TOPIC_BY_ID:
        from backend.data_store import fixed_seasonality
        grid = fixed_seasonality()
        sel = grid.filter(pl.col("topic_id") == topic_id).sort("week_of_year")
        return [
            {"week_of_year": int(r["week_of_year"]),
             "intensity": float(r["intensity"]),
             "n_years": int(r["n_years"])}
            for r in sel.iter_rows(named=True)
        ]

    # user topic — compute on the fly (still acceptable for a one-off view)
    topic = topics_service.get_user_topic(topic_id)
    if topic is None:
        return []
    emb = topic["embedding"].astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    sims = (corpus.embeddings @ emb).astype(np.float32)
    sims = (sims + 1.0) / 2.0
    df = pl.DataFrame({"id": corpus.embedding_ids.astype(np.uint32), "sim": sims})
    joined = df.join(
        corpus.news.select(["id", "iso_year", "iso_week"]),
        on="id", how="inner",
    )
    agg = (
        joined.group_by("iso_week")
        .agg([
            pl.col("sim").mean().alias("intensity"),
            pl.col("iso_year").n_unique().alias("n_years"),
        ])
        .sort("iso_week")
    )
    return [
        {"week_of_year": int(r["iso_week"]),
         "intensity": float(r["intensity"]),
         "n_years": int(r["n_years"])}
        for r in agg.iter_rows(named=True)
    ]
