"""Drill-down article ranking by (topic_id, iso_week)."""
from __future__ import annotations

import numpy as np
import polars as pl

from backend import config
from backend.data_store import load_corpus
from backend.services import topics as topics_service


def drill_down(topic_id: str, iso_week_str: str, top_n: int = 5) -> list[dict]:
    """iso_week_str is the '2023-W47' format emitted by /timeline."""
    corpus = load_corpus()
    try:
        iso_year, iso_week = iso_week_str.split("-W")
        iso_year, iso_week = int(iso_year), int(iso_week)
    except Exception:
        return []

    week_articles = corpus.news.filter(
        (pl.col("iso_year") == iso_year) & (pl.col("iso_week") == iso_week)
    )
    if week_articles.height == 0:
        return []

    if topic_id in config.FIXED_TOPIC_BY_ID:
        prob_col = config.FIXED_TOPIC_BY_ID[topic_id]["prob_col"]
        scored = week_articles.join(
            corpus.topic_probs.select(["id", prob_col]),
            on="id", how="inner",
        ).rename({prob_col: "relevance"})
    else:
        topic = topics_service.get_user_topic(topic_id)
        if topic is None:
            return []
        emb = topic["embedding"].astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        ids = week_articles["id"].to_list()
        rows = np.array([corpus.id_to_row[i] for i in ids if i in corpus.id_to_row],
                        dtype=np.int64)
        if len(rows) == 0:
            return []
        sims = (corpus.embeddings[rows] @ emb).astype(np.float32)
        sims = (sims + 1.0) / 2.0
        sim_df = pl.DataFrame({
            "id": pl.Series(ids, dtype=pl.UInt32)[:len(rows)],
            "relevance": sims,
        })
        scored = week_articles.join(sim_df, on="id", how="inner")

    # add finbert sentiment + xgb proba_up (from caches)
    fb = pl.read_parquet(config.FINBERT_EMBEDS, columns=["id", "sentiment", "score"])
    scored = scored.join(fb, on="id", how="left")
    scored = scored.join(
        corpus.xgb.select(["id", "proba_up"]).unique(subset=["id"]),
        on="id", how="left",
    )

    top = scored.sort("relevance", descending=True).head(top_n)
    out = []
    for r in top.iter_rows(named=True):
        snippet = (r.get("Lsa_summary") or r.get("Article") or "")[:320]
        out.append({
            "id": int(r["id"]),
            "title": r["Article_title"],
            "date": r["date_parsed"].isoformat() if r["date_parsed"] else None,
            "ticker": r["Stock_symbol"],
            "url": r["Url"],
            "snippet": snippet,
            "relevance": float(r["relevance"]),
            "sentiment": r.get("sentiment"),
            "sentiment_score": (float(r["score"]) if r.get("score") is not None else None),
            "proba_up": (float(r["proba_up"]) if r.get("proba_up") is not None else None),
        })
    return out
