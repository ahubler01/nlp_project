"""Per-ticker aggregates and topic x ticker matrix."""
from __future__ import annotations

import numpy as np
import polars as pl

from backend import config
from backend.data_store import Corpus, load_corpus, topic_ticker_matrix
from backend.services import topics as topics_service


def _window_ids(corpus: Corpus, weeks: int) -> pl.DataFrame:
    keys = corpus.window_week_keys(weeks)
    win = pl.DataFrame(
        {"iso_year": [k[0] for k in keys], "iso_week": [k[1] for k in keys]}
    )
    return corpus.news.select(["id", "Stock_symbol", "iso_year", "iso_week"]).join(
        win, on=["iso_year", "iso_week"], how="inner"
    )


def per_ticker(topic_id: str, top_n: int = 20, weeks: int = config.DEFAULT_WEEKS) -> list[dict]:
    corpus = load_corpus()
    win = _window_ids(corpus, weeks)

    if topic_id in config.FIXED_TOPIC_BY_ID:
        prob_col = config.FIXED_TOPIC_BY_ID[topic_id]["prob_col"]
        scored = win.join(
            corpus.topic_probs.select(["id", prob_col]),
            on="id", how="inner",
        ).rename({prob_col: "relevance"})
    else:
        topic = topics_service.get_user_topic(topic_id)
        if topic is None:
            return []
        emb = topic["embedding"].astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        sims = (corpus.embeddings @ emb).astype(np.float32)
        sims = (sims + 1.0) / 2.0
        sim_df = pl.DataFrame({
            "id": corpus.embedding_ids.astype(np.uint32),
            "relevance": sims,
        })
        scored = win.join(sim_df, on="id", how="inner")

    # merge xgb predictions (best-effort; left join)
    scored = scored.join(
        corpus.xgb.select(["id", "proba_up"]),
        on="id", how="left",
    )

    agg = (
        scored.group_by("Stock_symbol")
        .agg([
            pl.col("relevance").mean().alias("mean_rel"),
            pl.len().alias("article_count"),
            pl.col("proba_up").mean().alias("proba_up_mean"),
        ])
        .with_columns(
            (pl.col("mean_rel") *
             (1.0 + (pl.col("proba_up_mean").fill_null(0.5) - 0.5).abs())
             ).alias("score")
        )
        .sort("score", descending=True)
        .head(top_n)
    )

    # fetch top 3 headlines per ticker for the returned rows
    top_tickers = agg["Stock_symbol"].to_list()
    titles_df = (
        scored.filter(pl.col("Stock_symbol").is_in(top_tickers))
        .sort("relevance", descending=True)
        .group_by("Stock_symbol", maintain_order=True)
        .head(3)
        .join(
            corpus.news.select(["id", "Article_title", "Url", "date_parsed"]),
            on="id", how="inner",
        )
    )

    headlines_by_ticker: dict[str, list[dict]] = {t: [] for t in top_tickers}
    for r in titles_df.iter_rows(named=True):
        headlines_by_ticker.setdefault(r["Stock_symbol"], []).append({
            "id": int(r["id"]),
            "title": r["Article_title"],
            "date": r["date_parsed"].isoformat() if r["date_parsed"] else None,
            "url": r["Url"],
        })

    out = []
    for r in agg.iter_rows(named=True):
        out.append({
            "ticker": r["Stock_symbol"],
            "score": float(r["score"] or 0.0),
            "article_count": int(r["article_count"]),
            "proba_up_mean": (float(r["proba_up_mean"])
                              if r["proba_up_mean"] is not None else None),
            "top_headlines": headlines_by_ticker.get(r["Stock_symbol"], []),
        })
    return out


def topic_matrix(tickers: list[str]) -> dict:
    corpus = load_corpus()
    tickers = [t for t in tickers if t in set(corpus.pool)]
    if not tickers:
        return {"rows": [], "cols": [t["id"] for t in config.FIXED_TOPICS], "values": []}

    grid = topic_ticker_matrix()  # topic_id, ticker, mean_rel
    pivot = (
        grid.filter(pl.col("ticker").is_in(tickers))
        .pivot(index="ticker", on="topic_id", values="mean_rel")
        .sort("ticker")
    )

    cols = [t["id"] for t in config.FIXED_TOPICS]
    rows_out: list[str] = []
    values_out: list[list[float]] = []
    # keep the user-requested row order
    ticker_rows = {r["ticker"]: r for r in pivot.iter_rows(named=True)}
    for t in tickers:
        if t not in ticker_rows:
            rows_out.append(t)
            values_out.append([0.0] * len(cols))
            continue
        row = ticker_rows[t]
        rows_out.append(t)
        values_out.append([float(row.get(c) or 0.0) for c in cols])

    return {"rows": rows_out, "cols": cols, "values": values_out}


def price_series(ticker: str, weeks: int = config.DEFAULT_WEEKS) -> list[dict]:
    corpus = load_corpus()
    cutoff = corpus.max_date.toordinal() - weeks * 7
    from datetime import date as _date
    cutoff_date = _date.fromordinal(cutoff)
    sel = (
        corpus.stock_price.filter(
            (pl.col("ticker") == ticker) & (pl.col("date") >= cutoff_date)
        )
        .sort("date")
    )
    return [
        {"date": r["date"].isoformat(),
         "close": float(r["close"]) if r["close"] is not None else None,
         "volume": int(r["volume"]) if r["volume"] is not None else 0}
        for r in sel.iter_rows(named=True)
    ]
