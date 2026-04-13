"""Precompute the weekly-intensity grid, topic×ticker matrix, and seasonality tables.

Run once after cloning (`make warmup`). Re-run whenever the parquet caches change.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from backend import config  # noqa: E402
from backend.data_store import load_corpus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("warmup")


def build_fixed_timeline_grid() -> pl.DataFrame:
    """One row per (topic_id, iso_year, iso_week) with mean topic prob + article count."""
    corpus = load_corpus()
    joined = corpus.news.select(["id", "iso_year", "iso_week"]).join(
        corpus.topic_probs, on="id", how="inner",
    )

    # melt the 13 prob_ columns into long form once, then group
    long = joined.unpivot(
        index=["id", "iso_year", "iso_week"],
        on=config.FIXED_PROB_COLS,
        variable_name="prob_col",
        value_name="prob",
    )
    prob_to_id = {t["prob_col"]: t["id"] for t in config.FIXED_TOPICS}
    long = long.with_columns(
        pl.col("prob_col").replace(prob_to_id).alias("topic_id")
    )
    grid = (
        long.group_by(["topic_id", "iso_year", "iso_week"])
        .agg([
            pl.col("prob").mean().alias("intensity"),
            pl.col("id").n_unique().alias("article_count"),
        ])
        .sort(["topic_id", "iso_year", "iso_week"])
    )
    grid.write_parquet(config.CACHE_TIMELINE_GRID)
    log.info("Wrote %s (%d rows)", config.CACHE_TIMELINE_GRID, grid.height)
    return grid


def build_topic_ticker_matrix() -> pl.DataFrame:
    """(topic_id, ticker) -> mean topic relevance over the most recent `DEFAULT_WEEKS`."""
    corpus = load_corpus()
    window_keys = corpus.window_week_keys(config.DEFAULT_WEEKS)
    win = pl.DataFrame(
        {"iso_year": [k[0] for k in window_keys],
         "iso_week": [k[1] for k in window_keys]}
    )
    base = (
        corpus.news.select(["id", "Stock_symbol", "iso_year", "iso_week"])
        .join(win, on=["iso_year", "iso_week"], how="inner")
        .join(corpus.topic_probs, on="id", how="inner")
    )
    rows = []
    for topic in config.FIXED_TOPICS:
        col = topic["prob_col"]
        agg = (
            base.group_by("Stock_symbol")
            .agg(pl.col(col).mean().alias("mean_rel"))
            .with_columns(pl.lit(topic["id"]).alias("topic_id"))
            .rename({"Stock_symbol": "ticker"})
            .select(["topic_id", "ticker", "mean_rel"])
        )
        rows.append(agg)
    out = pl.concat(rows, how="vertical")
    out.write_parquet(config.CACHE_TOPIC_TICKER)
    log.info("Wrote %s (%d rows)", config.CACHE_TOPIC_TICKER, out.height)
    return out


def build_fixed_seasonality() -> pl.DataFrame:
    """(topic_id, week_of_year) -> mean intensity across all years in the corpus."""
    corpus = load_corpus()
    if corpus.corpus_span_weeks < config.SEASONALITY_MIN_SPAN_WEEKS:
        log.warning(
            "Corpus span %d weeks < minimum %d — writing empty seasonality table",
            corpus.corpus_span_weeks, config.SEASONALITY_MIN_SPAN_WEEKS,
        )
        empty = pl.DataFrame(
            schema={"topic_id": pl.Utf8, "week_of_year": pl.Int32,
                    "intensity": pl.Float64, "n_years": pl.Int32}
        )
        empty.write_parquet(config.CACHE_SEASONALITY)
        return empty

    joined = corpus.news.select(["id", "iso_year", "iso_week"]).join(
        corpus.topic_probs, on="id", how="inner",
    )
    frames = []
    for topic in config.FIXED_TOPICS:
        col = topic["prob_col"]
        agg = (
            joined.group_by("iso_week")
            .agg([
                pl.col(col).mean().alias("intensity"),
                pl.col("iso_year").n_unique().alias("n_years"),
            ])
            .with_columns(pl.lit(topic["id"]).alias("topic_id"))
            .rename({"iso_week": "week_of_year"})
            .select(["topic_id", "week_of_year", "intensity", "n_years"])
        )
        frames.append(agg)
    out = pl.concat(frames, how="vertical").sort(["topic_id", "week_of_year"])
    out.write_parquet(config.CACHE_SEASONALITY)
    log.info("Wrote %s (%d rows)", config.CACHE_SEASONALITY, out.height)
    return out


def main() -> None:
    log.info("Starting warmup …")
    build_fixed_timeline_grid()
    build_topic_ticker_matrix()
    build_fixed_seasonality()
    log.info("Warmup complete.")


if __name__ == "__main__":
    main()
