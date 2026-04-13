"""Read-only access to parquet caches + memory-mapped article embeddings."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import polars as pl

from backend import config

log = logging.getLogger(__name__)


@dataclass
class Corpus:
    news: pl.DataFrame            # subset_news with iso_year / iso_week columns added
    topic_probs: pl.DataFrame     # id + 13 prob columns
    xgb: pl.DataFrame             # id, ticker, date, proba_up, pred
    stock_price: pl.DataFrame     # date, ticker, close, volume
    embeddings: np.ndarray        # (N, 384) float32 memmap
    embedding_ids: np.ndarray     # (N,) uint32 — aligns rows in `embeddings` to `id`
    id_to_row: dict[int, int]     # id -> row index in embeddings
    pool: list[str]               # sorted list of tickers present in subset_news
    date_min: str
    date_max: str
    max_date: "pl.Date"           # as date
    current_week_key: tuple[int, int]  # (iso_year, iso_week) of max_date

    @property
    def corpus_span_weeks(self) -> int:
        return int((self.news["date_parsed"].max() - self.news["date_parsed"].min()).days // 7)

    def window_week_keys(self, weeks: int = config.DEFAULT_WEEKS) -> list[tuple[int, int]]:
        """Return the ordered list of (iso_year, iso_week) keys for the last `weeks` ISO weeks."""
        # enumerate all unique (iso_year, iso_week) that appear, take the trailing `weeks`.
        keys = (
            self.news.select(["iso_year", "iso_week"])
            .unique()
            .sort(["iso_year", "iso_week"])
        )
        keys_list = list(zip(keys["iso_year"].to_list(), keys["iso_week"].to_list()))
        return keys_list[-weeks:]


_corpus: Optional[Corpus] = None


def _add_iso_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("date_parsed").dt.iso_year().alias("iso_year"),
        pl.col("date_parsed").dt.week().alias("iso_week"),
    )


def load_corpus() -> Corpus:
    """Load (once) every precomputed artefact the backend needs."""
    global _corpus
    if _corpus is not None:
        return _corpus

    log.info("Loading subset_news …")
    news = pl.read_parquet(config.SUBSET_NEWS)
    news = _add_iso_columns(news)

    log.info("Loading topic probabilities …")
    topic_probs = pl.read_parquet(config.TOPIC_PROBS)

    log.info("Loading xgb predictions …")
    xgb = pl.read_parquet(config.XGB_PREDICTIONS)
    # normalise date column to a pl.Date for clean joins
    if xgb["date"].dtype != pl.Date:
        xgb = xgb.with_columns(pl.col("date").cast(pl.Date))

    log.info("Loading stock prices …")
    stock_price = pl.read_parquet(config.STOCK_PRICE).select(
        ["date", "ticker", "close", "volume"]
    )
    if stock_price["date"].dtype != pl.Date:
        stock_price = stock_price.with_columns(pl.col("date").cast(pl.Date))

    log.info("Memory-mapping article embeddings …")
    embeddings = np.load(config.ARTICLE_EMBEDDINGS_NPY, mmap_mode="r")
    embedding_ids = pl.read_parquet(config.ARTICLE_EMBEDDING_IDS)["id"].to_numpy()
    id_to_row = {int(i): row for row, i in enumerate(embedding_ids.tolist())}

    pool = sorted(news["Stock_symbol"].unique().to_list())
    date_min = str(news["date_parsed"].min())
    date_max = str(news["date_parsed"].max())
    max_date = news["date_parsed"].max()

    max_row = news.filter(pl.col("date_parsed") == max_date).head(1)
    current_week_key = (
        int(max_row["iso_year"][0]),
        int(max_row["iso_week"][0]),
    )

    _corpus = Corpus(
        news=news,
        topic_probs=topic_probs,
        xgb=xgb,
        stock_price=stock_price,
        embeddings=embeddings,
        embedding_ids=embedding_ids,
        id_to_row=id_to_row,
        pool=pool,
        date_min=date_min,
        date_max=date_max,
        max_date=max_date,
        current_week_key=current_week_key,
    )
    log.info(
        "Corpus ready: %d rows, %d tickers, %s → %s",
        len(news), len(pool), date_min, date_max,
    )
    return _corpus


@lru_cache(maxsize=1)
def fixed_timeline_grid() -> pl.DataFrame:
    """Load the precomputed (topic, iso_year, iso_week) -> intensity/count grid.
    Regenerate via scripts/warmup.py."""
    if not config.CACHE_TIMELINE_GRID.exists():
        from backend.scripts.warmup import build_fixed_timeline_grid
        build_fixed_timeline_grid()
    return pl.read_parquet(config.CACHE_TIMELINE_GRID)


@lru_cache(maxsize=1)
def topic_ticker_matrix() -> pl.DataFrame:
    """Precomputed (topic_id, ticker) -> mean weighted score, over the 24w window."""
    if not config.CACHE_TOPIC_TICKER.exists():
        from backend.scripts.warmup import build_topic_ticker_matrix
        build_topic_ticker_matrix()
    return pl.read_parquet(config.CACHE_TOPIC_TICKER)


@lru_cache(maxsize=1)
def fixed_seasonality() -> pl.DataFrame:
    if not config.CACHE_SEASONALITY.exists():
        from backend.scripts.warmup import build_fixed_seasonality
        build_fixed_seasonality()
    return pl.read_parquet(config.CACHE_SEASONALITY)
