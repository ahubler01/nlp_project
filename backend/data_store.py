"""Load all parquets at import time. Heavy models are lazy-loaded."""
import numpy as np
import polars as pl
from config import (
    SUBSET_NEWS_PATH, TOPIC_PROBS_PATH, FINBERT_PATH,
    XGB_PREDS_PATH, STOCK_PRICE_PATH, ARTICLE_EMB_PATH,
    ARTICLE_EMB_IDS_PATH, ARTICLE_SENTIMENTS_PATH, POOL,
)

print("[data_store] loading parquets …")

news = pl.read_parquet(SUBSET_NEWS_PATH)
topic_probs = pl.read_parquet(TOPIC_PROBS_PATH)
xgb_preds = pl.read_parquet(XGB_PREDS_PATH)
stock_price = pl.read_parquet(STOCK_PRICE_PATH)

# real sentiment from FinBERT pipeline (pos/neg/neu probabilities)
sentiments = pl.read_parquet(
    ARTICLE_SENTIMENTS_PATH,
    columns=["article_id", "pos_prob", "neg_prob", "neu_prob"],
).rename({"article_id": "id"})

# memory-map article embeddings
article_embs = np.load(str(ARTICLE_EMB_PATH), mmap_mode="r")
article_emb_ids = pl.read_parquet(ARTICLE_EMB_IDS_PATH)

# ── join news + topics + sentiment ──────────────────────────────
merged = (
    news
    .join(topic_probs, on="id", how="left")
    .join(sentiments, on="id", how="left")
    .with_columns(
        pl.when(pl.col("pos_prob") > pl.col("neg_prob"))
          .then(pl.when(pl.col("pos_prob") > pl.col("neu_prob"))
                .then(pl.lit("positive"))
                .otherwise(pl.lit("neutral")))
          .otherwise(
            pl.when(pl.col("neg_prob") > pl.col("neu_prob"))
            .then(pl.lit("negative"))
            .otherwise(pl.lit("neutral"))
          )
          .alias("sentiment")
    )
)

# filter stock prices to pool tickers
pool_prices = stock_price.filter(pl.col("ticker").is_in(POOL))

print(f"[data_store] {len(news):,} articles, {len(pool_prices):,} price rows loaded")
