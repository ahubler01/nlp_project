# FinLens — Financial News Intelligence Dashboard

Local dashboard over a closed 139k-article corpus that surfaces the lifecycle, seasonality,
and ticker impact of financial news topics. The corpus ends at **2023-12-16**; everything is
served from precomputed parquet caches except user-defined topics, which are embedded live
with MiniLM and scored against the cached article embeddings.

## Start in 5 commands

```bash
make install   # backend + frontend deps (conda env ie_env is recommended)
make models    # verify every model artefact exists under models/
make warmup    # precompute cache/{fixed_timeline_grid, topic_ticker_matrix, fixed_seasonality}.parquet
make dev       # FastAPI on :8000, Vite on :3000
# open http://localhost:3000
```

- `make warmup` is **only** required once, or whenever a predictions parquet is regenerated.
- User-defined topics persist to `cache/topics.db` (SQLite). Only the 384-d MiniLM embedding
  is cached — the per-article cosine scores and timeline are recomputed in-process, as
  specified.

## Architecture

```
backend/
  main.py                 FastAPI app + CORS + lifespan hook
  config.py               paths, topic labels, 29-ticker pool derived at load
  data_store.py           read-only corpus (polars) + memory-mapped article embeddings
  loaders.py              lazy getters for MiniLM / BERTopic / NER / FinBERT / XGB
  services/
    phase_detector.py     Gaussian-fit cycle + momentum detector
    timeline.py           weekly intensity (fixed from cache, user via MiniLM cosine)
    topics.py             SQLite-backed user-topic CRUD + MiniLM embed
    stocks.py             per-ticker aggregates + topic × ticker matrix + price series
    articles.py           drill-down ranking for a given (topic, iso-week)
  scripts/
    check_models.py       make models
    warmup.py             make warmup
frontend/                 Vite + React 18 + Recharts + TanStack Query
```

All inference is cached: no model needs to load at startup unless a user topic is created
or deleted. `/timeline`, `/phase`, `/stocks`, `/matrix`, `/seasonality`, `/articles` read
straight from parquet.

## API (port 8000)

- `GET  /health`
- `GET  /topics` · `POST /topics {label, description?}` · `DELETE /topics/{id}`
- `GET  /timeline?topic_id=...&weeks=24`
- `GET  /phase?topic_id=...`
- `GET  /stocks?topic_id=...&top_n=20`
- `GET  /matrix?tickers=AAPL,JPM,NVDA`
- `GET  /articles?topic_id=...&iso_week=2023-W47&top_n=5`
- `GET  /seasonality?topic_id=...`
- `GET  /price?ticker=MSFT&weeks=24`
- `GET  /universe` (list of tickers present in subset_news)

The dev frontend proxies `/api/*` → `:8000`.

## Corpus & caches

| Path | Source |
|---|---|
| `data/preprocessed/subset_news.parquet` | 139,522 articles, 29 tickers, 2009-10 → 2023-12 |
| `data/preprocessed/article_embeddings.npy` | 384-d MiniLM (memory-mapped at startup) |
| `data/predictions/full_df_topic_probabilities.parquet` | 13 zero-shot topic probabilities |
| `data/predictions/xgb_tb_predictions.parquet` | Triple-barrier `proba_up`/`pred` |
| `data/predictions/finbert_embeddings.parquet` | Sentiment + 768-d CLS embeddings |
| `data/Stock_price/stock_price.parquet` | Daily OHLCV for overlays |
| `cache/fixed_timeline_grid.parquet` | (topic_id, iso_year, iso_week) → (intensity, n) |
| `cache/topic_ticker_matrix.parquet` | (topic_id, ticker) → mean relevance over 24w |
| `cache/fixed_seasonality.parquet` | (topic_id, week_of_year) → mean intensity |

## Notes / sharp edges

- **Ticker universe**: the subset has 29 tickers (not 63). `/stocks` restricts to this set.
- **No live XGBoost for new articles**: `models/xgb_tb/emb_pipeline.pkl` is not persisted.
  XGBoost results are served from the parquet cache only. To enable live scoring, re-run
  notebook 09's "Train final XGBoost" cell with the `joblib.dump(...)` snippet from
  `docs/INFERENCE.md §5`.
- **Seasonality**: corpus span is ~14 years, well above the 104-week minimum — enabled.
- **User topic policy**: the embedding is cached in SQLite; per-article cosine similarities
  and the weekly intensity curve are **not** persisted (recomputed per request).
