# AGENT TASK — FinLens: Financial News Intelligence Dashboard

## Objective

Build **FinLens**, a local production dashboard that surfaces the lifecycle, seasonality, and ticker impact of financial news topics. A working React/Recharts UI prototype exists (dark terminal aesthetic — IBM Plex Mono, Syne display, `#020817` background). Your job is to wire every already-trained model in [models/](../models/) into a FastAPI backend, replace all simulated data with live inference over [data/preprocessed/subset_news.parquet](../data/preprocessed/subset_news.parquet), and ship a self-contained local app.

This project does **not** pull from any external dataset (no FNS-PID, no live feed). All corpus data is local and most inference artefacts are already precomputed — the backend's job is mostly to read, filter, aggregate, and serve, with a narrow live path for user-defined topics.

Read [docs/INFERENCE.md](./INFERENCE.md) before starting — it spells out every loader and feature-construction detail.

---

## Corpus & precomputed artefacts (authoritative paths)

All paths relative to repo root.

| File | Content | Used for |
|---|---|---|
| [data/preprocessed/subset_news.parquet](../data/preprocessed/subset_news.parquet) | Raw news, schema: `id: UInt32, Date: str, date_parsed: Date, Article_title, Stock_symbol, Url, Publisher, Author, Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary` | Source of truth. `Stock_symbol` is already populated per row — NER is only needed for articles the user ingests live. |
| [data/preprocessed/article_embeddings.npy](../data/preprocessed/article_embeddings.npy) + [article_embedding_ids.parquet](../data/preprocessed/article_embedding_ids.parquet) | MiniLM (384-d) embeddings per article, aligned by `id` | Cosine search for user topics; memory-map at startup. |
| [data/predictions/lsa_summaries.parquet](../data/predictions/lsa_summaries.parquet) | `id, lsa_summary` | Redundant with `subset_news.Lsa_summary`; keep it as the durable cache for newly-ingested articles. |
| [data/predictions/full_df_topic_probabilities.parquet](../data/predictions/full_df_topic_probabilities.parquet) | `id, prob_<label_13>` float32 | 13-topic zero-shot probabilities, precomputed. |
| [data/predictions/finbert_embeddings.parquet](../data/predictions/finbert_embeddings.parquet) | `id, sentiment, score, emb_0…emb_767` | FinBERT sentiment + 768-d CLS embeddings. |
| [data/predictions/xgb_tb_predictions.parquet](../data/predictions/xgb_tb_predictions.parquet) | `id, ticker, date, proba_up, pred` | Triple-barrier direction signal per article. |
| [data/Stock_price/stock_price.parquet](../data/Stock_price/stock_price.parquet) | Daily OHLCV per ticker | Market indicator features + price-chart overlay. |

**63-ticker universe** — the entire pipeline is scoped to the `POOL` defined in [notebook/03_NER.ipynb](../notebook/03_NER.ipynb). Do **not** try to handle arbitrary S&P 500 tickers; the NER model's `LabelEncoder` and XGBoost's `feat_cols` are both fixed to this universe.

---

## Models to integrate (already trained — load, don't retrain)

### 1. MiniLM embedder — [models/all-MiniLM-L6-v2/](../models/all-MiniLM-L6-v2/)
Used for (a) the zero-shot topic classifier and (b) article/query cosine retrieval. Load once:
```python
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("models/all-MiniLM-L6-v2", device="cpu")
```

### 2. Zero-shot topic classifier (inline class, no separate artefact)
Fixed 13-topic label set. The trained corpus-level probabilities already live in [full_df_topic_probabilities.parquet](../data/predictions/full_df_topic_probabilities.parquet); the class is only needed for **new** (user-ingested) articles or user-defined ad-hoc topic queries.

Labels + descriptions + the `ZeroShotTopicClassifier` implementation are in [docs/INFERENCE.md §3](./INFERENCE.md). Document prefix at encode time: `"[<TICKER>] [<DATE>] <TITLE> | <LSA_SUMMARY>"`.

### 3. BERTopic — [models/bertopic_news/](../models/bertopic_news/)
Already trained on the full corpus with the same 13 zero-shot anchors. Load via `BERTopic.load(..., embedding_model=embed_model)`. Use it for:
- **Corpus-level cluster inspection** in an admin view (`get_topic_info()`).
- Not the per-article scoring — `full_df_topic_probabilities.parquet` is canonical for that.

**Do not** surface "auto-discovered clusters" as a separate UI concept. The corpus was trained with the 13 labels as zero-shot anchors, so residual HDBSCAN topics are small and noisy. If you want a "⚡ discovered" flavour, use the top-k BERTopic keywords for the same 13 labels (via `get_topic(topic_id)`) as label enrichment, not as new topics.

### 4. NER ticker classifier — [models/ticker_ner_tfidf_2.joblib](../models/ticker_ner_tfidf_2.joblib)
A `joblib` **dict** bundle: `{"vectorizer": HashingVectorizer, "clf": SGDClassifier, "le": LabelEncoder}`. Input = full article (title + body). Output = one ticker from the 63-ticker pool + softmax confidence. See [docs/INFERENCE.md §1](./INFERENCE.md).

> Do **not** use rapidfuzz / alias tables / S&P 500 constituent lookups. Those approaches were benchmarked in [notebook/03_NER.ipynb](../notebook/03_NER.ipynb) and lost to the TF-IDF classifier — that is the single production path.

### 5. FinBERT — [models/finbert_model/](../models/finbert_model/), [finbert_tokenizer/](../models/finbert_tokenizer/), [finbert_sentiment/](../models/finbert_sentiment/)
`AutoModel` for 768-d CLS embeddings + HF `pipeline("text-classification")` for sentiment. Pre-applied; you only need this for live-ingested articles. See [docs/INFERENCE.md §4](./INFERENCE.md).

### 6. XGBoost direction signal — [models/xgb_tb/](../models/xgb_tb/)
Bundle: `xgb_tb.pkl`, `scaler.pkl`, `feat_cols.pkl`, `threshold.pkl`. Output: `proba_up` and binary `pred`. Prebaked predictions are in [xgb_tb_predictions.parquet](../data/predictions/xgb_tb_predictions.parquet) — serve from there.

> ⚠️ The embedding PCA used to go from 768-d FinBERT to the `pc_0…pc_9` features is currently **not** persisted (notebook 09 refits it each run). For the backend, treat `xgb_tb_predictions.parquet` as the canonical signal and only recompute for articles *already present* in `finbert_embeddings.parquet`. If you need live XGBoost inference on brand-new articles, first persist `emb_pipeline.pkl` by re-running notebook 09's §"Train final XGBoost" cell with `joblib.dump(Pipeline([("sc", StandardScaler()), ("pca", PCA(10))]).fit(emb_mat), "models/xgb_tb/emb_pipeline.pkl")`.

---

## Ingestion & caching strategy

The backend serves a **closed corpus**: on startup, load precomputed artefacts into memory. Live inference is only needed when:
1. The user **creates a new topic** → embed the query string with MiniLM, cosine-score against the cached `article_embeddings.npy`, aggregate weekly.
2. The user **uploads / pastes a new article** (optional feature) → run the full pipeline (NER → LSA → zero-shot → FinBERT → append). This is out of scope for v1 unless explicitly prioritised.

**Weekly aggregation**: use `date_parsed` (already a `Date` column — no parsing needed). The rolling window is the last 24 ISO weeks relative to `subset_news["date_parsed"].max()`. Precompute and cache the `(topic_id, iso_week) → intensity` grid at startup; it is ~13 topics × 24 weeks = 312 cells.

**Per-ticker relevance**: join `subset_news` with `full_df_topic_probabilities` on `id`, group by `Stock_symbol`, take the weighted mean of `prob_<topic>` × (optionally `proba_up` from XGBoost for a "conviction-weighted" variant).

---

## Backend — FastAPI

Run on `:8000`. Lazy-load every heavy model (MiniLM, BERTopic, NER, FinBERT) on first request that needs it; article embeddings + parquet caches load at startup (memory-mapped for the `.npy`).

### Endpoints

```
GET  /health                         → { status, models_loaded: { miniLM, bertopic, ner, finbert, xgb } }

GET  /topics                         → { topics: [{ id, label, description, kind: "fixed"|"user", color }] }
                                       Seeded with the 13 fixed zero-shot labels. User topics appended.
POST /topics                         → body: { "label": "central bank liquidity" }
                                       Backend embeds the label with MiniLM, stores in an in-process dict + SQLite
                                       table (topics.db), returns the new topic with a generated id.
DELETE /topics/{topic_id}            → 204. Cannot delete `kind: "fixed"`.

GET  /timeline?topic_id=...&weeks=24 → [{ iso_week, week_start, intensity, article_count }]
                                       For fixed topics: mean of prob_<label> per ISO week over last `weeks` weeks.
                                       For user topics: mean cosine sim (clipped to [0,1]) between query embedding
                                       and MiniLM article embeddings per ISO week.

GET  /phase?topic_id=...             → { phase, label, emoji, pct_of_peak, momentum,
                                         relative_magnitude, cycle: { peak_week_estimate, sigma, cycle_progress_pct } }

GET  /stocks?topic_id=...&top_n=20   → [{ ticker, score, article_count, proba_up_mean, top_headlines:[{id,title,date,url}] }]
                                       Articles filtered to the 24-week window; score = mean topic relevance × (1 + |mean proba_up - 0.5|).

GET  /matrix?tickers=AAPL,JPM,NVDA   → { rows: tickers, cols: topic_ids, values: [[...]] }
                                       mean topic relevance per (ticker, topic) over the 24-week window.

GET  /articles?topic_id=...&iso_week=...&top_n=5
                                     → [{ id, title, date, ticker, url, snippet, relevance, sentiment, proba_up }]
                                       Drill-down source: for fixed topics rank by prob_<label>; for user topics rank
                                       by cosine sim to query embedding.

GET  /seasonality?topic_id=...       → [{ week_of_year, intensity, n_years }]
                                       Aggregate intensity by ISO week-of-year across the whole corpus (not just 24w).

GET  /price?ticker=AAPL&weeks=24     → [{ date, close, volume }]   overlay for the ticker drill-down.
```

Add CORS for `http://localhost:3000`. All responses are JSON; dates are ISO strings.

### Phase detector

`backend/phase_detector.py`:

```python
import numpy as np
from scipy.optimize import curve_fit

def _gauss(x, a, mu, sigma): return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def detect_phase(curve: list[float], current_week: int, window: int = 3) -> dict:
    c = np.asarray(curve, dtype=float)
    peak = float(c.max()) or 1e-9
    rel = float(c[current_week] / peak)

    lo = max(0, current_week - window)
    mid = max(0, current_week - 2 * window)
    momentum = float(c[lo:current_week + 1].mean() - c[mid:lo].mean()) if lo > mid else 0.0

    if rel < 0.25:                          phase, label, emoji = "dormant", "Dormant", "🌑"
    elif momentum > 0.05:                   phase, label, emoji = "rising",  "Rising",  "📈"
    elif momentum < -0.05 and rel > 0.4:    phase, label, emoji = "fading",  "Fading",  "📉"
    else:                                   phase, label, emoji = "peak",    "Peak",    "🔥"

    # Gaussian fit to the most recent local-max region.
    cycle = {"peak_week_estimate": None, "sigma": None, "cycle_progress_pct": None}
    try:
        peak_idx = int(np.argmax(c[max(0, current_week - 12):current_week + 1])) + max(0, current_week - 12)
        lo_f, hi_f = max(0, peak_idx - 4), min(len(c), peak_idx + 5)
        x = np.arange(lo_f, hi_f); y = c[lo_f:hi_f]
        if len(x) >= 5:
            (a, mu, sigma), _ = curve_fit(_gauss, x, y, p0=[y.max(), peak_idx, 2.0], maxfev=2000)
            progress = float((current_week - mu) / max(abs(sigma), 1e-6))
            cycle = {"peak_week_estimate": float(mu), "sigma": float(abs(sigma)),
                     "cycle_progress_pct": float(np.clip(progress, -3, 3))}
    except Exception:
        pass

    return {"phase": phase, "label": label, "emoji": emoji,
            "pct_of_peak": rel, "momentum": momentum,
            "relative_magnitude": rel, "cycle": cycle}
```

---

## Frontend — React + Vite + Recharts

### Data layer
- `@tanstack/react-query` for fetching. **Polling is off by default** (the corpus is static). Enable 60s polling only on `/timeline` + `/phase` when a debug flag is on.
- Optimistic insert on `POST /topics` (append with a temp id, swap on success).
- Loading skeletons: shimmer on `#1e293b`.

### New UI feature — Article drill-down
Clicking a week bar (seasonality) or lifecycle-chart point opens a slide-in right panel with `GET /articles?topic_id=...&iso_week=...&top_n=5`. Show per article: ticker chip, MiniLM cosine or fixed-topic prob (whichever applies), FinBERT sentiment chip, truncated title + snippet, XGBoost `proba_up` badge (green ≥ threshold, red < threshold), external link to `Url`. No full page reload.

### Keep existing look
IBM Plex Mono + Syne, `#020817` background, dark terminal feel.

---

## Feasibility of analyses & plots

Each prototype visualisation is only kept if the corpus genuinely supports it — otherwise drop it.

| Chart | Feasible? | Notes |
|---|---|---|
| **Weekly intensity curve (24w)** | ✅ | `prob_<label>` is already per-article; group-by ISO week. |
| **Phase / momentum badge** | ✅ | Deterministic from the 24-week curve. |
| **Seasonality heatmap (week-of-year)** | ⚠ conditional | Corpus must span ≥ 2 calendar years for this to be meaningful. Check `subset_news["date_parsed"].min()` / `max()` at startup; if span < 104 weeks, hide the seasonality tab and emit a console warning instead of faking it. |
| **Per-topic top tickers** | ✅ | Join + group-by. |
| **Topic × ticker heatmap** | ✅ | Small matrix (13 × N). |
| **Gaussian cycle-fit overlay** | ⚠ | Needs a clean local max; fall back to reporting `null` for `peak_week_estimate` if `curve_fit` fails (handled in the code above). |
| **Price overlay on ticker drill-down** | ✅ | Served from `stock_price.parquet`. |
| **XGBoost conviction weighting** | ✅ | Merge `xgb_tb_predictions.parquet`. |
| **User-defined topics via MiniLM + cosine** | ✅ | The `POST /topics` flow below: embed the label with MiniLM, score cosine against `article_embeddings.npy`, aggregate weekly. Fully supported — this is the "discover a new topic" UX. |
| **BERTopic HDBSCAN residuals as separate auto-clusters** | ❌ | Different from user topics. The saved model is zero-shot-locked to the 13 anchors; residuals are small and noisy. If you want corpus-level discovery, spin up a *fresh* BERTopic (no `zeroshot_topic_list`) on the cached embeddings — treat as an offline admin job, not a live endpoint. |
| **Live FNS-PID stream** | ❌ | No external feed in this repo. Remove from scope. |

---

## Environment & tooling

```
Python 3.11+ (conda env in environment.yml)
fastapi uvicorn[standard]
polars pyarrow
sentence-transformers transformers torch
bertopic umap-learn hdbscan
scikit-learn xgboost joblib
sumy nltk  # + python -m nltk.downloader punkt
scipy numpy pandas
rapidfuzz  # only if you add alias lookup as a fallback

Node 20+
React 18, Vite
@tanstack/react-query recharts
```

### Makefile

```makefile
install:        ## install backend + frontend deps
	pip install -e .
	cd frontend && npm install

models:         ## verify all local model artefacts are present
	python backend/scripts/check_models.py

warmup:         ## precompute 24w intensity grid + topic×ticker matrix, write to cache/
	python backend/scripts/warmup.py

dev:            ## run FastAPI on :8000 and Vite on :3000 concurrently
	(cd backend && uvicorn main:app --reload --port 8000) & \
	(cd frontend && npm run dev)
```

Do **not** add a `make models` target that downloads from HuggingFace — all weights already live in [models/](../models/). `make models` just verifies presence.

---

## Deliverables

```
backend/
  main.py                  # FastAPI app, CORS, lifespan event loads caches
  config.py                # paths, pool, topic labels/descriptions (single source of truth)
  loaders.py               # lazy getters for MiniLM / BERTopic / NER / FinBERT / XGB bundles
  data_store.py            # polars reads of parquet caches, memmap of article_embeddings.npy
  services/
    topics.py              # CRUD + MiniLM embedding of user topics (SQLite persistence)
    timeline.py            # weekly intensity, fixed vs user topic paths
    phase_detector.py      # (see code above)
    stocks.py              # per-ticker aggregates joined with xgb predictions
    articles.py            # drill-down ranking
    ingest.py              # (optional v2) NER → LSA → zero-shot → FinBERT for new articles
  scripts/
    check_models.py
    warmup.py
frontend/
  src/
    App.jsx                # real API calls, React Query, no mocks
    hooks/{useTopics,useTimeline,usePhase,useStocks,useArticles}.js
    components/
      LifecycleChart.jsx
      SeasonalityHeatmap.jsx
      TickerMatrix.jsx
      ArticlePanel.jsx     # right-hand drill-down
      SkeletonCard.jsx
      TopicComposer.jsx    # POST /topics
Makefile
README.md                  # startup in ≤ 5 commands
```

`README.md`: clone → `make install` → `make models` → `make warmup` → `make dev`. Document that the corpus is frozen at the latest date in `subset_news` and that "live" applies to user-defined topics only.

---

## Constraints & sharp edges

- **Never re-embed the corpus on a request.** `article_embeddings.npy` is ~400 MB; memory-map with `np.load(..., mmap_mode="r")` at startup. Query → dot product → top-k is O(N·D) and runs in < 1 s on CPU for ~1M rows.
- **User-topic intensity**: normalise cosine similarity to `[0, 1]` with `(sim + 1) / 2` before weekly aggregation so user topics and fixed topics are on a comparable scale in the UI.
- **Week alignment**: always use `date_parsed.iso_calendar().week` + `.year` as the join key. The year component matters for `seasonality`.
- **Ticker filtering**: every `/timeline`, `/stocks`, `/matrix` call must restrict to the 63-ticker `POOL`. A one-shot `pool = pl.read_parquet(...)["Stock_symbol"].unique()` at startup is fine.
- **SQLite for user topics**: `topics.db` with `(id TEXT PK, label TEXT, description TEXT, embedding BLOB, created_at)`. Survives restarts. Fixed topics are not stored — they're hard-coded in `config.py`.
- **BERTopic lazy-load**: UMAP + HDBSCAN deps are heavy; load on first `/admin/*` hit, not at startup.
- **No force-retraining anything.** If an artefact appears stale, surface a warning from `/health` and continue serving.

---

## Verification checklist (agent must self-check before declaring done)

- [ ] `/health` returns 200 with every artefact path in `models/` confirmed to exist.
- [ ] `/timeline?topic_id=big_tech_and_software` returns a 24-point array whose sum matches a polars query run directly against `full_df_topic_probabilities.parquet`.
- [ ] `POST /topics { "label": "central bank liquidity" }` → `GET /timeline?topic_id=<new>` completes in < 3 s end-to-end.
- [ ] `/stocks?topic_id=oil_gas_and_energy` puts `XOM` and `CVX` in the top 5.
- [ ] Drill-down on any bar opens the slide-in panel with 5 real articles (non-empty `Url`, matching `Stock_symbol`).
- [ ] Bundling the frontend and starting `make dev` gives a working app at `http://localhost:3000` with zero mock data imports remaining.

The key seams to watch: **ingest → cache → serve** (read-only for v1; no mutation of the parquet files at runtime), and the **user-topic live path** (MiniLM encode + matmul + weekly group-by — keep it in-process).
