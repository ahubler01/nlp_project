# AGENT TASK — FinLens: Financial News Intelligence Dashboard (Student Project)

## Objective

Build **FinLens**, a simple, clean local web app that lets a user **explore financial news topics one view at a time**. This is a student project — prioritise clarity and simplicity over feature density. No complex dashboards, no crammed terminal UI. One clean white-themed page, a sidebar to pick the view, and a single focused panel on the right.

Wire the **already-trained models** in [models/](../models/) into a FastAPI backend, read from [data/preprocessed/subset_news.parquet](../data/preprocessed/subset_news.parquet), and serve everything locally. No external datasets. No live feeds.

Read [docs/INFERENCE.md](./INFERENCE.md) before starting — it explains the loaders and feature construction.

---

## Design principles (read first)

1. **One thing at a time.** The user picks a view from a sidebar — they never see everything at once.
2. **White, light theme.** Clean background (`#ffffff` / `#f7f7f8`), dark text (`#111827`), a single accent color (`#2563eb` blue). Use `Inter` or system-ui font. No fancy typography.
3. **Simple components.** Prefer plain HTML + CSS and basic Recharts charts. No custom animations, no glassmorphism, no gradients.
4. **Student-grade code.** Readable Python and React. Obvious variable names. No premature abstraction. Short functions. Comments only where a reader would be confused.
5. **Fail loudly, not silently.** If an artefact is missing, show a clear error card in the UI rather than a half-loaded chart.

---

## The 4 views (and only these)

The sidebar has exactly **four items**. Each maps to one page. Nothing else.

### 1. Topic Explorer
- User picks one topic from a dropdown (13 fixed zero-shot topics + any user-defined ones).
- Shows:
  - A single line chart: **weekly intensity over the last 24 weeks**.
  - A phase badge (🌑 Dormant / 📈 Rising / 🔥 Peak / 📉 Fading).
  - A short list of the **top 5 tickers** for that topic (ticker, article count, mean `proba_up`).
- Clicking any week on the chart opens a small modal with up to 5 headlines from that week.

### 2. Topic Graph (cointegration network)
- A node-link graph where **nodes = topics** and **edges = cointegration** of their weekly intensity series.
- Compute pairwise **Engle–Granger cointegration** (`statsmodels.tsa.stattools.coint`) on the 24-week intensity curves. Draw an edge when `p_value < 0.05`. Edge thickness inversely proportional to `p_value`.
- Render with a lightweight force layout library (`react-force-graph-2d` or plain D3 if simpler). Node color = topic category. Hover = label + p-values of incident edges.
- Precompute the adjacency matrix at startup and serve it as JSON — do **not** recompute per request.

### 3. Chat with the corpus
- A chat panel where the user enters a topic + time window (natural language, e.g. *"Summarise oil & gas news for March 2025"*) and gets back a summary.
- Backend parses topic + date range, retrieves matching articles, and returns:
  - A **concatenated extractive summary** built from the pre-computed `Lsa_summary` / `Textrank_summary` columns in `subset_news.parquet`. Concatenate the top-N article summaries (ranked by topic probability), then run **sumy LSA** once more over that concatenation to produce a final 5-sentence summary.
  - Aggregate **FinBERT sentiment** (share of positive / neutral / negative) over the window.
  - List of the 5 source articles (title, ticker, date, URL).
- **No external LLM.** Only local models already in [models/](../models/) + the already-computed summary columns. Keep it simple: this is retrieval + extractive summarisation, not generative.

### 4. Ticker Browser
- User picks a ticker from the 63-ticker pool.
- Shows:
  - Daily close price chart (24 weeks) from [data/Stock_price/stock_price.parquet](../data/Stock_price/stock_price.parquet).
  - Top 3 topics that mention this ticker.
  - Last 5 articles for this ticker (title, date, sentiment chip, URL).

That's it. If a feature doesn't fit in one of these four pages, it doesn't ship.

---

## Corpus & precomputed artefacts

All paths are relative to repo root.

| File | Content | Role |
|---|---|---|
| [data/preprocessed/subset_news.parquet](../data/preprocessed/subset_news.parquet) | Raw news + summaries. Columns include `id, Date, date_parsed, Article_title, Stock_symbol, Url, Publisher, Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary` | Source of truth |
| [data/preprocessed/article_embeddings.npy](../data/preprocessed/article_embeddings.npy) + [article_embedding_ids.parquet](../data/preprocessed/article_embedding_ids.parquet) | MiniLM 384-d article embeddings | User-topic cosine retrieval |
| [data/predictions/full_df_topic_probabilities.parquet](../data/predictions/full_df_topic_probabilities.parquet) | 13-topic zero-shot probabilities | Topic intensity + ticker scoring |
| [data/predictions/finbert_embeddings.parquet](../data/predictions/finbert_embeddings.parquet) | FinBERT sentiment + 768-d CLS | Sentiment chips + chatbot aggregation |
| [data/predictions/xgb_tb_predictions.parquet](../data/predictions/xgb_tb_predictions.parquet) | `proba_up`, `pred` per (ticker, date) | Top-ticker scoring |
| [data/Stock_price/stock_price.parquet](../data/Stock_price/stock_price.parquet) | Daily OHLCV | Price chart |

63-ticker universe = the `POOL` defined in [notebook/03_NER.ipynb](../notebook/03_NER.ipynb). Do not handle tickers outside this pool.

---

## Models (load — don't retrain)

| Model | Path | Used where |
|---|---|---|
| MiniLM embedder | [models/all-MiniLM-L6-v2/](../models/all-MiniLM-L6-v2/) | User topic queries |
| Zero-shot classifier (inline class) | see [INFERENCE.md §3](./INFERENCE.md) | Only for user-defined topics |
| BERTopic | [models/bertopic_news/](../models/bertopic_news/) | Optional: keyword enrichment only |
| NER TF-IDF bundle | [models/ticker_ner_tfidf_2.joblib](../models/ticker_ner_tfidf_2.joblib) | Only if you add live article ingestion (out of scope for v1) |
| FinBERT | [models/finbert_model/](../models/finbert_model/) + tokenizer + sentiment pipeline | Sentiment aggregation in chat view |
| XGBoost direction signal | [models/xgb_tb/](../models/xgb_tb/) + prebaked predictions | Ticker ranking |

Lazy-load heavy models on first request. Load precomputed parquets at startup. Memory-map `article_embeddings.npy`.

---

## Backend — FastAPI on `:8000`

Keep the API small. Eight endpoints total.

```
GET  /health                         → { status, artefacts_present: {...} }

GET  /topics                         → list of fixed + user topics
POST /topics   { "label": "..." }    → embed with MiniLM, save to topics.db, return topic
DELETE /topics/{id}                  → only for user topics

GET  /timeline?topic_id=...          → 24-week intensity + phase + top-5 tickers (Topic Explorer)

GET  /graph                          → { nodes:[{id,label}], edges:[{source,target,weight,pvalue}] }
                                       Precomputed cointegration adjacency (Topic Graph view)

POST /chat    { "query": "..." }     → { summary, sentiment:{pos,neu,neg}, articles:[...] }
                                       Parses topic + date window, retrieves, summarises via sumy LSA

GET  /ticker?symbol=AAPL             → { prices:[...], top_topics:[...], recent_articles:[...] }
                                       Ticker Browser view
```

CORS: allow `http://localhost:3000`. Dates are ISO strings. Keep response shapes small and obvious.

### Cointegration graph (precompute at startup)

```python
from statsmodels.tsa.stattools import coint

def build_topic_graph(weekly_intensity: dict[str, np.ndarray]) -> dict:
    topics = list(weekly_intensity)
    edges = []
    for i, a in enumerate(topics):
        for b in topics[i+1:]:
            _, pvalue, _ = coint(weekly_intensity[a], weekly_intensity[b])
            if pvalue < 0.05:
                edges.append({"source": a, "target": b,
                              "pvalue": float(pvalue),
                              "weight": float(1.0 - pvalue)})
    nodes = [{"id": t, "label": t} for t in topics]
    return {"nodes": nodes, "edges": edges}
```

### Chat endpoint (student-simple pipeline)

1. Parse the query with a tiny regex / rule parser to extract (a) topic keywords and (b) a date range. If no date range is found, default to the last 4 weeks.
2. Match the topic: cosine similarity between MiniLM-embedded query and the 13 fixed topic labels; pick the best match (threshold 0.35, else fall back to keyword overlap).
3. Filter `subset_news` to `date_parsed` in range AND top-N articles by `prob_<topic>`.
4. Concatenate their `Lsa_summary` fields, run sumy LSA on the concatenation → 5 sentences.
5. Aggregate FinBERT sentiment counts over those articles.
6. Return summary + sentiment + the 5 source articles.

No streaming, no token-by-token output. Just a single JSON response.

### Phase detector

Keep the Gaussian-fit phase detector from the previous version — but only use the simple outputs (`phase`, `label`, `emoji`, `pct_of_peak`, `momentum`). Skip the cycle-fit fields in the UI; they are distracting for a student project.

---

## Frontend — React + Vite

### Layout
- **Left sidebar** (fixed, 220 px): app name "FinLens", then four nav links:
  1. Topic Explorer
  2. Topic Graph
  3. Chat
  4. Tickers
- **Main area**: only the currently selected view. No secondary panels. No drawers. Just the view.

### Theme
- Background: `#ffffff`. Card background: `#f9fafb`. Border: `#e5e7eb`.
- Text: `#111827` primary, `#6b7280` secondary.
- Accent: `#2563eb` for active nav, buttons, chart line.
- Sentiment chips: green `#16a34a`, grey `#6b7280`, red `#dc2626`.
- Font: `Inter, system-ui, sans-serif`. No custom font loading unless trivial.

### Components (one file each, small)
```
src/
  App.jsx                  # router + sidebar
  pages/
    TopicExplorer.jsx
    TopicGraph.jsx
    Chat.jsx
    Tickers.jsx
  components/
    Card.jsx
    Sentimentchip.jsx
    PhaseBadge.jsx
    LineChart.jsx          # thin wrapper around Recharts
  api.js                   # fetch helpers (plain fetch, no React Query needed)
```

React Query is **optional**. For a student project, plain `useEffect + useState + fetch` is fine and easier to read. Do **not** introduce state management libraries.

### Graph view
Use `react-force-graph-2d`. Node radius proportional to total topic intensity. Edge width proportional to `weight` (= `1 - pvalue`). Click a node → navigate to Topic Explorer for that topic.

### Chat view
- Single input box at the top (full width).
- Below: last response as three stacked cards — **Summary** (paragraph), **Sentiment** (three coloured bars), **Sources** (list of 5 articles).
- No multi-turn memory. Each query is independent. Keep it simple.

---

## Project layout

```
backend/
  main.py                  # FastAPI app + CORS + startup hook
  config.py                # paths, POOL, topic labels
  loaders.py               # lazy model loaders
  data_store.py            # parquet + mmap loads at startup
  services/
    topics.py              # topic CRUD (SQLite for user topics)
    timeline.py            # 24-week intensity, fixed + user paths
    phase_detector.py
    graph.py               # cointegration adjacency (precomputed)
    chat.py                # retrieve + summarise
    tickers.py             # ticker browser aggregates
  scripts/
    check_models.py
    warmup.py              # precompute intensity grid + graph
frontend/
  src/ (see above)
Makefile
README.md                  # ≤ 5 commands to run
```

### Makefile

```makefile
install:
	pip install -e .
	cd frontend && npm install

models:
	python backend/scripts/check_models.py

warmup:
	python backend/scripts/warmup.py

dev:
	(cd backend && uvicorn main:app --reload --port 8000) & \
	(cd frontend && npm run dev)
```

---

## Environment

```
Python 3.11
fastapi uvicorn[standard]
polars pyarrow
sentence-transformers transformers torch
scikit-learn xgboost joblib
sumy nltk                # + python -m nltk.downloader punkt
statsmodels              # for cointegration
scipy numpy pandas

Node 20
React 18, Vite, Recharts, react-force-graph-2d
```

No Tailwind required — plain CSS modules are fine. If the agent prefers Tailwind, use it only with default tokens (no custom palette beyond the theme colors above).

---

## Hard constraints

- **No mock data** in the frontend. Every number comes from the FastAPI backend.
- **No external API calls.** Everything runs locally.
- **Don't re-embed the corpus per request.** Memory-map `article_embeddings.npy`.
- **Restrict every ticker operation to the 63-ticker POOL.**
- **Cointegration is precomputed once at startup**, cached in `cache/topic_graph.json`.
- **Keep files short.** No backend file longer than ~200 lines. No React component longer than ~150 lines. If it grows past that, split it.
- **No feature not listed in the 4 views.** If something is interesting but doesn't fit, leave it out.

---

## Verification checklist

- [ ] `make dev` launches the app at `http://localhost:3000` against a working backend on `:8000`.
- [ ] The sidebar has exactly 4 items; each navigates to a single uncluttered page.
- [ ] Theme is white/light. No dark backgrounds anywhere.
- [ ] Topic Explorer renders a 24-week intensity chart + phase badge + top 5 tickers for `big_tech_and_software`.
- [ ] Topic Graph renders a node-link diagram with at least a few edges (cointegration p < 0.05 pairs).
- [ ] Chat returns a 5-sentence summary, sentiment breakdown, and 5 source articles for the query *"Summarise oil & gas news in the last month"*.
- [ ] Ticker Browser for `AAPL` shows a 24-week price line + top topics + 5 recent articles.
- [ ] No frontend file imports mock/fake data.
