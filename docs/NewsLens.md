# NewsLens

**A media intelligence platform that surfaces how news narratives shape public perception of companies — and what that means for how assets move together.**

---

## Table of Contents

1. [Product Overview](#1-product-overview)
2. [Dataset](#2-dataset)
3. [Page 1 — The Observatory](#3-page-1--the-observatory)
4. [Page 2 — The Newsroom](#4-page-2--the-newsroom)
5. [Model & Pipeline Architecture](#5-model--pipeline-architecture)
   - 5.1 [NER + Ticker Resolution](#51-ner--ticker-resolution)
   - 5.2 [Sentence Embeddings](#52-sentence-embeddings)
   - 5.3 [Multi-task NLP Model (FinBERT)](#53-multi-task-nlp-model-finbert)
   - 5.4 [Topic Model (BERTopic)](#54-topic-model-bertopic)
   - 5.5 [Cointegration Graph](#55-cointegration-graph)
   - 5.6 [SHAP Explainability Layer](#56-shap-explainability-layer)
   - 5.7 [Article Similarity Graph](#57-article-similarity-graph)
6. [Derived Metrics & Scatter Axes](#6-derived-metrics--scatter-axes)
7. [Full Data Flow Diagram](#7-full-data-flow-diagram)
8. [Tech Stack](#8-tech-stack)
9. [Project Approach](#9-project-approach)

---

## 1. Product Overview

NewsLens is a two-screen analytical tool built on top of financial news data. It answers two questions:

- **Which companies are structurally linked** — by their price movements and by the stories told about them?
- **What is the news actually saying** — and which specific words and narratives drive the signals we compute?

The platform is not a prediction engine for trading. It is an **explainability and intelligence tool** for analysts, researchers, and portfolio managers who need to understand the information landscape around a set of assets. Every score shown in the interface can be traced back to the exact words that produced it, via SHAP explanations.

---

## 2. Dataset

### Primary source: FNSPID

The Financial News and Stock Price Integration Dataset contains 15.7 million time-aligned financial news records for 4,775 S&P500 companies, covering 1999–2023. Each record has the following schema:

| Field | Type | Description |
|---|---|---|
| `Date` | String | Publication date |
| `Article_title` | String | Headline |
| `Stock_symbol` | String | Associated ticker |
| `Publisher` | String | News outlet |
| `Author` | String | Journalist |
| `Article` | String | Full article body |
| `Lsa_summary` | String | Extractive summary (LSA) |
| `Luhn_summary` | String | Extractive summary (Luhn) |
| `Textrank_summary` | String | Extractive summary (TextRank) |
| `Lexrank_summary` | String | Extractive summary (LexRank) |

### Augmentation: market price data

Historical OHLCV (open, high, low, close, volume) data for each ticker is available from the existing market dataset. This is used to:

- Compute cointegration between ticker pairs (Page 1 graph)
- Generate binary labels for the future-movement prediction head (price movement ≥ ±1.5% within 3 days of publication)

No other external sources are required.

---

## 3. Page 1 — The Observatory

The landing page has two side-by-side panels and a statistics bar below.

```
┌─────────────────────────────┬──────────────────────────────┐
│                             │                              │
│    Ticker Knowledge Graph   │    NLP Scatter Plot          │
│    (cointegration edges)    │    (configurable axes)       │
│                             │                              │
└─────────────────────────────┴──────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│   Market statistics bar           [ Newsroom → ]             │
└──────────────────────────────────────────────────────────────┘
```

### Left panel: Ticker Knowledge Graph

**What it shows.** A force-directed graph where each node is a ticker. Edges connect pairs of tickers that are **cointegrated** — meaning their price series share a long-term structural relationship. Tickers that drift apart tend to converge back, making them behave like a system rather than two independent assets.

**Why this is useful.** A portfolio analyst can see at a glance which holdings are redundant (cointegrated = moving together) and which provide genuine diversification. This is a more robust signal than correlation, which is short-term and unstable.

**Visual encoding:**

- Node size: volume of news coverage (number of articles in the selected time window)
- Node color: dominant topic cluster from BERTopic (each cluster gets a distinct color)
- Edge thickness: strength of cointegration (the Engle-Granger test statistic)
- Community groups: detected via the Louvain algorithm and shown as soft background regions

Clicking a node highlights it, shows its ticker profile card in a small overlay, and populates the market statistics bar below.

### Right panel: NLP Scatter Plot

**What it shows.** Every ticker in the selected universe is a dot on a 2D plane. Both axes are NLP-derived metrics produced by the model pipeline. The user selects which metric to place on each axis from a dropdown.

**Available axes:**

| Metric | What it measures | Source |
|---|---|---|
| Sentiment score | Average tone of coverage (-1 to +1) | FinBERT regression head |
| Controversy score | Divergence of sentiment across publishers | FinBERT classifier head |
| Narrative intensity | Volume of recent coverage (normalized) | Aggregation |
| Topic diversity | Entropy of topic distribution | BERTopic aggregation |

A ticker in the top-right of a sentiment × controversy plot is being covered positively but divisively — publishers disagree about the narrative. A ticker in the bottom-left is being ignored and covered negatively. These quadrants are interpretable without any financial knowledge.

### Statistics bar

A row of simple metric cards computed from market and article data for the selected ticker:

- Article volume (last 7 days)
- Sentiment trend (direction of change over the last 7 days)
- Number of unique publishers
- Average daily return over the selected window (from market data)

### Newsroom button

Takes the user to Page 2 with the currently selected ticker(s) as context.

---

## 4. Page 2 — The Newsroom

The drill-down page focuses on individual articles for the selected ticker(s). It has a graph panel on the left and an analytics panel on the right.

```
┌──────────────────────────────┬──────────────────────────────┐
│                              │                              │
│    Article Graph             │    SHAP Explanation          │
│    (nodes = articles)        │    Top articles by signal    │
│    Edge toggle:              │    Aggregate NLP stats       │
│    - Semantic similarity     │                              │
│    - Concept filter          │                              │
│                              │                              │
└──────────────────────────────┴──────────────────────────────┘
```

### Left panel: Article Graph

Each node is an individual article. Edges connect articles based on the selected edge type.

**Edge type 1 — Semantic similarity (default).** Two articles are connected if the cosine similarity of their sentence embeddings exceeds a threshold (typically 0.6). This produces clusters of articles covering the same event or narrative thread. The clusters emerge without any manual labeling — they are a direct result of the geometry of the embedding space.

**Edge type 2 — Concept edges.** The user selects a predefined concept (earnings, regulatory action, leadership change, product launch, legal dispute). Each concept is represented by a small set of seed sentences. Articles are connected if both score above a threshold on that concept's similarity score. This gives a filtered view: "show me only articles connected through the earnings narrative."

**Node encoding:**

- Node color: sentiment of the article (teal = positive, coral = negative, gray = neutral)
- Node size: predicted future-movement probability (the article's score from the movement prediction head)
- Node border: publisher identity (different dash patterns for top publishers)

Clicking a node loads its detail in the right panel.

### Right panel: three sections

**Section 1 — SHAP explanation.** When an article node is selected, this section shows a horizontal waterfall bar chart. Each bar is a token from the article. Bars extending right (teal) are tokens that pushed the sentiment score upward; bars extending left (coral) pushed it downward. The length of each bar is the SHAP value magnitude. The chart answers the question: *why does the model think this article is positive or negative?*

This is precomputed offline for every article and cached. At display time it is a simple lookup and D3 render — no live inference required.

**Section 2 — Top articles by predicted movement.** Articles for the selected ticker are ranked by their predicted future-movement probability (output of the movement prediction head). They are grouped by concept tag (earnings, regulation, etc.) and displayed as a ranked list within each group. This answers: *which articles historically precede significant market activity for this ticker?*

**Section 3 — Aggregate NLP statistics.** A compact dashboard for the selected ticker:

- Sentiment trend sparkline (daily average over the last 30 days)
- Top 5 named entities mentioned alongside this ticker (from NER)
- Publisher breakdown (bar chart of article volume by outlet)
- Topic distribution (which BERTopic topics dominate the coverage)

---

## 5. Model & Pipeline Architecture

All models are run **offline** in a preprocessing pipeline. The results are stored in a database and served to the frontend via a REST API. No heavy inference happens at query time — the UI is fast because all expensive computation is precomputed.

### 5.1 NER + Ticker Resolution

```
Article text
     │
     ▼
┌─────────────────────────────────┐
│  spaCy en_core_web_trf          │
│  Named Entity Recognition       │
│  → extracts ORG entities        │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Ticker lookup table            │
│  (built from FNSPID stock_symbol│
│  field + alias dictionary)      │
│  → maps "Apple" → AAPL         │
└─────────────────────────────────┘
     │
     ▼
  per-article ticker list
  (multi-label, one article
   can mention multiple tickers)
     │
     ▼
┌─────────────────────────────────┐
│  Co-mention index               │
│  ticker_A + ticker_B in same    │
│  article → edge increment       │
└─────────────────────────────────┘
```

**Purpose:** Ground articles to their relevant tickers. The `Stock_symbol` field in FNSPID is the direct mapping for articles that have it, but NER extends coverage to articles that mention additional companies beyond the primary ticker.

**Output stored:** `article_id → [ticker_1, ticker_2, ...]` and co-mention count matrix.

---

### 5.2 Sentence Embeddings

```
Article text (or best extractive summary)
     │
     ▼
┌─────────────────────────────────────┐
│  sentence-transformers              │
│  all-MiniLM-L6-v2                   │
│  384-dimensional dense vector       │
└─────────────────────────────────────┘
     │
     ├──► stored in FAISS index (vector similarity search)
     │
     └──► used as input to BERTopic
```

**Why this model:** It is fast, lightweight (22M parameters), and produces high-quality semantic representations. Running it once over all FNSPID articles is feasible on a single GPU in a few hours.

**Output stored:** `article_id → float[384]` in a FAISS flat index, queryable by cosine similarity.

---

### 5.3 Multi-task NLP Model (FinBERT)

This is the central trained model of the system. It is a fine-tuned `ProsusAI/finbert` with **three output heads** sharing a common BERT encoder. Training once gives three outputs simultaneously.

```
Article text
     │
     ▼
┌──────────────────────────────────────────┐
│  FinBERT encoder (shared)                │
│  12-layer transformer                    │
│  pre-trained on financial text           │
│  → [CLS] token embedding (768-dim)       │
└──────────────────────────────────────────┘
     │
     ├──────────────────────┬──────────────────────┐
     ▼                      ▼                      ▼
┌──────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ Head 1       │   │ Head 2           │   │ Head 3             │
│ Sentiment    │   │ Controversy      │   │ Future movement    │
│ regression   │   │ binary classifier│   │ binary classifier  │
│ output: -1→1 │   │ output: 0/1      │   │ output: prob(move) │
└──────────────┘   └──────────────────┘   └────────────────────┘
```

**Head 1 — Sentiment regression**

Labels: generated automatically using TextBlob or VADER on the full article text as weak supervision. These are noisy but sufficient for regression pre-training. The FNSPID paper confirms this labeling approach. Output is a float from -1 (strongly negative) to +1 (strongly positive).

**Head 2 — Controversy classifier**

Labels: derived from the dataset itself. For each event (cluster of articles covering the same story, identified by embedding similarity), compute the standard deviation of sentiment scores across publishers. If std > 0.3, the story is labeled controversial (1), otherwise not (0). No external annotation needed. This is a genuinely novel labeling strategy.

**Head 3 — Future movement classifier**

Labels: derived by joining article publication dates with the market price dataset. If the ticker's closing price moved ≥ ±1.5% in the 3 days following publication, the article is labeled 1 (significant movement), otherwise 0. This produces a clean binary classification target with no manual work.

**Training:** Fine-tune all three heads jointly using a weighted multi-task loss:

```
total_loss = λ₁ · MSE(sentiment) + λ₂ · BCE(controversy) + λ₃ · BCE(movement)
```

The λ weights are tuned so no single head dominates. The shared encoder learns representations that are useful for all three tasks simultaneously.

**Output stored:** For each article: `sentiment_score`, `controversy_prob`, `movement_prob`.

---

### 5.4 Topic Model (BERTopic)

```
Article embeddings (from 5.2)
     │
     ▼
┌─────────────────────────────────────────┐
│  UMAP                                   │
│  dimensionality reduction 384 → 5 dims  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  HDBSCAN                                │
│  density-based clustering               │
│  → cluster assignments (topic IDs)      │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  c-TF-IDF (class-based TF-IDF)          │
│  finds most distinctive words           │
│  per cluster → human-readable labels    │
│  e.g. "earnings", "regulatory",         │
│  "product launch", "leadership"         │
└─────────────────────────────────────────┘
     │
     ▼
  per-article: topic_id + topic_probability_vector
  per-ticker: topic distribution (aggregated)
  → topic diversity = entropy(topic_probability_vector)
```

**Why BERTopic:** It requires no predefined number of topics and no manual labeling. The labels emerge from the data. It is state-of-the-art for short financial texts and produces interpretable, named topics that map naturally to real event types.

**Output stored:** `article_id → topic_id`, `topic_id → label_string`, `ticker → topic_distribution[]`.

---

### 5.5 Cointegration Graph

```
Market price data (OHLCV, per ticker)
     │
     ▼
┌─────────────────────────────────────────┐
│  For each pair (ticker_A, ticker_B):    │
│  Engle-Granger cointegration test       │
│  (statsmodels.tsa.stattools.coint)      │
│  → p-value + test statistic             │
└─────────────────────────────────────────┘
     │
     ▼
  keep pairs where p-value < 0.05
     │
     ▼
┌─────────────────────────────────────────┐
│  NetworkX graph                         │
│  nodes = tickers                        │
│  edges = cointegrated pairs             │
│  edge weight = test statistic           │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Louvain community detection            │
│  → cluster assignment per ticker        │
│  (used for node color in the UI)        │
└─────────────────────────────────────────┘
```

**Why cointegration and not correlation:** Correlation measures short-term co-movement and is unstable over time. Cointegration tests whether two time series share a long-run equilibrium — a much more robust structural relationship. It is the standard method used in pairs trading and portfolio construction research.

**Output stored:** Edge list `(ticker_A, ticker_B, weight)` and `ticker → community_id`.

---

### 5.6 SHAP Explainability Layer

```
Trained FinBERT model (from 5.3)
     │
     ▼
┌──────────────────────────────────────────────────┐
│  shap.Explainer (partition explainer)            │
│  designed for transformer models                 │
│  computes Shapley values for each input token    │
│  with respect to the sentiment head output       │
└──────────────────────────────────────────────────┘
     │
     ▼
  for each article:
  token_1: +0.23  ← pushed sentiment positive
  token_2: -0.41  ← pushed sentiment negative
  token_3: +0.07
  ...
     │
     ▼
  stored as {article_id → [(token, shap_value), ...]}
     │
     ▼
  served to frontend on article click
  rendered as horizontal waterfall bar chart (D3)
```

**What SHAP values mean:** SHAP (SHapley Additive exPlanations) decomposes a model's prediction into contributions from each input feature — in this case, each token. A token with a large positive SHAP value actively pushed the model toward a more positive sentiment score. This makes the model's reasoning fully transparent and auditable.

**Why this matters academically:** It directly addresses the black-box criticism of neural networks. The professor can see not just what score an article received, but exactly which words drove that score. This is the most methodologically rigorous component of the system.

---

### 5.7 Article Similarity Graph

```
Article embeddings (from 5.2, cached in FAISS)
     │
     ▼
  Query: retrieve all articles for selected ticker(s)
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Pairwise cosine similarity matrix               │
│  (efficient with FAISS inner product search)     │
└──────────────────────────────────────────────────┘
     │
     ├── Edge type: Semantic similarity
     │   keep pairs where cosine_sim > 0.6
     │   → sparse graph, natural event clusters
     │
     └── Edge type: Concept filter
         ┌────────────────────────────────────────┐
         │  Concept seed embeddings               │
         │  e.g. "earnings" =                     │
         │  avg_embed(["quarterly earnings",      │
         │  "revenue beat expectations",          │
         │  "EPS missed forecasts"])              │
         └────────────────────────────────────────┘
              │
              ▼
         cosine_sim(article_embed, concept_embed) > threshold
         → article tagged with concept
         → edge exists between articles sharing tag
```

**Output:** JSON `{nodes: [...], edges: [...]}` served per query. Built at query time from precomputed embeddings — fast because FAISS handles the similarity search efficiently.

---

## 6. Derived Metrics & Scatter Axes

| Metric | Type | Model | Description |
|---|---|---|---|
| Sentiment score | Regression output | FinBERT Head 1 | Tone of coverage, -1 to +1 |
| Controversy score | Classifier output | FinBERT Head 2 | Publisher sentiment divergence |
| Movement probability | Classifier output | FinBERT Head 3 | P(significant price move in 3 days) |
| Narrative intensity | Aggregation | None | Normalized article count per ticker per week |
| Topic diversity | BERTopic aggregation | BERTopic | Entropy of per-ticker topic distribution |
| Publisher bias | Aggregation | None | Delta between publisher avg sentiment and global avg |

All metrics are per-ticker and updated as the time window slider changes. The scatter plot redraws on axis selection — no re-inference, just a re-query of precomputed values.

---

## 7. Full Data Flow Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║                        DATA SOURCES                             ║
║  FNSPID (articles, summaries, tickers)  +  Market data (OHLCV) ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║                    OFFLINE PREPROCESSING                        ║
║                                                                  ║
║  ┌──────────────┐  ┌────────────────┐  ┌───────────────────┐   ║
║  │ spaCy NER    │  │ sentence-       │  │ Engle-Granger     │   ║
║  │ + ticker     │  │ transformers    │  │ cointegration     │   ║
║  │ resolution   │  │ embeddings      │  │ test (pairwise)   │   ║
║  └──────────────┘  └────────────────┘  └───────────────────┘   ║
║         │                  │                      │             ║
║         │          ┌───────┴──────┐               │             ║
║         │          │              │               │             ║
║         │    ┌──────────┐  ┌───────────┐          │             ║
║         │    │ BERTopic │  │ FinBERT   │          │             ║
║         │    │ topic    │  │ multi-task│          │             ║
║         │    │ model    │  │ + SHAP    │          │             ║
║         │    └──────────┘  └───────────┘          │             ║
║         │          │              │               │             ║
╚═════════╪══════════╪══════════════╪═══════════════╪════════════╝
          │          │              │               │
          ▼          ▼              ▼               ▼
╔══════════════════════════════════════════════════════════════════╗
║                         STORAGE                                 ║
║                                                                  ║
║  PostgreSQL              FAISS               NetworkX            ║
║  - article metadata      - article           - cointegration     ║
║  - sentiment scores        embeddings          graph             ║
║  - controversy scores    - similarity        - community IDs     ║
║  - movement probs          index                                 ║
║  - topic assignments                                             ║
║  - SHAP values                                                   ║
║  - co-mention counts                                             ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
                     FastAPI REST API
                     (query-time assembly)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
        ╔═══════════════════╗  ╔═══════════════════╗
        ║  Page 1           ║  ║  Page 2           ║
        ║  Observatory      ║  ║  Newsroom         ║
        ║                   ║  ║                   ║
        ║  Ticker graph     ║  ║  Article graph    ║
        ║  NLP scatter      ║  ║  SHAP waterfall   ║
        ║  Stats bar        ║  ║  Top articles     ║
        ╚═══════════════════╝  ╚═══════════════════╝
```

---

## 8. Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| NLP backbone | `ProsusAI/finbert` (HuggingFace) | Pre-trained transformer on financial text |
| NER | `spaCy en_core_web_trf` | Named entity extraction |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, high-quality 384-dim embeddings |
| Topic modeling | `BERTopic` | Unsupervised topic discovery with readable labels |
| Explainability | `shap` (partition explainer) | Token-level Shapley values for transformer models |
| Vector search | `FAISS` (Meta) | Efficient cosine similarity search over embeddings |
| Cointegration | `statsmodels.tsa.stattools.coint` | Engle-Granger test for price series |
| Graph analysis | `NetworkX` + `python-louvain` | Graph construction + community detection |
| Database | `PostgreSQL` | Structured storage for all precomputed outputs |
| Backend API | `FastAPI` | REST endpoints serving precomputed data to frontend |
| Frontend graph | `D3.js` (force-directed) | Interactive knowledge graph and article graph |
| Frontend charts | `D3.js` | Scatter plot, SHAP waterfall, sparklines |
| Market data | Existing dataset (OHLCV) | Price series for cointegration + movement labels |

---

## 9. Project Approach

### MVP scope

The MVP demonstrates the full pipeline on a subset of 20–30 tickers and their associated articles. All models are trained and all outputs precomputed. The frontend shows both pages with live interaction.

### Stages

**Stage 1 — Data preparation.** Load FNSPID, clean and deduplicate, join with market price data on ticker and date. Generate movement labels (±1.5% in 3 days). Generate weak sentiment labels (TextBlob/VADER). Split into train/validation/test.

**Stage 2 — Offline model training.** Train FinBERT multi-task model (Heads 1, 2, 3). Run BERTopic on article embeddings. Run sentence-transformer to generate FAISS index. Precompute SHAP values for all articles.

**Stage 3 — Graph construction.** Run spaCy NER over all articles and build co-mention index. Run Engle-Granger pairwise cointegration tests over selected tickers. Run Louvain community detection on the cointegration graph.

**Stage 4 — Backend API.** Implement FastAPI endpoints: ticker graph data, scatter metrics, article graph (semantic + concept), SHAP lookup, ranked article list, aggregate stats.

**Stage 5 — Frontend.** Build Page 1 (D3 force graph + scatter plot + stats bar) and Page 2 (D3 article graph + SHAP waterfall + stats panel). Wire all interactive controls.

### Team roles

| Role | Responsibilities |
|---|---|
| NLP / ML | FinBERT fine-tuning, SHAP, BERTopic, NER pipeline |
| Data engineering | FNSPID preprocessing, market data join, label generation, FAISS, PostgreSQL |
| Graph & backend | Cointegration tests, NetworkX, Louvain, FastAPI endpoints |
| Frontend | D3.js graphs, scatter plot, SHAP chart, interactive controls |

---

*NewsLens — turning news into structure, and structure into insight.*
