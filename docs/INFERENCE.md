# NewsLens — Model Inference Guide

End-to-end reference for loading and running every trained artifact in `models/` so a backend service can score a news article (or batch) through the full pipeline.

Paths below are relative to the repo root. `DEVICE` can be `'mps'`, `'cuda'`, or `'cpu'`.

---

## Pipeline overview

```
article (title + body)
  ├─► [1] NER  ─────────────► ticker                (models/ticker_ner_tfidf_2.joblib)
  ├─► [2] LSA summariser ───► lsa_summary           (sumy, no weights)
  │        │
  │        ├─► [3] BERTopic zero-shot ──► 13 topic probs   (models/bertopic_news + all-MiniLM-L6-v2)
  │        └─► [4] FinBERT ─────────────► sentiment + 768-d embedding (models/finbert_*)
  │
  └─► [5] XGBoost triple-barrier ──► P(up), signal (models/xgb_tb/)
           (features = PCA(FinBERT emb) + topic probs + market indicators + ticker one-hot)
```

The optional Flan-T5 step (topic-description augmentation) is a one-off used at training time; it is **not** needed at inference unless you want to regenerate the `TOPIC_DESCRIPTIONS` dictionary.

---

## 1. NER — ticker identification

**Artifact:** [models/ticker_ner_tfidf_2.joblib](../models/ticker_ner_tfidf_2.joblib)
A single joblib dict: `{"vectorizer": HashingVectorizer, "clf": SGDClassifier, "le": LabelEncoder}`.

**Input:** full article text (title + body concatenated with a space).
**Output:** predicted ticker (string) + confidence in `[0, 1]`.

```python
import joblib, numpy as np

bundle = joblib.load("models/ticker_ner_tfidf_2.joblib")
vectorizer, clf, le = bundle["vectorizer"], bundle["clf"], bundle["le"]

def predict_ticker(texts: list[str]):
    X = vectorizer.transform(texts)
    proba = clf.predict_proba(X)
    idx = np.argmax(proba, axis=1)
    return le.inverse_transform(idx), proba[np.arange(len(idx)), idx]

labels, confs = predict_ticker(["Google earnings beat", "Tesla recall"])
```

Labels come from the 63-ticker `POOL` defined in [notebook/03_NER.ipynb](../notebook/03_NER.ipynb). The model is char-ngram TF-IDF-style (hashing), trained with incremental `partial_fit`, so no GPU and no tokeniser is required. Runs on CPU.

---

## 2. LSA summariser

No trained weights. Pure `sumy`. Used to condense each article to ~3 sentences before feeding downstream models.

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

_lsa = LsaSummarizer()

def summarise(article: str, n: int = 3) -> str:
    parser = PlaintextParser.from_string(article or "", Tokenizer("english"))
    return " ".join(str(s) for s in _lsa(parser.document, n))
```

Requires: `sumy`, plus the NLTK `punkt` tokenizer (`python -m nltk.downloader punkt`).

---

## 3. BERTopic — zero-shot topic probabilities

**Artifacts:**
- [models/bertopic_news/](../models/bertopic_news/) — serialized BERTopic model (safetensors).
- [models/all-MiniLM-L6-v2/](../models/all-MiniLM-L6-v2/) — sentence embedder used by BERTopic and the zero-shot classifier.

At inference the 13-topic probability vector is produced **not** through `topic_model.transform`, but via the lightweight zero-shot cosine classifier used in [notebook/05_Topics.ipynb](../notebook/05_Topics.ipynb) (identical labels, always covers all 13 — no residuals). This is what the downstream XGBoost expects.

The exact documents fed in at training time are `"[<TICKER>] [<DATE>] <TITLE> | <LSA_SUMMARY>"`; mirror that prefix at inference for best alignment.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

TOPIC_LABELS = {
    0:  "Big Tech & Software",
    1:  "Semiconductors & Hardware",
    2:  "AI & Machine Learning",
    3:  "Earnings & Guidance",
    4:  "Macro & Fed Policy",
    5:  "Oil, Gas & Energy",
    6:  "Biotech & Pharma",
    7:  "Crypto & Fintech",
    8:  "EV & Clean Energy",
    9:  "M&A & IPO",
    10: "Dividends & Income",
    11: "Consumer & Retail",
    12: "Markets & Sentiment",
}

TOPIC_DESCRIPTIONS = {
    0:  "Big tech and enterprise software: Apple, Microsoft, Google, Meta, Amazon, cloud, SaaS, platforms, services",
    1:  "Semiconductors and hardware: Nvidia, AMD, Intel, TSMC, chips, GPUs, wafers, foundries, processors",
    2:  "Artificial intelligence and machine learning: ChatGPT, OpenAI, LLMs, generative AI, deep learning, inference, data centers",
    3:  "Earnings and guidance: quarterly results, EPS, revenue beat or miss, outlook, forecast, margins, estimates",
    4:  "Macro and Federal Reserve policy: inflation, interest rates, FOMC, CPI, Treasury yields, recession, GDP",
    5:  "Oil, gas and energy: crude oil, OPEC, Exxon, Chevron, refineries, natural gas, pipelines, petroleum",
    6:  "Biotech and pharmaceuticals: drug approval, FDA, clinical trials, vaccines, cancer therapy, pharma companies",
    7:  "Cryptocurrency and fintech: bitcoin, ethereum, blockchain, Coinbase, DeFi, PayPal, digital assets, stablecoins",
    8:  "Electric vehicles and clean energy: Tesla, Rivian, batteries, charging, solar, renewables, lithium, emissions",
    9:  "Mergers, acquisitions and IPOs: takeovers, buyouts, SPACs, public offerings, valuations, private equity deals",
    10: "Dividends and income investing: dividend yield, payout, buybacks, share repurchase, shareholder returns",
    11: "Consumer and retail: retail sales, Walmart, e-commerce, holiday shopping, consumer demand, prices",
    12: "Markets and sentiment: bull market, bear market, rally, selloff, volatility, VIX, S&P 500, Dow, indexes",
}

embed_model = SentenceTransformer("models/all-MiniLM-L6-v2", device=DEVICE)

class ZeroShotTopicClassifier:
    def __init__(self, embed_model, labels, descriptions, temperature=20.0):
        self.embed = embed_model
        self.ids = sorted(labels)
        self.labels = [labels[i] for i in self.ids]
        self.temperature = temperature
        self.cat_emb = embed_model.encode(
            [descriptions[i] for i in self.ids],
            normalize_embeddings=True,
        ).astype("float32")

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        q = self.embed.encode(texts, normalize_embeddings=True).astype("float32")
        logits = (q @ self.cat_emb.T) * self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

topic_clf = ZeroShotTopicClassifier(embed_model, TOPIC_LABELS, TOPIC_DESCRIPTIONS)

def build_topic_doc(ticker: str, date: str, title: str, lsa_summary: str) -> str:
    return f"[{ticker}] [{date}] {title} | {lsa_summary}"

probs = topic_clf.predict_proba([build_topic_doc("NVDA", "2025-02-10", title, summary)])
# probs.shape == (1, 13), ordered by sorted TOPIC_LABELS keys
```

**Downstream column naming (what XGBoost feature names expect):**

```python
prob_cols = [
    "prob_" + lbl.lower().replace("&", "and").replace(",", "").replace(" ", "_")
    for lbl in topic_clf.labels
]
# -> ['prob_big_tech_and_software', 'prob_semiconductors_and_hardware', ...]
```

**If you actually need the BERTopic assignment** (e.g. for analytics, not XGBoost features):

```python
from bertopic import BERTopic
topic_model = BERTopic.load("models/bertopic_news", embedding_model=embed_model)
topics, _ = topic_model.transform(docs, embeddings=embed_model.encode(docs))
```

### Optional — Flan-T5 topic description augmentation (training-time only)

Stored at `models/google-flan-t5-small/`. Used once to generate richer zero-shot topic descriptions. Backend does **not** need this at inference unless the topic set is being modified.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("models/google-flan-t5-small")
mdl = AutoModelForSeq2SeqLM.from_pretrained("models/google-flan-t5-small").to(DEVICE)

SYSTEM_PROMPT = """You are a helpful assistant for augmenting topic modeling results.
Given a topic name, generate a concise topic description...
Topic: "{topic}"
Description:"""

def describe(topic: str) -> str:
    prompt = SYSTEM_PROMPT.format(topic=topic)
    ids = tok(prompt, return_tensors="pt").input_ids.to(mdl.device)
    out = mdl.generate(ids, max_length=50, num_beams=5, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)
```

---

## 4. FinBERT — sentiment + 768-d embedding

**Artifacts:**
- [models/finbert_tokenizer/](../models/finbert_tokenizer/)
- [models/finbert_model/](../models/finbert_model/) — `AutoModel` for CLS-token embeddings.
- [models/finbert_sentiment/](../models/finbert_sentiment/) — HuggingFace `pipeline("text-classification")` bundle.

**Input:** LSA summary (the training pipeline used `Lsa_summary`, truncated to 512 tokens). Sentiment labels: `{positive, negative, neutral}`; `score` is the softmax confidence of the predicted label.

```python
import torch, numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline

tokenizer = AutoTokenizer.from_pretrained("models/finbert_tokenizer")
embed_model_fin = AutoModel.from_pretrained("models/finbert_model").to(DEVICE).eval()
sentiment_pipe = pipeline(
    "text-classification",
    model="models/finbert_sentiment",
    tokenizer="models/finbert_tokenizer",
    device=0 if DEVICE == "cuda" else -1,
)

@torch.no_grad()
def finbert_score(text: str):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=512, padding=True).to(DEVICE)
    out = embed_model_fin(**enc)
    emb = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()   # (768,)
    sent = sentiment_pipe(text[:512])[0]                            # {"label", "score"}
    return sent["label"], float(sent["score"]), emb

label, score, emb = finbert_score(lsa_summary)
# XGBoost uses a *signed* score: +score if positive, -score if negative, 0 if neutral
# (the training data in finbert_embeddings.parquet already encodes this convention —
#  check the sign convention if you change the sentiment model.)
```

Batch inference is straightforward — tokenise with `padding=True`, take `outputs.last_hidden_state[:, 0, :]`, run the pipeline on the list of texts.

---

## 5. XGBoost — triple-barrier direction signal

**Artifacts:** [models/xgb_tb/](../models/xgb_tb/)
- `xgb_tb.pkl` — trained `XGBClassifier`.
- `scaler.pkl` — `StandardScaler` fitted on the training feature matrix.
- `feat_cols.pkl` — ordered list of feature column names (use this to line up inputs).
- `threshold.pkl` — recommended decision threshold (median threshold from CV folds).
- `emb_pipeline.pkl` — fitted `Pipeline(StandardScaler → PCA(n_components=10))` applied to raw 768-d FinBERT embeddings before aggregation.

**Output:** `proba_up` (P(forward return clears +barrier)) and a binary `pred` at the saved threshold. Post-hoc a per-ticker rolling 60th percentile can be used to emit long/short/flat (see notebook 09 §Post-hoc).

### Feature vector (per `(ticker, date)`)

The vector is built from three sources and must match `feat_cols` exactly. Granular spec:

| Group | Cols | How to build |
|---|---|---|
| **FinBERT score aggregates** | `score`, `score_std`, `score_max`, `score_min`, `score_abs_max`, `score_pos_rate`, `n_news`, `score_roll3`, `n_news_roll3`, `score_lag1` | Group articles by `(ticker, date)`; apply mean/std/max/min/abs_max/positive-rate/count. Then per-ticker sorted by date add 3d rolling mean of `score`, 3d rolling sum of `n_news`, 1d lag of `score`. |
| **FinBERT embedding PCA** | `pc_0 … pc_9` | Apply the fitted `emb_pipeline.pkl` (StandardScaler → PCA) to the raw 768-d FinBERT embeddings, then per-`(ticker, date)` take the column means of the 10 PCs. The pipeline is fit on the training set and persisted — load it and call `.transform(emb_mat)` rather than refitting. |
| **Topic probabilities** | `prob_<label>` × 13 | From step 3; per-`(ticker, date)` use the mean across that day's articles. Column-name mapping shown above. |
| **Market indicators (per ticker)** | `mkt_ret_5`, `mkt_ret_21`, `mkt_vol_21`, `mkt_vol_63`, `mkt_mom`, `mkt_ma_ratio`, `mkt_hl`, `mkt_vol_z`, `mkt_rsi_14`, `mkt_ret_1`, `mkt_ret_norm_5`, `mkt_range_z`, `mkt_gap`, `mkt_xs_rank` | Rolling stats on OHLCV prices. Exact formulas in [notebook/09_xgboost_tb.ipynb](../notebook/09_xgboost_tb.ipynb) cell §Market indicators. All are past-only (no look-ahead). |
| **Ticker one-hot** | `tkr_<TICKER>` | `pd.get_dummies(ticker, prefix="tkr")`. The set of ticker columns is whatever was in training — read from `feat_cols`. Missing tickers → raise or return "unsupported". |

> ✅ **PCA persistence.** As of the latest training run, the embedding projection is shipped as `models/xgb_tb/emb_pipeline.pkl` — a `Pipeline(StandardScaler → PCA(n_components=10))` fit on the training corpus. Inference must load this pipeline and call `.transform(...)` on raw FinBERT embeddings instead of refitting PCA. Refit only when retraining the XGBoost model against a new corpus snapshot, and re-save the pipeline alongside it.

### Inference

```python
import joblib, pandas as pd

xgb = joblib.load("models/xgb_tb/xgb_tb.pkl")
scaler = joblib.load("models/xgb_tb/scaler.pkl")
feat_cols = joblib.load("models/xgb_tb/feat_cols.pkl")
threshold = joblib.load("models/xgb_tb/threshold.pkl")
emb_pipeline = joblib.load("models/xgb_tb/emb_pipeline.pkl")  # StandardScaler + PCA(10)

# Project a batch of raw 768-d FinBERT embeddings into the 10 PCs used at training:
#   emb_pcs = emb_pipeline.transform(finbert_emb_matrix)  # (N, 10)
# Then aggregate per (ticker, date) with .mean() into pc_0 … pc_9.

def score_row(feature_dict: dict) -> tuple[float, int]:
    row = pd.DataFrame([feature_dict]).reindex(columns=feat_cols).fillna(0.0)
    X = scaler.transform(row.values)
    proba_up = float(xgb.predict_proba(X)[0, 1])
    pred = int(proba_up >= threshold)
    return proba_up, pred
```

`feature_dict` must contain every name in `feat_cols`; unknown keys are dropped via `reindex`, missing keys fill `0.0` — validate strictness in the backend according to your SLA.

---

## Putting it together — request handler sketch

```python
def score_article(title: str, body: str, date: str) -> dict:
    text = f"{title or ''} {body or ''}".strip()

    # 1. NER
    ticker, ticker_conf = predict_ticker([text])
    ticker, ticker_conf = ticker[0], float(ticker_conf[0])

    # 2. LSA summary
    summary = summarise(body or title, n=3)

    # 3. Topic probabilities
    doc = build_topic_doc(ticker, date, title or "", summary)
    topic_probs = topic_clf.predict_proba([doc])[0]           # shape (13,)

    # 4. FinBERT
    sent_label, sent_score, finbert_emb = finbert_score(summary)
    signed_score = sent_score if sent_label == "positive" else \
                   -sent_score if sent_label == "negative" else 0.0

    # 5. Build the (ticker, date) feature row and run XGBoost
    features = build_feature_row(
        ticker=ticker,
        date=date,
        finbert_emb=finbert_emb,
        signed_score=signed_score,
        topic_probs=topic_probs,
        # market indicators fetched from the prices store for (ticker, date)
    )
    proba_up, pred = score_row(features)

    return {
        "ticker": ticker,
        "ticker_confidence": ticker_conf,
        "summary": summary,
        "sentiment": {"label": sent_label, "score": sent_score},
        "topic_probs": dict(zip(topic_clf.labels, topic_probs.tolist())),
        "proba_up": proba_up,
        "signal": pred,
    }
```

`build_feature_row` is the integration-owned function that joins the per-day article stream with the market-indicator store (see `data/Stock_price/stock_price.parquet` schema and notebook 09 §Market indicators for the exact polars expressions).

---

## Dependencies

```
numpy, pandas, polars
scikit-learn, xgboost, joblib
sentence-transformers, bertopic, umap-learn, hdbscan
transformers, torch
sumy, nltk           # + `python -m nltk.downloader punkt`
pyarrow              # parquet I/O
```

See [environment.yml](../environment.yml) for the exact pinned versions used at training time.

---

## Data-store expectations (for the backend)

- **`data/Stock_price/stock_price.parquet`** — daily OHLCV per ticker; required to compute market indicators at inference.
- **`data/predictions/finbert_embeddings.parquet`** — historical FinBERT outputs (768-d + sentiment). Used for PCA refits and backfills.
- **`data/predictions/full_df_topic_probabilities.parquet`** — historical 13-topic probabilities per article.
- **`data/predictions/xgb_tb_predictions.parquet`** — historical XGBoost predictions (reference for drift monitoring).

These are historical caches produced by the notebooks; the live service only needs the `models/` artifacts plus a price feed to reproduce the same features.
