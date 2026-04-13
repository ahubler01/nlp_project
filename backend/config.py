"""Single source of truth for paths, topic labels, and the ticker pool."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Data ---------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
PREPROC_DIR = DATA_DIR / "preprocessed"
PRED_DIR = DATA_DIR / "predictions"
PRICE_DIR = DATA_DIR / "Stock_price"

SUBSET_NEWS = PREPROC_DIR / "subset_news.parquet"
ARTICLE_EMBEDDINGS_NPY = PREPROC_DIR / "article_embeddings.npy"
ARTICLE_EMBEDDING_IDS = PREPROC_DIR / "article_embedding_ids.parquet"
TOPIC_PROBS = PRED_DIR / "full_df_topic_probabilities.parquet"
FINBERT_EMBEDS = PRED_DIR / "finbert_embeddings.parquet"
XGB_PREDICTIONS = PRED_DIR / "xgb_tb_predictions.parquet"
LSA_SUMMARIES = PRED_DIR / "lsa_summaries.parquet"
STOCK_PRICE = PRICE_DIR / "stock_price.parquet"

# --- Models -------------------------------------------------------------
MODELS_DIR = REPO_ROOT / "models"
MINILM_DIR = MODELS_DIR / "all-MiniLM-L6-v2"
BERTOPIC_DIR = MODELS_DIR / "bertopic_news"
NER_BUNDLE = MODELS_DIR / "ticker_ner_tfidf_2.joblib"
FINBERT_MODEL = MODELS_DIR / "finbert_model"
FINBERT_TOKENIZER = MODELS_DIR / "finbert_tokenizer"
FINBERT_SENTIMENT = MODELS_DIR / "finbert_sentiment"
XGB_DIR = MODELS_DIR / "xgb_tb"
XGB_EMB_PIPELINE = XGB_DIR / "emb_pipeline.pkl"  # optional; brief notes it may be missing

MODEL_PATHS = {
    "miniLM": MINILM_DIR,
    "bertopic": BERTOPIC_DIR,
    "ner": NER_BUNDLE,
    "finbert": FINBERT_MODEL,
    "finbert_tokenizer": FINBERT_TOKENIZER,
    "finbert_sentiment": FINBERT_SENTIMENT,
    "xgb": XGB_DIR / "xgb_tb.pkl",
    "xgb_scaler": XGB_DIR / "scaler.pkl",
    "xgb_feat_cols": XGB_DIR / "feat_cols.pkl",
    "xgb_threshold": XGB_DIR / "threshold.pkl",
    "xgb_emb_pipeline": XGB_EMB_PIPELINE,
}

# --- Cache --------------------------------------------------------------
CACHE_DIR = REPO_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TIMELINE_GRID = CACHE_DIR / "fixed_timeline_grid.parquet"
CACHE_TOPIC_TICKER = CACHE_DIR / "topic_ticker_matrix.parquet"
CACHE_SEASONALITY = CACHE_DIR / "fixed_seasonality.parquet"
TOPICS_DB = CACHE_DIR / "topics.db"

# --- Topics -------------------------------------------------------------
TOPIC_LABELS = {
    0: "Big Tech & Software",
    1: "Semiconductors & Hardware",
    2: "AI & Machine Learning",
    3: "Earnings & Guidance",
    4: "Macro & Fed Policy",
    5: "Oil, Gas & Energy",
    6: "Biotech & Pharma",
    7: "Crypto & Fintech",
    8: "EV & Clean Energy",
    9: "M&A & IPO",
    10: "Dividends & Income",
    11: "Consumer & Retail",
    12: "Markets & Sentiment",
}

TOPIC_DESCRIPTIONS = {
    0: "Big tech and enterprise software: Apple, Microsoft, Google, Meta, Amazon, cloud, SaaS, platforms, services",
    1: "Semiconductors and hardware: Nvidia, AMD, Intel, TSMC, chips, GPUs, wafers, foundries, processors",
    2: "Artificial intelligence and machine learning: ChatGPT, OpenAI, LLMs, generative AI, deep learning, inference, data centers",
    3: "Earnings and guidance: quarterly results, EPS, revenue beat or miss, outlook, forecast, margins, estimates",
    4: "Macro and Federal Reserve policy: inflation, interest rates, FOMC, CPI, Treasury yields, recession, GDP",
    5: "Oil, gas and energy: crude oil, OPEC, Exxon, Chevron, refineries, natural gas, pipelines, petroleum",
    6: "Biotech and pharmaceuticals: drug approval, FDA, clinical trials, vaccines, cancer therapy, pharma companies",
    7: "Cryptocurrency and fintech: bitcoin, ethereum, blockchain, Coinbase, DeFi, PayPal, digital assets, stablecoins",
    8: "Electric vehicles and clean energy: Tesla, Rivian, batteries, charging, solar, renewables, lithium, emissions",
    9: "Mergers, acquisitions and IPOs: takeovers, buyouts, SPACs, public offerings, valuations, private equity deals",
    10: "Dividends and income investing: dividend yield, payout, buybacks, share repurchase, shareholder returns",
    11: "Consumer and retail: retail sales, Walmart, e-commerce, holiday shopping, consumer demand, prices",
    12: "Markets and sentiment: bull market, bear market, rally, selloff, volatility, VIX, S&P 500, Dow, indexes",
}

# 13 colors (dark-terminal palette)
TOPIC_COLORS = [
    "#22d3ee", "#f472b6", "#a78bfa", "#fbbf24", "#34d399",
    "#60a5fa", "#f87171", "#c084fc", "#4ade80", "#fb923c",
    "#fde047", "#e879f9", "#94a3b8",
]


def topic_id_from_label(label: str) -> str:
    """Same convention as the `prob_<id>` column names in full_df_topic_probabilities.parquet."""
    return (
        label.lower()
        .replace("&", "and")
        .replace(",", "")
        .replace(" ", "_")
    )


FIXED_TOPICS = [
    {
        "id": topic_id_from_label(TOPIC_LABELS[i]),
        "label": TOPIC_LABELS[i],
        "description": TOPIC_DESCRIPTIONS[i],
        "kind": "fixed",
        "color": TOPIC_COLORS[i],
        "prob_col": "prob_" + topic_id_from_label(TOPIC_LABELS[i]),
        "numeric_id": i,
    }
    for i in sorted(TOPIC_LABELS)
]
FIXED_TOPIC_BY_ID = {t["id"]: t for t in FIXED_TOPICS}
FIXED_PROB_COLS = [t["prob_col"] for t in FIXED_TOPICS]

# --- App ----------------------------------------------------------------
DEFAULT_WEEKS = 24
SEASONALITY_MIN_SPAN_WEEKS = 104  # need ≥2 calendar years
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",  # Vite dev default
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
