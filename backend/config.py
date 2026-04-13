"""Paths, constants, and topic labels."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

# ── data paths ──────────────────────────────────────────────────
DATA = ROOT / "data"
NLP  = DATA / "nlp_data"

SUBSET_NEWS_PATH       = NLP / "preprocessed" / "subset_news.parquet"
ARTICLE_EMB_PATH       = NLP / "preprocessed" / "article_embeddings.npy"
ARTICLE_EMB_IDS_PATH   = NLP / "preprocessed" / "article_embedding_ids.parquet"
TOPIC_PROBS_PATH       = NLP / "predictions"  / "full_df_topic_probabilities.parquet"
FINBERT_PATH           = NLP / "predictions"  / "finbert_embeddings.parquet"
XGB_PREDS_PATH         = NLP / "predictions"  / "xgb_tb_predictions.parquet"
STOCK_PRICE_PATH       = DATA / "Stock_price" / "stock_price.parquet"
ARTICLE_SENTIMENTS_PATH = DATA / "features" / "article_sentiments.parquet"
TOPIC_DESC_PATH        = NLP / "topic_descriptions.json"

CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── 63-ticker pool ──────────────────────────────────────────────
POOL = [
    "AAPL","MSFT","GOOG","GOOGL","AMZN","TSLA","META","NVDA","AMD","INTC",
    "CRM","NFLX","ADBE","PYPL","UBER","SQ","SHOP","ZM","SNAP","COIN",
    "PLTR","ORCL","QQQ","SPY","DIA","IWM","T","VZ","JPM","GS","MS",
    "WFC","BAC","C","XOM","CVX","JNJ","PFE","MRNA","GILD","MRK","UNH",
    "ABT","WMT","COST","TGT","HD","KO","PEP","SBUX","MCD","BA","GE",
    "CAT","MMM","DIS","CMCSA","V","MA","MU","QCOM","TXN","AVGO","F","GM",
]

# ── 13 fixed topics ─────────────────────────────────────────────
TOPIC_COLS = [
    "prob_big_tech_and_software",
    "prob_semiconductors_and_hardware",
    "prob_ai_and_machine_learning",
    "prob_earnings_and_guidance",
    "prob_macro_and_fed_policy",
    "prob_oil_gas_and_energy",
    "prob_biotech_and_pharma",
    "prob_crypto_and_fintech",
    "prob_ev_and_clean_energy",
    "prob_manda_and_ipo",
    "prob_dividends_and_income",
    "prob_consumer_and_retail",
    "prob_markets_and_sentiment",
]

def _load_topic_labels() -> dict[int, dict]:
    with open(TOPIC_DESC_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

TOPIC_LABELS = _load_topic_labels()

# Map column name → topic id and vice-versa
COL_TO_ID = {col: i for i, col in enumerate(TOPIC_COLS)}
ID_TO_COL = {i: col for i, col in enumerate(TOPIC_COLS)}

WEEKS = 24
