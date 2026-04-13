"""Lazy loaders for heavy ML artefacts. Call these only when a live path needs them."""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any, Optional

import joblib

from backend import config

log = logging.getLogger(__name__)

_miniLM = None
_miniLM_lock = threading.Lock()


def get_minilm():
    """SentenceTransformer used for (a) topic zero-shot, (b) cosine search."""
    global _miniLM
    if _miniLM is None:
        with _miniLM_lock:
            if _miniLM is None:
                from sentence_transformers import SentenceTransformer

                log.info("Loading MiniLM from %s …", config.MINILM_DIR)
                _miniLM = SentenceTransformer(str(config.MINILM_DIR), device="cpu")
    return _miniLM


@lru_cache(maxsize=1)
def get_ner_bundle() -> dict:
    log.info("Loading NER bundle …")
    return joblib.load(config.NER_BUNDLE)


@lru_cache(maxsize=1)
def get_bertopic():
    from bertopic import BERTopic
    log.info("Loading BERTopic (lazy) …")
    return BERTopic.load(str(config.BERTOPIC_DIR), embedding_model=get_minilm())


@lru_cache(maxsize=1)
def get_finbert() -> dict[str, Any]:
    """AutoModel for 768-d CLS embeddings + sentiment pipeline."""
    from transformers import AutoModel, AutoTokenizer, pipeline

    log.info("Loading FinBERT …")
    tokenizer = AutoTokenizer.from_pretrained(str(config.FINBERT_TOKENIZER))
    model = AutoModel.from_pretrained(str(config.FINBERT_MODEL)).eval()
    sentiment = pipeline(
        "text-classification",
        model=str(config.FINBERT_SENTIMENT),
        tokenizer=str(config.FINBERT_TOKENIZER),
        device=-1,
    )
    return {"tokenizer": tokenizer, "model": model, "sentiment": sentiment}


@lru_cache(maxsize=1)
def get_xgb_bundle() -> Optional[dict[str, Any]]:
    """Optional — only used if live XGBoost scoring is requested.
    Returns None if artefacts are missing (e.g. emb_pipeline.pkl not persisted)."""
    paths = [
        config.XGB_DIR / "xgb_tb.pkl",
        config.XGB_DIR / "scaler.pkl",
        config.XGB_DIR / "feat_cols.pkl",
        config.XGB_DIR / "threshold.pkl",
    ]
    if not all(p.exists() for p in paths):
        return None
    bundle = {
        "xgb": joblib.load(config.XGB_DIR / "xgb_tb.pkl"),
        "scaler": joblib.load(config.XGB_DIR / "scaler.pkl"),
        "feat_cols": joblib.load(config.XGB_DIR / "feat_cols.pkl"),
        "threshold": joblib.load(config.XGB_DIR / "threshold.pkl"),
    }
    if config.XGB_EMB_PIPELINE.exists():
        bundle["emb_pipeline"] = joblib.load(config.XGB_EMB_PIPELINE)
    return bundle


def models_loaded_status() -> dict[str, bool]:
    return {
        "miniLM": _miniLM is not None,
        "bertopic": get_bertopic.cache_info().currsize > 0,
        "ner": get_ner_bundle.cache_info().currsize > 0,
        "finbert": get_finbert.cache_info().currsize > 0,
        "xgb": get_xgb_bundle.cache_info().currsize > 0,
    }
