"""Verify every local model / parquet artefact the backend depends on."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import config  # noqa: E402


def main() -> int:
    required = {
        "MiniLM dir": config.MINILM_DIR,
        "BERTopic dir": config.BERTOPIC_DIR,
        "NER bundle": config.NER_BUNDLE,
        "FinBERT model": config.FINBERT_MODEL,
        "FinBERT tokenizer": config.FINBERT_TOKENIZER,
        "FinBERT sentiment": config.FINBERT_SENTIMENT,
        "XGB model": config.XGB_DIR / "xgb_tb.pkl",
        "XGB scaler": config.XGB_DIR / "scaler.pkl",
        "XGB feat cols": config.XGB_DIR / "feat_cols.pkl",
        "XGB threshold": config.XGB_DIR / "threshold.pkl",
        "subset_news parquet": config.SUBSET_NEWS,
        "article embeddings npy": config.ARTICLE_EMBEDDINGS_NPY,
        "article embedding ids": config.ARTICLE_EMBEDDING_IDS,
        "topic probs parquet": config.TOPIC_PROBS,
        "finbert embeds parquet": config.FINBERT_EMBEDS,
        "xgb predictions parquet": config.XGB_PREDICTIONS,
        "stock price parquet": config.STOCK_PRICE,
    }
    optional = {
        "XGB emb pipeline (only needed for live XGB inference)": config.XGB_EMB_PIPELINE,
    }

    missing: list[str] = []
    print("=== Required ===")
    for name, path in required.items():
        ok = path.exists()
        marker = "OK" if ok else "MISSING"
        print(f"  [{marker}] {name}: {path}")
        if not ok:
            missing.append(f"{name} -> {path}")

    print("=== Optional ===")
    for name, path in optional.items():
        marker = "OK" if path.exists() else "WARN"
        print(f"  [{marker}] {name}: {path}")

    if missing:
        print(f"\n{len(missing)} required artefact(s) missing.")
        return 1
    print("\nAll required artefacts present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
