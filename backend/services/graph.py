"""Precompute topic cointegration graph at startup."""
import json
import numpy as np
import polars as pl
from pathlib import Path
from statsmodels.tsa.stattools import coint

from config import TOPIC_COLS, TOPIC_LABELS, WEEKS, CACHE_DIR, ID_TO_COL
import data_store as ds

GRAPH_CACHE = CACHE_DIR / "topic_graph.json"


def build_weekly_intensity() -> dict[str, np.ndarray]:
    """Build a {topic_label: 24-week intensity array} dict."""
    df = ds.merged.select(["date_parsed"] + TOPIC_COLS).drop_nulls(subset=["date_parsed"])
    max_date = df["date_parsed"].max()
    min_date = max_date - pl.duration(weeks=WEEKS)
    df = df.filter(pl.col("date_parsed") >= min_date)

    weekly = (
        df.with_columns(pl.col("date_parsed").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg([pl.col(c).mean() for c in TOPIC_COLS])
        .sort("week")
    )

    result = {}
    for i, col in enumerate(TOPIC_COLS):
        label = TOPIC_LABELS[i]["label"]
        result[label] = weekly[col].to_numpy().astype(float)
    return result


def build_topic_graph(weekly_intensity: dict[str, np.ndarray]) -> dict:
    topics = list(weekly_intensity)
    edges = []
    for i, a in enumerate(topics):
        for b in topics[i + 1:]:
            series_a = weekly_intensity[a]
            series_b = weekly_intensity[b]
            if len(series_a) < 5 or len(series_b) < 5:
                continue
            try:
                _, pvalue, _ = coint(series_a, series_b)
                if pvalue < 0.05:
                    edges.append({
                        "source": a,
                        "target": b,
                        "pvalue": round(float(pvalue), 4),
                        "weight": round(float(1.0 - pvalue), 4),
                    })
            except Exception:
                continue

    # total intensity per topic for node sizing
    totals = {t: float(np.sum(v)) for t, v in weekly_intensity.items()}
    max_total = max(totals.values()) if totals else 1.0

    nodes = [
        {"id": t, "label": t, "size": round(totals.get(t, 0) / max_total, 3)}
        for t in topics
    ]
    return {"nodes": nodes, "edges": edges}


def get_graph() -> dict:
    """Return cached graph or compute + cache."""
    if GRAPH_CACHE.exists():
        with open(GRAPH_CACHE) as f:
            return json.load(f)

    print("[graph] computing cointegration graph …")
    wi = build_weekly_intensity()
    graph = build_topic_graph(wi)
    with open(GRAPH_CACHE, "w") as f:
        json.dump(graph, f)
    print(f"[graph] done — {len(graph['edges'])} edges")
    return graph
