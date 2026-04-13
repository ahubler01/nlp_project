"""Chat endpoint: retrieve + extractive summarise."""
import re
from datetime import date, timedelta
import polars as pl
import numpy as np

from config import TOPIC_COLS, TOPIC_LABELS, ID_TO_COL
import data_store as ds


def _parse_query(query: str) -> tuple[str, date, date]:
    """Extract topic keywords and date range from natural language query."""
    today = ds.merged["date_parsed"].max()  # use dataset end as "today"
    start = today - timedelta(weeks=4)
    end = today

    # try to find month + year
    month_match = re.search(
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s*(\d{4})?",
        query.lower(),
    )
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    if month_match:
        m = months[month_match.group(1)]
        y = int(month_match.group(2)) if month_match.group(2) else today.year
        start = date(y, m, 1)
        if m == 12:
            end = date(y + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(y, m + 1, 1) - timedelta(days=1)

    # "last N weeks/months"
    last_match = re.search(r"last\s+(\d+)\s+(week|month)", query.lower())
    if last_match:
        n = int(last_match.group(1))
        unit = last_match.group(2)
        if unit == "week":
            start = today - timedelta(weeks=n)
        else:
            start = today - timedelta(days=30 * n)
        end = today

    # strip stop words using word boundaries so "in" doesn't corrupt "industrial"
    topic_kw = re.sub(
        r"\b(summarise|summarize|summary|news|for|the|in|last|about|"
        r"\d+|weeks?|months?|january|february|march|april|may|june|"
        r"july|august|september|october|november|december)\b",
        "", query.lower(),
    )
    topic_kw = re.sub(r"\s+", " ", topic_kw).strip()

    return topic_kw, start, end


def _match_topic(keywords: str) -> tuple[int, str]:
    """Match keywords to the best fixed topic."""
    best_id, best_score = 0, 0
    kw_tokens = set(keywords.lower().split())
    for i, info in TOPIC_LABELS.items():
        label_tokens = set(info["label"].lower().replace("&", "and").split())
        desc_tokens = set(info["description"].lower().replace(",", "").split())
        overlap = len(kw_tokens & (label_tokens | desc_tokens))
        if overlap > best_score:
            best_score = overlap
            best_id = i
    return best_id, TOPIC_COLS[best_id]


def _summarise(texts: list[str], n_sentences: int = 5) -> str:
    """Run sumy LSA over concatenated text."""
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer
        combined = " ".join(t for t in texts if t)
        if not combined.strip():
            return "No summary content available."
        parser = PlaintextParser.from_string(combined, Tokenizer("english"))
        summarizer = LsaSummarizer()
        sentences = summarizer(parser.document, n_sentences)
        return " ".join(str(s) for s in sentences)
    except Exception as e:
        return f"Summarisation unavailable: {e}"


def chat(topic_id: int, query: str) -> dict:
    _, start, end = _parse_query(query)
    topic_col = TOPIC_COLS[topic_id]
    label = TOPIC_LABELS[topic_id]["label"]

    # filter by date + sort by topic probability
    df = (
        ds.merged
        .filter(
            (pl.col("date_parsed") >= start)
            & (pl.col("date_parsed") <= end)
        )
        .sort(topic_col, descending=True)
        .head(20)
    )

    if len(df) == 0:
        return {
            "topic": label,
            "date_range": [start.isoformat(), end.isoformat()],
            "summary": "No articles found for this topic and time window.",
            "sentiment": {"positive": 0, "neutral": 0, "negative": 0},
            "articles": [],
        }

    # extractive summary from Lsa_summary column
    summaries = df["Lsa_summary"].to_list()
    summary = _summarise(summaries)

    # sentiment aggregation using FinBERT probabilities
    pos = float(df["pos_prob"].mean()) if "pos_prob" in df.columns else 0.0
    neu = float(df["neu_prob"].mean()) if "neu_prob" in df.columns else 0.0
    neg = float(df["neg_prob"].mean()) if "neg_prob" in df.columns else 0.0

    # source articles
    articles = [
        {
            "title": r["Article_title"],
            "ticker": r["Stock_symbol"],
            "date": r["date_parsed"].isoformat() if r["date_parsed"] else None,
            "url": r["Url"],
            "sentiment": r["sentiment"],
        }
        for r in df.head(5).iter_rows(named=True)
    ]

    return {
        "topic": label,
        "date_range": [start.isoformat(), end.isoformat()],
        "summary": summary,
        "sentiment": {
            "positive": round(pos, 3),
            "neutral": round(neu, 3),
            "negative": round(neg, 3),
        },
        "articles": articles,
    }
