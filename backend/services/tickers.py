"""Ticker browser: price chart + top topics + recent articles."""
import polars as pl
from config import TOPIC_COLS, TOPIC_LABELS, WEEKS
import data_store as ds

# derive available tickers from the actual data
_AVAILABLE_TICKERS = sorted(
    ds.merged["Stock_symbol"]
    .drop_nulls()
    .unique()
    .to_list()
)


def get_ticker(symbol: str) -> dict:
    symbol = symbol.upper()
    if symbol not in _AVAILABLE_TICKERS:
        return {"error": f"{symbol} not in ticker pool"}

    prices = _price_chart(symbol)
    top_topics = _top_topics(symbol)
    articles = _recent_articles(symbol)
    sentiment_trend = _sentiment_trend(symbol)

    return {
        "symbol": symbol,
        "prices": prices,
        "top_topics": top_topics,
        "recent_articles": articles,
        "sentiment_trend": sentiment_trend,
    }


def _price_chart(symbol: str) -> list[dict]:
    df = (
        ds.pool_prices
        .filter(pl.col("ticker") == symbol)
        .sort("date", descending=True)
        .head(168)  # ~24 weeks of trading days
    )
    return [
        {"date": r["date"], "close": round(r["close"], 2)}
        for r in df.sort("date").iter_rows(named=True)
    ]


def _top_topics(symbol: str) -> list[dict]:
    ticker_articles = ds.merged.filter(pl.col("Stock_symbol") == symbol)
    if len(ticker_articles) == 0:
        return []

    scores = []
    for i, col in enumerate(TOPIC_COLS):
        mean_prob = ticker_articles[col].mean()
        if mean_prob is not None:
            scores.append((i, TOPIC_LABELS[i]["label"], float(mean_prob)))

    scores.sort(key=lambda x: x[2], reverse=True)
    return [
        {"topic_id": s[0], "label": s[1], "mean_prob": round(s[2], 4)}
        for s in scores[:3]
    ]


def _recent_articles(symbol: str) -> list[dict]:
    df = (
        ds.merged
        .filter(pl.col("Stock_symbol") == symbol)
        .sort("date_parsed", descending=True)
        .head(5)
    )
    return [
        {
            "title": r["Article_title"],
            "date": r["date_parsed"].isoformat() if r["date_parsed"] else None,
            "sentiment": r["sentiment"],
            "url": r["Url"],
        }
        for r in df.iter_rows(named=True)
    ]


def _sentiment_trend(symbol: str) -> list[dict]:
    """Weekly net sentiment (pos - neg) and its week-over-week change."""
    df = (
        ds.merged
        .filter(pl.col("Stock_symbol") == symbol)
        .select(["date_parsed", "pos_prob", "neg_prob"])
        .drop_nulls()
    )
    if len(df) == 0:
        return []

    weekly = (
        df.with_columns(pl.col("date_parsed").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(
            pl.col("pos_prob").mean().alias("pos"),
            pl.col("neg_prob").mean().alias("neg"),
            pl.len().alias("count"),
        )
        .sort("week")
        .with_columns(
            (pl.col("pos") - pl.col("neg")).alias("net_sentiment"),
        )
    )
    # growth rate = week-over-week change in net sentiment
    weekly = weekly.with_columns(
        (pl.col("net_sentiment") - pl.col("net_sentiment").shift(1))
        .alias("growth_rate"),
    )

    return [
        {
            "week": r["week"].isoformat(),
            "net_sentiment": round(float(r["net_sentiment"]), 4),
            "growth_rate": round(float(r["growth_rate"]), 4) if r["growth_rate"] is not None else None,
            "count": r["count"],
        }
        for r in weekly.iter_rows(named=True)
    ]


def list_tickers() -> list[str]:
    return _AVAILABLE_TICKERS
