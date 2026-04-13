"""24-week topic intensity + top-5 tickers."""
import polars as pl
from config import TOPIC_COLS, ID_TO_COL, TOPIC_LABELS, WEEKS
import data_store as ds
from services.phase_detector import detect_phase


def weekly_intensity(topic_id: int) -> dict:
    col = ID_TO_COL[topic_id]
    label = TOPIC_LABELS[topic_id]["label"]

    df = ds.merged.select(["id", "date_parsed", "Stock_symbol", col]).drop_nulls()

    max_date = df["date_parsed"].max()
    min_date = max_date - pl.duration(weeks=WEEKS)
    df = df.filter(pl.col("date_parsed") >= min_date)

    # weekly mean intensity
    weekly = (
        df.with_columns(pl.col("date_parsed").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col(col).mean().alias("intensity"), pl.len().alias("count"))
        .sort("week")
    )

    weeks_list = [
        {"week": r["week"].isoformat(), "intensity": round(r["intensity"], 4), "count": r["count"]}
        for r in weekly.iter_rows(named=True)
    ]

    # phase detection
    intensities = weekly["intensity"].to_list()
    phase = detect_phase(intensities)

    # top-5 tickers
    top_tickers = _top_tickers(df, col)

    return {
        "topic_id": topic_id,
        "label": label,
        "weeks": weeks_list,
        "phase": phase,
        "top_tickers": top_tickers,
    }


def _top_tickers(df: pl.DataFrame, prob_col: str) -> list[dict]:
    ticker_stats = (
        df.filter(pl.col("Stock_symbol").is_not_null())
        .group_by("Stock_symbol")
        .agg(
            pl.len().alias("article_count"),
            pl.col(prob_col).mean().alias("mean_prob"),
        )
        .sort("article_count", descending=True)
        .head(5)
    )
    # join with xgb predictions for proba_up
    xgb_ticker_avg = (
        ds.xgb_preds
        .group_by("ticker")
        .agg(pl.col("proba_up").mean().alias("mean_proba_up"))
    )
    result = ticker_stats.join(
        xgb_ticker_avg,
        left_on="Stock_symbol",
        right_on="ticker",
        how="left",
    )
    return [
        {
            "ticker": r["Stock_symbol"],
            "article_count": r["article_count"],
            "mean_prob": round(r["mean_prob"], 4),
            "mean_proba_up": round(r["mean_proba_up"], 4) if r["mean_proba_up"] else None,
        }
        for r in result.iter_rows(named=True)
    ]


def headlines_for_week(topic_id: int, week: str) -> list[dict]:
    """Return up to 5 headlines for a given topic + ISO week string."""
    col = ID_TO_COL[topic_id]
    from datetime import date, timedelta
    week_date = date.fromisoformat(week)
    week_end = week_date + timedelta(days=7)

    df = (
        ds.merged
        .filter(
            (pl.col("date_parsed") >= week_date)
            & (pl.col("date_parsed") < week_end)
        )
        .sort(col, descending=True)
        .head(5)
        .select(["Article_title", "Stock_symbol", "date_parsed", "Url"])
    )
    return [
        {
            "title": r["Article_title"],
            "ticker": r["Stock_symbol"],
            "date": r["date_parsed"].isoformat() if r["date_parsed"] else None,
            "url": r["Url"],
        }
        for r in df.iter_rows(named=True)
    ]
