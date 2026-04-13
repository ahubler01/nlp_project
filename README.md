# FinLens — Financial News Intelligence

FinLens is a full-stack NLP application that analyzes financial news across a pool of 63 stocks. It combines topic modeling, FinBERT sentiment analysis, and stock price data to surface insights through an interactive web dashboard.

## Features

- **Topic Explorer** — Browse 13 financial topics (Big Tech, Semiconductors, AI/ML, Crypto, Macro & Fed Policy, etc.) and track their news intensity over time. Click any week to read the top headlines.
- **Topic Graph** — Visualizes cointegration relationships between topics based on weekly news intensity, revealing which themes move together.
- **Chat** — Query news by topic and date range using natural language (e.g. *"summarize biotech news for March 2024"*). Returns an extractive summary and sentiment breakdown.
- **Tickers** — Look up sentiment, topic distribution, and XGBoost-predicted price signals for any of the 63 tracked stocks.

## Tech Stack

| Layer | Tools |
|---|---|
| NLP pipeline | Polars, FinBERT, BERTopic, XGBoost, sumy (LSA) |
| Backend | FastAPI, uvicorn |
| Frontend | React + Vite, Recharts, react-force-graph-2d |
| Data | Parquet files (news, embeddings, predictions, stock prices) |

## Project Structure

```
nlp_project/
├── backend/          # FastAPI app + services (topics, timeline, graph, chat, tickers)
├── frontend/         # React + Vite dashboard
├── notebook/         # Jupyter notebooks for each NLP pipeline stage
│   ├── 01_dataset_downloader.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_NER.ipynb
│   ├── 04_Embeddings.ipynb
│   ├── 05_Topics.ipynb
│   ├── 06_Scoring.ipynb
│   └── 09_xgboost_tb.ipynb
├── data/             # Raw and processed data (stock news, prices, features)
└── models/           # Saved model artifacts
```

## Getting Started

### 1. Install dependencies

```bash
make install
```

This installs Python packages and Node modules.

### 2. Run the app

```bash
make dev
```

- Backend runs at `http://localhost:8000`
- Frontend runs at `http://localhost:5173`

To run them separately:

```bash
make backend   # FastAPI only
make frontend  # React only
```

## NLP Pipeline (Notebooks)

The notebooks walk through the full pipeline in order:

1. **Dataset Downloader** — fetch financial news articles
2. **EDA** — exploratory data analysis
3. **NER** — named entity recognition
4. **Embeddings** — sentence embeddings for articles
5. **Topics** — BERTopic topic modeling (13 topics)
6. **Scoring** — FinBERT sentiment scoring
7. **XGBoost** — price signal prediction from NLP features

## Stocks Covered

63 tickers including AAPL, MSFT, NVDA, TSLA, META, AMZN, JPM, XOM, MRNA, and more — spanning Tech, Semiconductors, Finance, Energy, Healthcare, Consumer, and ETFs.
