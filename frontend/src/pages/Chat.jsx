import React, { useState, useEffect } from "react";
import { postChat, getTopics } from "../api";

export default function Chat() {
  const [topics, setTopics] = useState([]);
  const [topicId, setTopicId] = useState(0);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getTopics().then((list) => {
      setTopics(list);
      if (list.length > 0) setTopicId(list[0].id);
    }).catch(() => {});
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await postChat(topicId, query.trim());
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const sentimentBar = (label, value, color) => (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
      <span style={{ width: 70, fontSize: 13, color: "#6b7280", textTransform: "capitalize" }}>
        {label}
      </span>
      <div
        style={{
          flex: 1,
          height: 8,
          borderRadius: 4,
          background: "#f3f4f6",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${(value * 100).toFixed(0)}%`,
            height: "100%",
            borderRadius: 4,
            background: color,
            transition: "width 0.4s ease",
          }}
        />
      </div>
      <span style={{ width: 42, fontSize: 13, color: "#6b7280", textAlign: "right" }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );

  return (
    <div>
      <h1 className="page-title">Chat with the Corpus</h1>
      <p className="page-subtitle">
        Pick a topic, optionally specify a time window, and get extractive
        summaries from 140k+ financial news articles
      </p>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 24 }}>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <select
            className="select-field"
            value={topicId}
            onChange={(e) => setTopicId(Number(e.target.value))}
            style={{ minWidth: 220 }}
          >
            {topics.map((t) => (
              <option key={t.id} value={t.id}>{t.label}</option>
            ))}
          </select>
          <input
            className="input-field"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder='Time window, e.g. "April 2023" or "last 4 weeks" (optional)'
            style={{ flex: 1, minWidth: 200 }}
          />
          <button
            className="btn btn-primary"
            type="submit"
            disabled={loading}
            style={{ whiteSpace: "nowrap" }}
          >
            {loading ? "Searching..." : "Ask"}
          </button>
        </div>
      </form>

      {error && <div className="error-card">{error}</div>}

      {result && !loading && (
        <>
          <div style={{ marginBottom: 16, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <span className="badge badge-rising">{result.topic}</span>
            <span className="badge" style={{ background: "#f3f4f6", color: "#6b7280" }}>
              {result.date_range[0]} &rarr; {result.date_range[1]}
            </span>
          </div>

          {/* Summary */}
          <div className="card">
            <div className="card-title">Summary</div>
            <p style={{ fontSize: 14, lineHeight: 1.7, color: "#374151" }}>
              {result.summary}
            </p>
          </div>

          {/* Sentiment */}
          <div className="card">
            <div className="card-title">Sentiment Breakdown</div>
            {sentimentBar("positive", result.sentiment.positive, "#16a34a")}
            {sentimentBar("neutral", result.sentiment.neutral, "#9ca3af")}
            {sentimentBar("negative", result.sentiment.negative, "#dc2626")}
          </div>

          {/* Source articles */}
          <div className="card">
            <div className="card-title">Source Articles</div>
            {result.articles.length === 0 ? (
              <p className="muted">No articles found.</p>
            ) : (
              <ul className="data-list">
                {result.articles.map((a, i) => (
                  <li
                    key={i}
                    style={{
                      flexDirection: "column",
                      alignItems: "flex-start",
                      gap: 4,
                    }}
                  >
                    <div style={{ fontWeight: 500, fontSize: 14 }}>
                      {a.url ? (
                        <a href={a.url} target="_blank" rel="noreferrer">
                          {a.title}
                        </a>
                      ) : (
                        a.title
                      )}
                    </div>
                    <div
                      style={{
                        display: "flex",
                        gap: 8,
                        alignItems: "center",
                        flexWrap: "wrap",
                      }}
                    >
                      {a.ticker && (
                        <span className="ticker-tag">{a.ticker}</span>
                      )}
                      <span className="muted">{a.date}</span>
                      {a.sentiment && (
                        <span className={`badge sentiment-${a.sentiment}`}>
                          {a.sentiment}
                        </span>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </>
      )}

      {!result && !loading && !error && (
        <div
          style={{
            textAlign: "center",
            padding: "60px 0",
            color: "#d1d5db",
          }}
        >
          <div style={{ fontSize: 48, marginBottom: 12 }}>&#128269;</div>
          <p style={{ fontSize: 14, color: "#9ca3af" }}>
            Select a topic and click Ask to search the corpus
          </p>
        </div>
      )}
    </div>
  );
}
