import React, { useState, useEffect } from "react";
import { getTopics, getTimeline, getHeadlines } from "../api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function TopicExplorer() {
  const [topics, setTopics] = useState([]);
  const [topicId, setTopicId] = useState(0);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modal, setModal] = useState(null);

  useEffect(() => {
    getTopics().then(setTopics).catch(() => {});
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getTimeline(topicId)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [topicId]);

  const handleChartClick = async (payload) => {
    if (!payload?.activePayload?.[0]) return;
    const week = payload.activePayload[0].payload.week;
    try {
      const headlines = await getHeadlines(topicId, week);
      setModal({ week, headlines });
    } catch {}
  };

  if (error) return <div className="error-card">{error}</div>;

  return (
    <div>
      <h1 className="page-title">Topic Explorer</h1>
      <p className="page-subtitle">
        Explore weekly intensity trends across 13 financial news topics
      </p>

      <div style={{ marginBottom: 24 }}>
        <select
          className="select-field"
          value={topicId}
          onChange={(e) => setTopicId(Number(e.target.value))}
        >
          {topics.map((t) => (
            <option key={t.id} value={t.id}>
              {t.label}
            </option>
          ))}
        </select>
      </div>

      {loading && <p className="loading">Loading timeline...</p>}

      {data && !loading && (
        <>
          {/* Phase badge */}
          <div style={{ marginBottom: 20, display: "flex", alignItems: "center", gap: 12 }}>
            <span className={`badge badge-${data.phase.phase}`}>
              {data.phase.emoji} {data.phase.label}
            </span>
            <span className="muted">
              {Math.round(data.phase.pct_of_peak * 100)}% of peak
            </span>
          </div>

          {/* Intensity chart */}
          <div className="card">
            <div className="card-title">Weekly Intensity — {data.label}</div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={data.weeks}
                onClick={handleChartClick}
                style={{ cursor: "pointer" }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis
                  dataKey="week"
                  tickFormatter={(v) => v.slice(5, 10)}
                  tick={{ fontSize: 11, fill: "#9ca3af" }}
                />
                <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} />
                <Tooltip
                  contentStyle={{
                    background: "#fff",
                    border: "1px solid #e5e7eb",
                    borderRadius: 8,
                    fontSize: 13,
                  }}
                  formatter={(v) => [v.toFixed(4), "Intensity"]}
                  labelFormatter={(v) => `Week of ${v.slice(0, 10)}`}
                />
                <Line
                  type="monotone"
                  dataKey="intensity"
                  stroke="#2563eb"
                  strokeWidth={2}
                  dot={{ fill: "#2563eb", r: 3 }}
                  activeDot={{ r: 5, fill: "#1d4ed8" }}
                />
              </LineChart>
            </ResponsiveContainer>
            <p className="muted" style={{ marginTop: 8 }}>
              Click a week to see top headlines
            </p>
          </div>

          {/* Top 5 tickers */}
          <div className="card">
            <div className="card-title">Top 5 Tickers</div>
            <ul className="data-list">
              {data.top_tickers.map((t) => (
                <li key={t.ticker}>
                  <div>
                    <span className="ticker-tag">{t.ticker}</span>
                    <span className="muted" style={{ marginLeft: 8 }}>
                      {t.article_count} articles
                    </span>
                  </div>
                  <div>
                    {t.mean_proba_up != null && (
                      <span
                        className={`badge ${
                          t.mean_proba_up > 0.52
                            ? "sentiment-positive"
                            : t.mean_proba_up < 0.48
                            ? "sentiment-negative"
                            : "sentiment-neutral"
                        }`}
                      >
                        P(up) {(t.mean_proba_up * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}

      {/* Headlines modal */}
      {modal && (
        <div className="modal-overlay" onClick={() => setModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setModal(null)}>
              &times;
            </button>
            <h3>Headlines — Week of {modal.week.slice(0, 10)}</h3>
            {modal.headlines.length === 0 ? (
              <p className="muted">No headlines found for this week.</p>
            ) : (
              <ul className="data-list">
                {modal.headlines.map((h, i) => (
                  <li key={i} style={{ flexDirection: "column", alignItems: "flex-start", gap: 4 }}>
                    <div style={{ fontWeight: 500 }}>
                      {h.url ? (
                        <a href={h.url} target="_blank" rel="noreferrer">
                          {h.title}
                        </a>
                      ) : (
                        h.title
                      )}
                    </div>
                    <div className="muted">
                      {h.ticker && <span className="ticker-tag" style={{ marginRight: 8 }}>{h.ticker}</span>}
                      {h.date}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
