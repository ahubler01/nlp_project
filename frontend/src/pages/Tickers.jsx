import React, { useState, useEffect } from "react";
import { getTickers, getTicker } from "../api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Area, ComposedChart, Bar, Cell,
} from "recharts";

export default function Tickers() {
  const [tickers, setTickers] = useState([]);
  const [symbol, setSymbol] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getTickers()
      .then((list) => {
        setTickers(list);
        if (list.length > 0) setSymbol(list[0]);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!symbol) return;
    setLoading(true);
    setError(null);
    getTicker(symbol)
      .then((d) => {
        if (d.error) { setError(d.error); setData(null); }
        else setData(d);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [symbol]);

  if (error) return <div className="error-card">{error}</div>;

  return (
    <div>
      <h1 className="page-title">Ticker Browser</h1>
      <p className="page-subtitle">
        Explore price history, top topics, and recent articles for each ticker
      </p>

      <div style={{ marginBottom: 24 }}>
        <select
          className="select-field"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
        >
          {tickers.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </div>

      {loading && <p className="loading">Loading ticker data...</p>}

      {data && !loading && (
        <>
          {/* Price chart */}
          <div className="card">
            <div className="card-title">Daily Close Price — {data.symbol}</div>
            {data.prices.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={data.prices}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(v) => v.slice(5, 10)}
                    tick={{ fontSize: 11, fill: "#9ca3af" }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "#9ca3af" }}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#fff",
                      border: "1px solid #e5e7eb",
                      borderRadius: 8,
                      fontSize: 13,
                    }}
                    formatter={(v) => [`$${v.toFixed(2)}`, "Close"]}
                  />
                  <Line
                    type="monotone"
                    dataKey="close"
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="muted">No price data available.</p>
            )}
          </div>

          {/* Sentiment growth rate */}
          <div className="card">
            <div className="card-title">
              Sentiment Growth Rate — {data.symbol}
            </div>
            <p className="muted" style={{ marginBottom: 12 }}>
              Week-over-week change in net sentiment (positive − negative)
            </p>
            {data.sentiment_trend && data.sentiment_trend.length > 1 ? (
            <>
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart data={data.sentiment_trend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                  <XAxis
                    dataKey="week"
                    tickFormatter={(v) => v.slice(5, 10)}
                    tick={{ fontSize: 11, fill: "#9ca3af" }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "#9ca3af" }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#fff",
                      border: "1px solid #e5e7eb",
                      borderRadius: 8,
                      fontSize: 13,
                    }}
                    formatter={(v) => [
                      v != null ? (v > 0 ? "+" : "") + v.toFixed(4) : "—",
                      "Sentiment Change",
                    ]}
                    labelFormatter={(v) => `Week of ${v.slice(0, 10)}`}
                  />
                  <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="4 4" />
                  <Bar
                    dataKey="growth_rate"
                    name="growth_rate"
                    maxBarSize={16}
                    radius={[3, 3, 3, 3]}
                    opacity={0.7}
                  >
                    {data.sentiment_trend.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={
                          entry.growth_rate > 0
                            ? "#16a34a"
                            : entry.growth_rate < 0
                            ? "#dc2626"
                            : "#d1d5db"
                        }
                      />
                    ))}
                  </Bar>
                </ComposedChart>
              </ResponsiveContainer>
              <div
                style={{
                  display: "flex",
                  gap: 20,
                  justifyContent: "center",
                  marginTop: 10,
                  fontSize: 12,
                  color: "#6b7280",
                }}
              >
                <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 10, height: 10, background: "#16a34a", borderRadius: 2, display: "inline-block", opacity: 0.7 }} />
                  Sentiment improving
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 10, height: 10, background: "#dc2626", borderRadius: 2, display: "inline-block", opacity: 0.7 }} />
                  Sentiment declining
                </span>
              </div>
            </>
            ) : (
              <p className="muted">Not enough data for sentiment trend.</p>
            )}
          </div>

          {/* Top topics */}
          <div className="card">
            <div className="card-title">Top 3 Topics</div>
            {data.top_topics.length > 0 ? (
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {data.top_topics.map((t) => (
                  <span
                    key={t.topic_id}
                    className="badge"
                    style={{
                      background: "#eff6ff",
                      color: "#2563eb",
                      fontSize: 13,
                      padding: "6px 14px",
                    }}
                  >
                    {t.label}
                    <span className="muted" style={{ marginLeft: 6 }}>
                      {(t.mean_prob * 100).toFixed(1)}%
                    </span>
                  </span>
                ))}
              </div>
            ) : (
              <p className="muted">No topic data available for this ticker.</p>
            )}
          </div>

          {/* Recent articles */}
          <div className="card">
            <div className="card-title">Recent Articles</div>
            {data.recent_articles.length > 0 ? (
              <ul className="data-list">
                {data.recent_articles.map((a, i) => (
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
                      }}
                    >
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
            ) : (
              <p className="muted">No articles found for this ticker.</p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
