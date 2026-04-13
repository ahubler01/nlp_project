export default function StocksTable({ rows, onPick }) {
  if (!rows || rows.length === 0) return <div style={{ color: "#64748b" }}>no data</div>;
  const max = Math.max(...rows.map((r) => r.score || 0)) || 1;
  return (
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Score</th>
          <th>Articles</th>
          <th>P(up)</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.ticker} className="click" onClick={() => onPick && onPick(r.ticker)}>
            <td style={{ fontWeight: 600 }}>{r.ticker}</td>
            <td>
              <span className="bar-cell"
                style={{ width: `${Math.round((r.score / max) * 100)}%` }} />
              <span style={{ marginLeft: 6, color: "#94a3b8" }}>{r.score.toFixed(3)}</span>
            </td>
            <td style={{ color: "#94a3b8" }}>{r.article_count}</td>
            <td style={{
              color: r.proba_up_mean == null ? "#64748b"
                : r.proba_up_mean >= 0.5 ? "#4ade80" : "#f87171",
            }}>
              {r.proba_up_mean == null ? "–" : r.proba_up_mean.toFixed(2)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
