import { useMemo } from "react";

function shade(v, vmax) {
  if (!v || vmax <= 0) return "rgba(148,163,184,0.04)";
  const t = Math.min(1, v / vmax);
  const alpha = 0.15 + 0.85 * t;
  return `rgba(34, 211, 238, ${alpha.toFixed(3)})`;
}

export default function TickerMatrix({ data, topicLookup }) {
  const vmax = useMemo(() => {
    if (!data || !data.values.length) return 1;
    let m = 0;
    for (const row of data.values) for (const v of row) if (v > m) m = v;
    return m || 1;
  }, [data]);

  if (!data || !data.rows.length) {
    return <div style={{ color: "#64748b" }}>select tickers above</div>;
  }

  const cols = data.cols;
  const grid = `80px repeat(${cols.length}, 1fr)`;

  return (
    <div className="matrix" style={{ gridTemplateColumns: grid }}>
      <div />
      {cols.map((c) => (
        <div key={c} className="topic-label" style={{
          transform: "rotate(-35deg)", transformOrigin: "left bottom",
          whiteSpace: "nowrap", paddingLeft: 4,
        }} title={topicLookup[c]?.label || c}>
          {(topicLookup[c]?.label || c).slice(0, 12)}
        </div>
      ))}
      {data.rows.map((ticker, ri) => (
        <>
          <div key={`t-${ticker}`} className="ticker-label"
               style={{ display: "flex", alignItems: "center" }}>
            {ticker}
          </div>
          {data.values[ri].map((v, ci) => (
            <div key={`${ticker}-${cols[ci]}`}
                 className="cell"
                 title={`${ticker} · ${topicLookup[cols[ci]]?.label || cols[ci]} · ${v.toFixed(3)}`}
                 style={{ background: shade(v, vmax) }} />
          ))}
        </>
      ))}
    </div>
  );
}
