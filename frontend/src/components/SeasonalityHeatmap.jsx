export default function SeasonalityHeatmap({ data, color = "#a78bfa" }) {
  if (!data || data.length === 0) {
    return (
      <div style={{ color: "#64748b", fontSize: 12 }}>
        not enough corpus span for seasonality (need ≥ 2 calendar years)
      </div>
    );
  }
  const byWeek = new Map(data.map((d) => [d.week_of_year, d]));
  const cells = [];
  const vmax = Math.max(...data.map((d) => d.intensity)) || 1;
  for (let w = 1; w <= 52; w++) {
    const d = byWeek.get(w);
    const v = d ? d.intensity : 0;
    const t = v / vmax;
    cells.push(
      <div key={w} className="cell" title={`W${w}: ${v.toFixed(3)} (${d?.n_years || 0} yrs)`}
        style={{
          background: d
            ? `color-mix(in oklab, ${color} ${Math.round(15 + t * 85)}%, #020817)`
            : "#111a2e",
        }}
      />
    );
  }
  return (
    <div>
      <div className="heatmap" style={{ gridTemplateColumns: "repeat(52, 1fr)" }}>
        {cells}
      </div>
      <div style={{
        display: "flex", justifyContent: "space-between",
        fontSize: 10, color: "#64748b", marginTop: 4,
      }}>
        <span>W1 Jan</span><span>W13 Apr</span><span>W26 Jul</span><span>W39 Oct</span><span>W52 Dec</span>
      </div>
    </div>
  );
}
