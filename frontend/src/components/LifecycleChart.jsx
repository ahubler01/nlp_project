import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid,
} from "recharts";

const TooltipInner = ({ active, payload }) => {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div style={{
      background: "#0b1324", border: "1px solid #1e293b", padding: 10,
      fontSize: 11, fontFamily: "IBM Plex Mono, monospace",
    }}>
      <div style={{ color: "#94a3b8", marginBottom: 4 }}>{p.iso_week}</div>
      <div>Intensity: <b>{p.intensity.toFixed(3)}</b></div>
      <div>Articles: <b>{p.article_count}</b></div>
      <div style={{ color: "#64748b", marginTop: 4 }}>click to drill down</div>
    </div>
  );
};

export default function LifecycleChart({ data, color = "#22d3ee", onPick }) {
  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 12, right: 12, left: -10, bottom: 0 }}
          onClick={(e) => {
            if (e && e.activePayload && e.activePayload.length) {
              onPick && onPick(e.activePayload[0].payload);
            }
          }}
        >
          <defs>
            <linearGradient id="lcFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.55} />
              <stop offset="100%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="iso_week"
            tick={{ fontSize: 10, fill: "#64748b", fontFamily: "IBM Plex Mono" }}
            interval={Math.max(0, Math.floor(data.length / 8))}
            stroke="#1e293b"
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#64748b", fontFamily: "IBM Plex Mono" }}
            stroke="#1e293b"
          />
          <Tooltip content={<TooltipInner />} />
          <Area
            type="monotone" dataKey="intensity"
            stroke={color} strokeWidth={2} fill="url(#lcFill)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
