import { useQuery } from "@tanstack/react-query";
import {
  LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid,
} from "recharts";
import { api } from "../api";

export default function TickerDrawer({ ticker, onClose }) {
  const { data, isLoading } = useQuery({
    queryKey: ["price", ticker],
    queryFn: () => api.price(ticker, 24),
    enabled: !!ticker,
  });

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <span className="close" onClick={onClose}>×</span>
        <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.15em" }}>
          TICKER · LAST 24 WEEKS
        </div>
        <h2>{ticker}</h2>
        {isLoading && <div className="skeleton" style={{ height: 240, marginTop: 12 }} />}
        {!isLoading && data && data.length > 0 && (
          <div style={{ width: "100%", height: 260, marginTop: 12 }}>
            <ResponsiveContainer>
              <LineChart data={data} margin={{ top: 12, right: 8, left: -10, bottom: 0 }}>
                <CartesianGrid stroke="#1e293b" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#64748b" }}
                       interval={Math.max(0, Math.floor(data.length / 6))} stroke="#1e293b" />
                <YAxis tick={{ fontSize: 10, fill: "#64748b" }} stroke="#1e293b"
                       domain={["auto", "auto"]} />
                <Tooltip contentStyle={{
                  background: "#0b1324", border: "1px solid #1e293b",
                  fontSize: 11, fontFamily: "IBM Plex Mono",
                }} />
                <Line type="monotone" dataKey="close" stroke="#22d3ee"
                      dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        {!isLoading && (!data || data.length === 0) && (
          <div style={{ color: "#64748b", marginTop: 20 }}>no price data for this ticker</div>
        )}
      </div>
    </div>
  );
}
