import { useQuery } from "@tanstack/react-query";
import { api } from "../api";

function sentimentClass(s) {
  if (!s) return "neu";
  const t = s.toLowerCase();
  if (t.includes("pos")) return "pos";
  if (t.includes("neg")) return "neg";
  return "neu";
}

export default function ArticlePanel({ topicId, topicLabel, isoWeek, onClose }) {
  const { data, isLoading } = useQuery({
    queryKey: ["articles", topicId, isoWeek],
    queryFn: () => api.articles(topicId, isoWeek, 5),
    enabled: !!topicId && !!isoWeek,
  });

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <span className="close" onClick={onClose}>×</span>
        <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.15em" }}>
          DRILL-DOWN · {isoWeek}
        </div>
        <h2>{topicLabel}</h2>
        <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 8 }}>
          {isLoading ? "loading…" : `${data?.length || 0} top articles for this week`}
        </div>
        {isLoading && <div className="skeleton" style={{ height: 140 }} />}
        {!isLoading && data && data.map((a) => (
          <div className="article" key={a.id}>
            <div className="meta">
              <span className="chip">{a.ticker}</span>
              <span className={`chip ${sentimentClass(a.sentiment)}`}>
                {(a.sentiment || "neutral")}
                {a.sentiment_score != null && ` · ${a.sentiment_score.toFixed(2)}`}
              </span>
              <span className="chip">
                rel {a.relevance.toFixed(2)}
              </span>
              {a.proba_up != null && (
                <span className="chip"
                      style={{ color: a.proba_up >= 0.5 ? "#4ade80" : "#f87171",
                               borderColor: a.proba_up >= 0.5 ? "#166534" : "#7f1d1d" }}>
                  P(up) {a.proba_up.toFixed(2)}
                </span>
              )}
              {a.date && <span style={{ color: "#64748b" }}>{a.date}</span>}
            </div>
            <div className="title">
              {a.url
                ? <a href={a.url} target="_blank" rel="noreferrer">{a.title}</a>
                : a.title}
            </div>
            <div className="snippet">{a.snippet}…</div>
          </div>
        ))}
        {!isLoading && data && data.length === 0 && (
          <div style={{ color: "#64748b", marginTop: 20 }}>no articles for this week</div>
        )}
      </div>
    </div>
  );
}
