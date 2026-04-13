import { useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import LifecycleChart from "./components/LifecycleChart";
import PhaseBadge from "./components/PhaseBadge";
import StocksTable from "./components/StocksTable";
import TickerMatrix from "./components/TickerMatrix";
import SeasonalityHeatmap from "./components/SeasonalityHeatmap";
import ArticlePanel from "./components/ArticlePanel";
import TickerDrawer from "./components/TickerDrawer";
import TopicComposer from "./components/TopicComposer";

function Skeleton({ h = 180 }) {
  return <div className="skeleton" style={{ height: h }} />;
}

export default function App() {
  const qc = useQueryClient();
  const [activeTopic, setActiveTopic] = useState(null);
  const [drillWeek, setDrillWeek] = useState(null);
  const [tickerDrawer, setTickerDrawer] = useState(null);

  const healthQ = useQuery({ queryKey: ["health"], queryFn: api.health });
  const topicsQ = useQuery({ queryKey: ["topics"], queryFn: api.topics });
  const universeQ = useQuery({ queryKey: ["universe"], queryFn: api.universe });

  const topics = topicsQ.data?.topics || [];
  const topicLookup = useMemo(() => {
    const m = {};
    topics.forEach((t) => (m[t.id] = t));
    return m;
  }, [topics]);

  // pick a default topic once topics load
  if (!activeTopic && topics.length > 0) {
    setActiveTopic(topics[0].id);
  }

  const timelineQ = useQuery({
    queryKey: ["timeline", activeTopic],
    queryFn: () => api.timeline(activeTopic, 24),
    enabled: !!activeTopic,
  });
  const phaseQ = useQuery({
    queryKey: ["phase", activeTopic],
    queryFn: () => api.phase(activeTopic, 24),
    enabled: !!activeTopic,
  });
  const stocksQ = useQuery({
    queryKey: ["stocks", activeTopic],
    queryFn: () => api.stocks(activeTopic, 12),
    enabled: !!activeTopic,
  });
  const seasonalityQ = useQuery({
    queryKey: ["seasonality", activeTopic],
    queryFn: () => api.seasonality(activeTopic),
    enabled: !!activeTopic,
  });

  const matrixTickers = useMemo(() => {
    const u = universeQ.data?.tickers || [];
    return u.slice(0, 20);
  }, [universeQ.data]);

  const matrixQ = useQuery({
    queryKey: ["matrix", matrixTickers],
    queryFn: () => api.matrix(matrixTickers),
    enabled: matrixTickers.length > 0,
  });

  const deleteMut = useMutation({
    mutationFn: (id) => api.deleteTopic(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["topics"] }),
  });

  const activeTopicObj = topicLookup[activeTopic];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">FinLens</div>
        <div className="tag">FINANCIAL NEWS INTELLIGENCE · v1.0</div>

        <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase",
                      letterSpacing: "0.12em", marginTop: 18 }}>
          Corpus
        </div>
        {healthQ.data ? (
          <div className="corpus-meta" style={{ lineHeight: 1.7, marginTop: 4 }}>
            {healthQ.data.corpus.rows.toLocaleString()} articles<br />
            {healthQ.data.corpus.tickers} tickers<br />
            {healthQ.data.corpus.date_min} → {healthQ.data.corpus.date_max}
          </div>
        ) : <div style={{ color: "#64748b" }}>…</div>}

        <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase",
                      letterSpacing: "0.12em", marginTop: 18 }}>
          Topics
        </div>
        <div className="topic-list">
          {topicsQ.isLoading && <Skeleton h={200} />}
          {topics.map((t) => (
            <div
              key={t.id}
              className={`topic ${activeTopic === t.id ? "active" : ""}`}
              onClick={() => setActiveTopic(t.id)}
              style={{ borderLeftColor: activeTopic === t.id ? t.color : "transparent" }}
            >
              <span className="dot" style={{ background: t.color }} />
              <span>{t.label}</span>
              <span className="kind">{t.kind === "user" ? "●" : ""}</span>
              {t.kind === "user" && (
                <span
                  className="x"
                  onClick={(e) => { e.stopPropagation(); deleteMut.mutate(t.id); }}
                  title="remove user topic"
                >×</span>
              )}
            </div>
          ))}
        </div>

        <TopicComposer onCreated={(t) => setActiveTopic(t.id)} />

        <div className="terminal-banner">
          local · closed corpus · inference cached
        </div>
      </aside>

      <main className="main">
        <div className="header-row">
          <div>
            <h1 className="display" style={{ fontSize: 28 }}>
              {activeTopicObj?.label || "—"}
            </h1>
            <div className="corpus-meta">
              {activeTopicObj?.description || "pick a topic from the left"}
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span className="loader-dot" />
            <span style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase",
                           letterSpacing: "0.15em" }}>
              {healthQ.data?.status === "ok" ? "online" : "booting"}
            </span>
          </div>
        </div>

        <div className="grid cols-2">
          <div className="card">
            <h3>Lifecycle · last 24 weeks</h3>
            {timelineQ.isLoading ? <Skeleton h={260} /> : (
              <LifecycleChart
                data={timelineQ.data || []}
                color={activeTopicObj?.color || "#22d3ee"}
                onPick={(p) => setDrillWeek(p.iso_week)}
              />
            )}
          </div>
          <div className="card">
            <h3>Phase</h3>
            {phaseQ.isLoading ? <Skeleton h={140} /> : <PhaseBadge phase={phaseQ.data} />}
          </div>
        </div>

        <div className="grid cols-2">
          <div className="card">
            <h3>Top tickers · 24-week score</h3>
            {stocksQ.isLoading ? <Skeleton h={240} /> : (
              <StocksTable rows={stocksQ.data || []} onPick={setTickerDrawer} />
            )}
          </div>
          <div className="card">
            <h3>Seasonality · week-of-year</h3>
            {seasonalityQ.isLoading ? <Skeleton h={120} /> : (
              <SeasonalityHeatmap data={seasonalityQ.data || []}
                                  color={activeTopicObj?.color || "#a78bfa"} />
            )}
          </div>
        </div>

        <div className="card" style={{ marginTop: 16 }}>
          <h3>Topic × ticker matrix · {matrixTickers.length} tickers × 13 topics</h3>
          {matrixQ.isLoading || !matrixQ.data ? <Skeleton h={260} /> : (
            <TickerMatrix data={matrixQ.data} topicLookup={topicLookup} />
          )}
        </div>
      </main>

      {drillWeek && activeTopic && (
        <ArticlePanel
          topicId={activeTopic}
          topicLabel={activeTopicObj?.label || activeTopic}
          isoWeek={drillWeek}
          onClose={() => setDrillWeek(null)}
        />
      )}
      {tickerDrawer && (
        <TickerDrawer ticker={tickerDrawer} onClose={() => setTickerDrawer(null)} />
      )}
    </div>
  );
}
