import React, { useState, useEffect, useRef, useCallback } from "react";
import { getGraph } from "../api";

export default function TopicGraph() {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [ForceGraph, setForceGraph] = useState(null);
  const fgRef = useRef();

  useEffect(() => {
    import("react-force-graph-2d").then((mod) => {
      setForceGraph(() => mod.default);
    });
  }, []);

  useEffect(() => {
    getGraph()
      .then((g) => {
        // build lookup for links
        const data = {
          nodes: g.nodes.map((n) => ({ ...n, val: (n.size || 0.3) * 20 })),
          links: g.edges.map((e) => ({
            source: e.source,
            target: e.target,
            pvalue: e.pvalue,
            weight: e.weight,
          })),
        };
        setGraphData(data);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // color palette for nodes
  const COLORS = [
    "#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626",
    "#0891b2", "#4f46e5", "#16a34a", "#ea580c", "#be185d",
    "#0d9488", "#7c2d12", "#6366f1",
  ];

  const nodeColor = useCallback(
    (node) => {
      if (!graphData) return "#2563eb";
      const idx = graphData.nodes.findIndex((n) => n.id === node.id);
      return COLORS[idx % COLORS.length];
    },
    [graphData]
  );

  const paintNode = useCallback(
    (node, ctx, globalScale) => {
      const r = Math.sqrt(node.val || 5) * 1.5;
      const color = nodeColor(node);

      // circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      // label
      const fontSize = Math.max(10 / globalScale, 3);
      ctx.font = `600 ${fontSize}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#111827";
      ctx.fillText(node.label, node.x, node.y + r + fontSize + 1);
    },
    [nodeColor]
  );

  if (error) return <div className="error-card">{error}</div>;
  if (loading || !ForceGraph)
    return <p className="loading">Computing cointegration graph...</p>;
  if (!graphData)
    return <p className="loading">No graph data available.</p>;

  return (
    <div>
      <h1 className="page-title">Topic Graph</h1>
      <p className="page-subtitle">
        Cointegration network — edges where Engle-Granger p &lt; 0.05
      </p>

      <div
        className="card"
        style={{ padding: 0, overflow: "hidden", position: "relative" }}
      >
        <ForceGraph
          ref={fgRef}
          graphData={graphData}
          width={900}
          height={520}
          backgroundColor="#f9fafb"
          nodeCanvasObject={paintNode}
          nodePointerAreaPaint={(node, color, ctx) => {
            const r = Math.sqrt(node.val || 5) * 1.5 + 4;
            ctx.beginPath();
            ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
          }}
          linkWidth={(link) => Math.max(link.weight * 3, 0.5)}
          linkColor={() => "#d1d5db"}
          linkDirectionalParticles={0}
          onNodeHover={(node) => setHovered(node)}
          cooldownTicks={80}
          d3AlphaDecay={0.04}
          d3VelocityDecay={0.3}
        />
        {/* hover tooltip */}
        {hovered && (
          <div
            style={{
              position: "absolute",
              top: 12,
              right: 12,
              background: "#fff",
              border: "1px solid #e5e7eb",
              borderRadius: 10,
              padding: "12px 16px",
              fontSize: 13,
              boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
              maxWidth: 240,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 6 }}>
              {hovered.label}
            </div>
            {graphData.links
              .filter(
                (l) =>
                  (l.source.id || l.source) === hovered.id ||
                  (l.target.id || l.target) === hovered.id
              )
              .map((l, i) => {
                const other =
                  (l.source.id || l.source) === hovered.id
                    ? l.target.label || l.target.id || l.target
                    : l.source.label || l.source.id || l.source;
                return (
                  <div key={i} className="muted" style={{ marginBottom: 2 }}>
                    ↔ {other}{" "}
                    <span style={{ color: "#2563eb" }}>
                      p={l.pvalue.toFixed(3)}
                    </span>
                  </div>
                );
              })}
          </div>
        )}
      </div>

      <div className="muted" style={{ marginTop: 8 }}>
        {graphData.nodes.length} topics, {graphData.links.length} cointegrated
        pairs
      </div>
    </div>
  );
}
