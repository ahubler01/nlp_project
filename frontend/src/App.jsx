import React, { useState } from "react";
import TopicExplorer from "./pages/TopicExplorer";
import Chat from "./pages/Chat";
import Tickers from "./pages/Tickers";
import "./App.css";

const VIEWS = [
  { key: "explorer", label: "Topic Explorer", icon: "1" },
  { key: "chat", label: "Chat", icon: "2" },
  { key: "tickers", label: "Tickers", icon: "3" },
];

export default function App() {
  const [view, setView] = useState("explorer");

  return (
    <>
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-icon">F</span>
          <span className="brand-text">FinLens</span>
        </div>
        <nav className="sidebar-nav">
          {VIEWS.map((v) => (
            <button
              key={v.key}
              className={`nav-item ${view === v.key ? "active" : ""}`}
              onClick={() => setView(v.key)}
            >
              <span className="nav-num">{v.icon}</span>
              {v.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">Financial News Intelligence</div>
      </aside>
      <main className="main-area">
        {view === "explorer" && <TopicExplorer />}
        {view === "chat" && <Chat />}
        {view === "tickers" && <Tickers />}
      </main>
    </>
  );
}
