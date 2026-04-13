const BASE = "/api";

async function fetchJSON(path, opts) {
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const getTopics = () => fetchJSON("/topics");

export const getTimeline = (topicId) =>
  fetchJSON(`/timeline?topic_id=${topicId}`);

export const getHeadlines = (topicId, week) =>
  fetchJSON(`/timeline/headlines?topic_id=${topicId}&week=${week}`);

export const getGraph = () => fetchJSON("/graph");

export const postChat = (topicId, query) =>
  fetchJSON("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic_id: topicId, query }),
  });

export const getTicker = (symbol) =>
  fetchJSON(`/ticker?symbol=${symbol}`);

export const getTickers = () => fetchJSON("/tickers");
