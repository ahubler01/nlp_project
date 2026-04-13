const BASE = "/api";

async function request(path, opts = {}) {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${msg}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  health: () => request("/health"),
  topics: () => request("/topics"),
  createTopic: (label, description) =>
    request("/topics", {
      method: "POST",
      body: JSON.stringify({ label, description }),
    }),
  deleteTopic: (id) => request(`/topics/${id}`, { method: "DELETE" }),
  timeline: (topicId, weeks = 24) =>
    request(`/timeline?topic_id=${encodeURIComponent(topicId)}&weeks=${weeks}`),
  phase: (topicId, weeks = 24) =>
    request(`/phase?topic_id=${encodeURIComponent(topicId)}&weeks=${weeks}`),
  stocks: (topicId, top_n = 12) =>
    request(`/stocks?topic_id=${encodeURIComponent(topicId)}&top_n=${top_n}`),
  matrix: (tickers) =>
    request(`/matrix?tickers=${encodeURIComponent(tickers.join(","))}`),
  articles: (topicId, isoWeek, top_n = 5) =>
    request(
      `/articles?topic_id=${encodeURIComponent(topicId)}&iso_week=${encodeURIComponent(isoWeek)}&top_n=${top_n}`
    ),
  seasonality: (topicId) =>
    request(`/seasonality?topic_id=${encodeURIComponent(topicId)}`),
  price: (ticker, weeks = 24) =>
    request(`/price?ticker=${encodeURIComponent(ticker)}&weeks=${weeks}`),
  universe: () => request("/universe"),
};
