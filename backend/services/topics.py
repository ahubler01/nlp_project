"""User-topic CRUD + MiniLM embedding. SQLite-backed so topics survive restarts."""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from typing import Optional

import numpy as np

from backend import config


_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(config.TOPICS_DB), check_same_thread=False)
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                description TEXT,
                embedding BLOB NOT NULL,
                color TEXT,
                created_at REAL
            )"""
        )
        _conn.commit()
    return _conn


def list_user_topics() -> list[dict]:
    cur = _db().execute(
        "SELECT id, label, description, color FROM topics ORDER BY created_at ASC"
    )
    return [
        {"id": r[0], "label": r[1], "description": r[2] or "", "color": r[3] or "#64748b",
         "kind": "user"}
        for r in cur.fetchall()
    ]


def get_user_topic(topic_id: str) -> Optional[dict]:
    cur = _db().execute(
        "SELECT id, label, description, embedding, color FROM topics WHERE id = ?",
        (topic_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    emb = np.frombuffer(row[3], dtype=np.float32).copy()
    return {
        "id": row[0], "label": row[1], "description": row[2] or "",
        "embedding": emb, "color": row[4] or "#64748b", "kind": "user",
    }


def create_user_topic(label: str, description: str = "") -> dict:
    from backend.loaders import get_minilm

    label = label.strip()
    if not label:
        raise ValueError("label must be non-empty")

    topic_id = "u_" + uuid.uuid4().hex[:10]
    text = description.strip() or label
    embed_model = get_minilm()
    emb = embed_model.encode([text], normalize_embeddings=True).astype("float32")[0]

    color = _palette(topic_id)
    with _lock:
        _db().execute(
            "INSERT INTO topics (id, label, description, embedding, color, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (topic_id, label, description, emb.tobytes(), color, time.time()),
        )
        _db().commit()

    return {"id": topic_id, "label": label, "description": description,
            "color": color, "kind": "user"}


def delete_user_topic(topic_id: str) -> bool:
    with _lock:
        cur = _db().execute("DELETE FROM topics WHERE id = ?", (topic_id,))
        _db().commit()
        return cur.rowcount > 0


def _palette(seed: str) -> str:
    # cycle through a muted dark-terminal palette
    colors = [
        "#22d3ee", "#f472b6", "#a78bfa", "#fbbf24", "#34d399",
        "#60a5fa", "#f87171", "#c084fc", "#4ade80", "#fb923c",
    ]
    return colors[hash(seed) % len(colors)]
