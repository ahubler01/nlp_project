"""Topic listing (fixed 13 topics)."""
from config import TOPIC_LABELS


def list_topics() -> list[dict]:
    return [
        {"id": tid, "label": info["label"], "description": info["description"]}
        for tid, info in sorted(TOPIC_LABELS.items())
    ]
