"""Simple Gaussian-fit phase detector for topic intensity curves."""
import numpy as np


def detect_phase(intensities: list[float]) -> dict:
    if len(intensities) < 4:
        return {"phase": "dormant", "label": "Dormant", "emoji": "Dormant",
                "pct_of_peak": 0.0, "momentum": 0.0}

    arr = np.array(intensities, dtype=float)
    peak = float(arr.max()) if arr.max() > 0 else 1.0
    current = float(arr[-1])
    pct_of_peak = current / peak

    # momentum = slope of last 4 points (normalised)
    recent = arr[-4:]
    x = np.arange(len(recent), dtype=float)
    slope = float(np.polyfit(x, recent, 1)[0])
    momentum = slope / peak if peak > 0 else 0.0

    if pct_of_peak < 0.2:
        phase, label, emoji = "dormant", "Dormant", "dormant"
    elif momentum > 0.02:
        phase, label, emoji = "rising", "Rising", "rising"
    elif pct_of_peak > 0.75:
        phase, label, emoji = "peak", "Peak", "peak"
    else:
        phase, label, emoji = "fading", "Fading", "fading"

    return {
        "phase": phase,
        "label": label,
        "emoji": emoji,
        "pct_of_peak": round(pct_of_peak, 3),
        "momentum": round(momentum, 4),
    }
