"""Phase / momentum / Gaussian-fit cycle detector for a weekly intensity curve."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def detect_phase(curve: list[float], current_week: int | None = None, window: int = 3) -> dict:
    c = np.asarray(curve, dtype=float)
    if c.size == 0:
        return {
            "phase": "dormant", "label": "Dormant", "emoji": "\U0001F311",
            "pct_of_peak": 0.0, "momentum": 0.0, "relative_magnitude": 0.0,
            "cycle": {"peak_week_estimate": None, "sigma": None, "cycle_progress_pct": None},
        }
    if current_week is None:
        current_week = len(c) - 1
    current_week = int(np.clip(current_week, 0, len(c) - 1))

    peak = float(c.max()) or 1e-9
    rel = float(c[current_week] / peak)

    lo = max(0, current_week - window)
    mid = max(0, current_week - 2 * window)
    momentum = (
        float(c[lo:current_week + 1].mean() - c[mid:lo].mean())
        if lo > mid else 0.0
    )

    if rel < 0.25:
        phase, label, emoji = "dormant", "Dormant", "\U0001F311"
    elif momentum > 0.05:
        phase, label, emoji = "rising", "Rising", "\U0001F4C8"
    elif momentum < -0.05 and rel > 0.4:
        phase, label, emoji = "fading", "Fading", "\U0001F4C9"
    else:
        phase, label, emoji = "peak", "Peak", "\U0001F525"

    cycle = {"peak_week_estimate": None, "sigma": None, "cycle_progress_pct": None}
    try:
        lo_peak = max(0, current_week - 12)
        peak_idx = int(np.argmax(c[lo_peak:current_week + 1])) + lo_peak
        lo_f, hi_f = max(0, peak_idx - 4), min(len(c), peak_idx + 5)
        x = np.arange(lo_f, hi_f)
        y = c[lo_f:hi_f]
        if len(x) >= 5:
            (a, mu, sigma), _ = curve_fit(_gauss, x, y, p0=[y.max(), peak_idx, 2.0], maxfev=2000)
            progress = float((current_week - mu) / max(abs(sigma), 1e-6))
            cycle = {
                "peak_week_estimate": float(mu),
                "sigma": float(abs(sigma)),
                "cycle_progress_pct": float(np.clip(progress, -3, 3)),
            }
    except Exception:
        pass

    return {
        "phase": phase, "label": label, "emoji": emoji,
        "pct_of_peak": rel, "momentum": momentum, "relative_magnitude": rel,
        "cycle": cycle,
    }
