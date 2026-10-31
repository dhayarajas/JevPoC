"""Plain-numpy metrics so the harness has no sklearn dependency."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: list, y_pred: list) -> float:
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else float("nan")


def macro_f1(y_true: list, y_pred: list) -> float:
    labels = sorted({y for y in y_true if y is not None})
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def reliability(conf: list[float], correct: list[bool], bins: int = 10) -> list[dict]:
    conf_a, corr_a = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf_a >= lo) & ((conf_a < hi) if hi < 1 else (conf_a <= hi))
        if m.any():
            out.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": int(m.sum()),
                    "conf": float(conf_a[m].mean()),
                    "acc": float(corr_a[m].mean()),
                }
            )
    return out


def ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    n = len(conf)
    return (
        float(sum(b["n"] / n * abs(b["acc"] - b["conf"]) for b in reliability(conf, correct, bins)))
        if n
        else float("nan")
    )


def auc(scores: list[float], labels: list[bool]) -> float:
    """Mann-Whitney AUC with tie handling."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def paired_bootstrap(a: list[bool], b: list[bool], n: int = 2000, seed: int = 0) -> dict[str, float]:
    """Accuracy difference a - b on the same cases, with a 95% interval and one-sided p (a <= b)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(a, float) - np.asarray(b, float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    diffs = x[idx].mean(axis=1)
    return {
        "diff": float(x.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float((diffs <= 0).mean()),
    }


def coverage_curve(conf: list[float], correct: list[bool], steps: int = 20) -> list[dict]:
    """Share of cases automated at each confidence threshold, and accuracy on that share."""
    out = []
    for t in np.linspace(0, 1, steps + 1):
        kept = [c for p, c in zip(conf, correct) if p >= t]
        out.append(
            {
                "threshold": float(t),
                "coverage": len(kept) / max(1, len(conf)),
                "accuracy": float(np.mean(kept)) if kept else 1.0,
            }
        )
    return out
