"""Detection metrics. Convention: label 1 = LLM, higher score = more LLM-like."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Best TPR with FPR <= target, using thresholds picked on this same set (optimistic)."""
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_for_fpr(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Threshold t such that predicting `s > t` gives FPR <= target on (y, s)."""
    human = np.sort(np.asarray(s)[np.asarray(y) == 0])
    if len(human) == 0:
        return float("inf")
    allowed = int(np.floor(target * len(human)))  # humans allowed above t
    if allowed >= len(human):
        return float("-inf")
    return float(human[len(human) - allowed - 1])


def at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    y = np.asarray(y)
    pred = np.asarray(s) > thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    tpr = tp / pos if pos else float("nan")
    fpr = fp / neg if neg else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * tpr / (precision + tpr) if tp else 0.0
    return {"tpr": tpr, "fpr": fpr, "precision": precision, "f1": f1}


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """Precision if only `prevalence` of real-world reviews were LLM-written."""
    num = tpr * prevalence
    den = num + fpr * (1 - prevalence)
    return float(num / den) if den > 0 else float("nan")


def bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    s: np.ndarray,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI, resampling humans and LLMs separately so both classes stay present."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(fn(y[idx], s[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
