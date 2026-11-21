"""Shared experiment definitions and evaluation, used by every model."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .normalize import bin_labels, strip_surface, strip_surface_nospace

TEXT_VIEWS = {
    "none": lambda t: t,
    "surface": strip_surface,
    "nospace": strip_surface_nospace,
}


def load_dataset(cfg) -> pd.DataFrame:
    return pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")


def _balance_train(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample the larger class so hold-out runs train on balanced data."""
    n = int(train["label"].value_counts().min())
    parts = [g.sample(n=n, random_state=seed) for _, g in train.groupby("label")]
    return pd.concat(parts).sort_index()


def experiment_frames(df: pd.DataFrame, exp: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Return (train, val, test, text_view) for an experiment name.

    main                 : all generators, all prompts
    ablation_surface     : same split, punctuation/emoji/jamo removed
    ablation_nospace     : same, spacing removed as well
    logo_<family>        : train/val without that generator family; test = human test + ALL rows of it
    lopo_<prompt>        : train/val without that prompt type;      test = human test + ALL rows of it
    """
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    if exp == "main":
        return tr, va, te, "none"
    if exp.startswith("ablation_"):
        view = exp.removeprefix("ablation_")
        if view not in TEXT_VIEWS:
            raise ValueError(exp)
        return tr, va, te, view
    for prefix, col in (("logo_", "family"), ("lopo_", "prompt_type")):
        if exp.startswith(prefix):
            held = exp.removeprefix(prefix)
            if held not in set(df.loc[df["label"] == 1, col]):
                raise ValueError(f"{col}={held!r} not in dataset")
            keep = lambda d: d[(d["label"] == 0) | (d[col] != held)]  # noqa: E731
            test = pd.concat([te[te["label"] == 0], df[(df["label"] == 1) & (df[col] == held)]])
            return _balance_train(keep(tr), seed), keep(va), test, "none"
    raise ValueError(f"unknown experiment {exp!r}")


def default_experiments(df: pd.DataFrame) -> list[str]:
    llm = df[df["label"] == 1]
    exps = ["main", "ablation_surface", "ablation_nospace"]
    exps += [f"logo_{f}" for f in sorted(llm["family"].unique())]
    if llm["prompt_type"].nunique() > 1:
        exps += [f"lopo_{p}" for p in sorted(llm["prompt_type"].unique())]
    return exps


def _clean(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _subset_metrics(y: np.ndarray, s: np.ndarray, thresholds: dict, cfg) -> dict:
    n_h, n_l = int((y == 0).sum()), int((y == 1).sum())
    out: dict = {"n_human": n_h, "n_llm": n_l}
    if n_l and n_h:
        out["auc"] = M.auc(y, s)
        out["ap"] = float(M.average_precision_score(y, s))
        for t in cfg["eval"]["fpr_targets"]:
            out[f"oracle_tpr@fpr{t:g}"] = M.tpr_at_fpr(y, s, t)
    for name, thr in thresholds.items():
        r = M.at_threshold(y, s, thr)
        if n_l and n_h:
            r["precision_at_prevalence"] = {
                f"{p:g}": M.precision_at_prevalence(r["tpr"], r["fpr"], p) for p in cfg["eval"]["prevalences"]
            }
        out[name] = {k: (_clean(v) if not isinstance(v, dict) else {kk: _clean(vv) for kk, vv in v.items()}) for k, v in r.items()}
    return out


def evaluate(val: pd.DataFrame, test: pd.DataFrame, cfg, seed: int = 0) -> dict:
    """Thresholds come from validation (as in deployment); metrics on test subsets."""
    e = cfg["eval"]
    thresholds = {f"val_thr_fpr{t:g}": M.threshold_for_fpr(val["label"].values, val["score"].values, t) for t in e["fpr_targets"]}
    res: dict = {
        "val_auc": M.auc(val["label"].values, val["score"].values) if val["label"].nunique() == 2 else None,
        "thresholds": thresholds,
        "subsets": {},
    }
    human = test[test["label"] == 0]
    subsets: dict[str, pd.DataFrame] = {"overall": test}
    llm = test[test["label"] == 1]
    for g in sorted(llm["source"].unique()):
        subsets[f"generator={g}"] = pd.concat([human, llm[llm["source"] == g]])
    for pt in sorted(llm["prompt_type"].unique()):
        subsets[f"prompt={pt}"] = pd.concat([human, llm[llm["prompt_type"] == pt]])
    for b in bin_labels(cfg["human"]["length_edges"]):
        sub = test[test["length_bin"] == b]
        if len(sub):
            subsets[f"length={b}"] = sub
    subsets[f"length>={e['gate_min_chars']}"] = test[test["length"] >= e["gate_min_chars"]]

    for name, sub in subsets.items():
        res["subsets"][name] = _subset_metrics(sub["label"].values, sub["score"].values, thresholds, cfg)

    y, s = test["label"].values, test["score"].values
    if e.get("n_bootstrap") and len(np.unique(y)) == 2:
        res["subsets"]["overall"]["auc_ci95"] = M.bootstrap_ci(M.auc, y, s, e["n_bootstrap"], seed)
        res["subsets"]["overall"]["oracle_tpr@fpr0.05_ci95"] = M.bootstrap_ci(
            lambda yy, ss: M.tpr_at_fpr(yy, ss, 0.05), y, s, e["n_bootstrap"], seed
        )
    return res


def save_run(out_dir: Path, test: pd.DataFrame, result: dict, extra: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["id", "label", "score", "source", "family", "prompt_type", "length", "length_bin", "split"]
    test[cols].to_parquet(out_dir / "predictions.parquet", index=False)
    payload = {**(extra or {}), **result}
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def fmt(x, digits: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def summary_line(model: str, exp: str, res: dict) -> str:
    o = res["subsets"]["overall"]
    thr = o.get("val_thr_fpr0.05", {})
    return (
        f"[{model}/{exp}] AUC {fmt(o.get('auc'), 4)} | "
        f"TPR@FPR5%(val thr) {fmt(thr.get('tpr'))} (실제 FPR {fmt(thr.get('fpr'))})"
    )
