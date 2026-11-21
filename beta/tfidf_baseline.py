"""Day 2 morning: TF-IDF (char n-gram) + Logistic Regression.

Runs every experiment by default: main, both ablations, generator hold-out
(logo_*) and prompt hold-out (lopo_*). Also dumps the top-weighted n-grams so
you can check by eye whether the model is reading punctuation or spacing.

Usage:
    python -m beta.tfidf_baseline
    python -m beta.tfidf_baseline --exp main,ablation_surface
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import metrics as M
from .config import base_parser, load_config, set_seed, setup_console
from .evaluate import TEXT_VIEWS, default_experiments, evaluate, experiment_frames, load_dataset, save_run, summary_line

MODEL = "tfidf_lr"


def make_pipeline(cfg: dict, C: float, seed: int) -> Pipeline:
    t = cfg["tfidf"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=t["analyzer"],
                    ngram_range=tuple(t["ngram_range"]),
                    min_df=t["min_df"],
                    max_features=t["max_features"],
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            ("lr", LogisticRegression(C=C, max_iter=2000, solver="liblinear", class_weight="balanced", random_state=seed)),
        ]
    )


def fit_select(train: pd.DataFrame, val: pd.DataFrame, cfg: dict, seed: int) -> tuple[Pipeline, float]:
    """Pick C on validation AUC. TF-IDF is fit on train only (inside the pipeline)."""
    best, best_auc, best_c = None, -1.0, None
    for C in cfg["tfidf"]["C_grid"]:
        pipe = make_pipeline(cfg, C, seed).fit(train["text"], train["label"])
        a = M.auc(val["label"].values, pipe.decision_function(val["text"]))
        if a > best_auc:
            best, best_auc, best_c = pipe, a, C
    return best, best_c


def top_features(pipe: Pipeline, k: int = 50) -> pd.DataFrame:
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["lr"].coef_.ravel()
    order = np.argsort(coef)
    llm = [(names[i], coef[i]) for i in order[::-1][:k]]
    human = [(names[i], coef[i]) for i in order[:k]]
    return pd.DataFrame(
        {
            "rank": range(1, k + 1),
            "llm_ngram": [repr(n) for n, _ in llm],
            "llm_weight": [round(float(w), 3) for _, w in llm],
            "human_ngram": [repr(n) for n, _ in human],
            "human_weight": [round(float(w), 3) for _, w in human],
        }
    )


def run_experiment(df: pd.DataFrame, exp: str, cfg, seed: int) -> dict:
    train, val, test, view = experiment_frames(df, exp, seed)
    tf = TEXT_VIEWS[view]
    train, val, test = (d.assign(text=d["text"].map(tf)) for d in (train, val, test))
    pipe, C = fit_select(train, val, cfg, seed)
    val = val.assign(score=pipe.decision_function(val["text"]))
    test = test.assign(score=pipe.decision_function(test["text"]))
    res = evaluate(val, test, cfg, seed)
    out = cfg.path("results_dir") / MODEL / exp
    save_run(out, test, res, {"model": MODEL, "experiment": exp, "text_view": view, "C": C,
                              "n_train": len(train), "n_val": len(val), "n_test": len(test)})
    if exp in ("main", "ablation_surface", "ablation_nospace"):
        feats = top_features(pipe)
        feats.to_csv(out / "top_features.csv", index=False, encoding="utf-8-sig")
        if exp == "main":
            print("\n상위 특징 (main). 구두점·공백·ㅋ·어미만 잔뜩 있으면 지름길 학습을 의심하세요:")
            print(feats.head(20).to_string(index=False))
    print(summary_line(MODEL, exp, res))
    return res


def main(argv: list[str] | None = None) -> None:
    setup_console()
    p = base_parser("TF-IDF + Logistic Regression 베이스라인")
    p.add_argument("--exp", default="", help="쉼표로 구분한 실험 이름 (기본: 전부)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    df = load_dataset(cfg)
    exps = args.exp.split(",") if args.exp else default_experiments(df)
    for exp in exps:
        run_experiment(df, exp, cfg, cfg["seed"])


if __name__ == "__main__":
    main()
