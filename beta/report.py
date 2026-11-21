"""Day 3: collect every run into results/REPORT.md, check the go/no-go gates,
and export the most confident mistakes for reading by hand.

Usage:
    python -m beta.report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import base_parser, load_config, setup_console
from .evaluate import fmt, load_dataset
from .normalize import bin_labels

THR = "val_thr_fpr0.05"


def load_runs(results: Path) -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for f in sorted(results.glob("*/*/metrics.json")):
        model, exp = f.parent.parent.name, f.parent.name
        runs.setdefault(model, {})[exp] = json.loads(f.read_text(encoding="utf-8"))
    return runs


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _sub(run: dict, name: str) -> dict:
    return run.get("subsets", {}).get(name, {})


def _thr(sub: dict, key: str) -> float | None:
    return sub.get(THR, {}).get(key)


def main_table(runs) -> str:
    rows = []
    for model, exps in runs.items():
        if "main" not in exps:
            continue
        o = _sub(exps["main"], "overall")
        ci = o.get("auc_ci95")
        prev = o.get(THR, {}).get("precision_at_prevalence", {})
        rows.append(
            [
                model,
                f"{fmt(o.get('auc'))}" + (f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else ""),
                fmt(o.get("ap")),
                fmt(o.get("oracle_tpr@fpr0.01")),
                fmt(o.get("oracle_tpr@fpr0.05")),
                fmt(_thr(o, "tpr")),
                fmt(_thr(o, "fpr")),
                fmt(_thr(o, "f1")),
                fmt(prev.get("0.05")),
            ]
        )
    return _table(
        ["모델", "AUC [95% CI]", "AP", "TPR@FPR1% (oracle)", "TPR@FPR5% (oracle)",
         "TPR (val 임계값, FPR5%)", "실제 FPR", "F1", "Precision (실제 비율 5% 가정)"],
        rows,
    )


def breakdown_table(runs, exp: str, prefix: str, order: list[str] | None = None) -> str:
    names = sorted({k for exps in runs.values() if exp in exps for k in exps[exp]["subsets"] if k.startswith(prefix)})
    if order:
        names = [f"{prefix}{o}" for o in order if f"{prefix}{o}" in names]
    rows = []
    for name in names:
        row = [name.removeprefix(prefix)]
        for model, exps in runs.items():
            if exp in exps:
                s = _sub(exps[exp], name)
                row.append(f"{fmt(s.get('auc'))} / {fmt(_thr(s, 'tpr'))} / {fmt(_thr(s, 'fpr'))} (n={s.get('n_human', 0)}+{s.get('n_llm', 0)})")
        rows.append(row)
    models = [m for m, exps in runs.items() if exp in exps]
    return _table([prefix.rstrip("=")] + [f"{m}: AUC / TPR / FPR" for m in models], rows)


def holdout_table(runs, prefix: str) -> tuple[str, dict[str, float]]:
    rows, means = [], {}
    for model, exps in runs.items():
        hold = {e: r for e, r in exps.items() if e.startswith(prefix)}
        aucs = []
        for e, r in sorted(hold.items()):
            o = _sub(r, "overall")
            aucs.append(o.get("auc"))
            rows.append([model, e.removeprefix(prefix), fmt(o.get("auc")), fmt(_thr(o, "tpr")), fmt(_thr(o, "fpr")),
                         f"{o.get('n_human', 0)}+{o.get('n_llm', 0)}"])
        if aucs and all(a is not None for a in aucs):
            means[model] = float(np.mean(aucs))
            rows.append([model, "**평균**", f"**{fmt(means[model])}**", "", "", ""])
    return _table(["모델", "빼둔 것", "AUC", "TPR (val 임계값)", "실제 FPR", "n (사람+LLM)"], rows), means


def ablation_table(runs) -> tuple[str, dict[str, float]]:
    rows, drops = [], {}
    for model, exps in runs.items():
        if "main" not in exps:
            continue
        base = _sub(exps["main"], "overall").get("auc")
        row = [model, fmt(base)]
        for view in ("ablation_surface", "ablation_nospace"):
            a = _sub(exps[view], "overall").get("auc") if view in exps else None
            row.append(fmt(a) + (f" ({a - base:+.3f})" if a is not None and base is not None else ""))
            if view == "ablation_surface" and a is not None and base is not None:
                drops[model] = base - a
        rows.append(row)
    return _table(["모델", "AUC 원문", "AUC 구두점·이모지·자모 제거", "AUC + 띄어쓰기 제거"], rows), drops


def gates(cfg, runs, drops, logo_means) -> str:
    g = cfg["eval"]["gates"]
    long_key = f"length>={cfg['eval']['gate_min_chars']}"
    rows = []
    for model, exps in runs.items():
        if "main" not in exps:
            continue
        logo = logo_means.get(model)
        logo_note = ""
        if logo is None and model.startswith("zeroshot"):
            # zero-shot never trains, so every generator is already "unseen"
            vals = [s.get("auc") for k, s in exps["main"]["subsets"].items() if k.startswith("generator=")]
            if vals and all(v is not None for v in vals):
                logo, logo_note = float(np.mean(vals)), " (zero-shot: 생성기별 평균)"
        long_tpr = _thr(_sub(exps["main"], long_key), "tpr")
        checks = [
            ("정규화 ablation 후 AUC 하락", drops.get(model), f"< {g['ablation_auc_drop_max']}",
             None if model not in drops else drops[model] < g["ablation_auc_drop_max"]),
            ("생성기 hold-out 평균 AUC" + logo_note, logo, f">= {g['logo_mean_auc_min']}",
             None if logo is None else logo >= g["logo_mean_auc_min"]),
            (f"{cfg['eval']['gate_min_chars']}자 이상 TPR (FPR5% 임계값)", long_tpr, f">= {g['long_tpr_at_fpr05_min']}",
             None if long_tpr is None else long_tpr >= g["long_tpr_at_fpr05_min"]),
        ]
        for name, val, target, ok in checks:
            verdict = "해당 없음" if ok is None else ("통과" if ok else "**미달**")
            rows.append([model, name, fmt(val), target, verdict])
    return _table(["모델", "기준", "값", "목표", "판정"], rows)


def export_errors(cfg, runs, df: pd.DataFrame, k: int = 30) -> list[Path]:
    out_paths = []
    results = cfg.path("results_dir")
    text = df.set_index("id")["text"]
    for model in runs:
        pred_path = results / model / "main" / "predictions.parquet"
        if not pred_path.exists():
            continue
        pred = pd.read_parquet(pred_path)
        thr = runs[model]["main"]["thresholds"].get(THR)
        fp = pred[pred["label"] == 0].nlargest(k, "score").assign(kind="사람 리뷰 중 AI 점수 높은 순 (오탐 후보)")
        fn = pred[pred["label"] == 1].nsmallest(k, "score").assign(kind="LLM 리뷰 중 AI 점수 낮은 순 (미탐 후보)")
        err = pd.concat([fp, fn])
        err["text"] = err["id"].map(text)
        flagged = err["score"] > thr if thr is not None else pd.Series(np.nan, index=err.index)
        # an actual mistake at the FPR-5% operating threshold
        err["wrong_at_fpr5_threshold"] = np.where(err["label"] == 0, flagged, ~flagged.astype(bool))
        path = results / f"error_analysis_{model}.csv"
        err[["kind", "id", "source", "prompt_type", "length", "score", "wrong_at_fpr5_threshold", "text"]].to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        out_paths.append(path)
    return out_paths


def main(argv: list[str] | None = None) -> None:
    setup_console()
    p = base_parser("결과 보고서 생성")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    bins = bin_labels(cfg["human"]["length_edges"])
    results = cfg.path("results_dir")
    runs = load_runs(results)
    if not runs:
        raise SystemExit(f"{results} 에 결과가 없습니다. 먼저 모델을 돌리세요.")
    df = load_dataset(cfg)

    abl, drops = ablation_table(runs)
    logo, logo_means = holdout_table(runs, "logo_")
    lopo, _ = holdout_table(runs, "lopo_")
    err_paths = export_errors(cfg, runs, df)

    counts = pd.crosstab(df["split"], df["label"].map({0: "human", 1: "llm"}))
    src = pd.crosstab(df["source"], df["prompt_type"])
    freeze = json.loads((cfg.path("processed_dir") / "test_freeze.json").read_text(encoding="utf-8"))

    md = f"""# 베타 결과 보고서

- 생성: 테스트셋 sha256 `{freeze['sha256'][:16]}` (고정 시각 {freeze['created_at']})
- 점수는 확률이 아니라 "LLM다움" 점수입니다. 임계값은 검증셋에서 FPR 1%/5%가 되도록 정한 뒤 테스트에 그대로 적용했습니다.
- "oracle"은 테스트셋에서 직접 고른 임계값이라 낙관적인 값입니다. 실제 운영 성능은 "val 임계값" 쪽을 보세요.

## 1. 진행 여부 기준

{gates(cfg, runs, drops, logo_means)}

## 2. 전체 성능 (main)

{main_table(runs)}

## 3. 지름길 점검: 정규화 ablation

괄호 안은 원문 대비 변화. 크게 떨어지면 모델이 구두점·이모지·띄어쓰기 같은 겉모습을 보고 있다는 뜻입니다.
TF-IDF 상위 특징은 `results/tfidf_lr/main/top_features.csv` 에서 직접 확인하세요.

{abl}

## 4. 미학습 생성기 (generator hold-out)

해당 계열을 학습·검증에서 빼고, 그 계열의 리뷰 전부와 사람 테스트셋으로 평가했습니다. 임계값은 빼고 남은 검증셋 기준입니다.

{logo}

## 5. 미학습 프롬프트 (prompt hold-out)

{lopo}

## 6. 길이 구간별 (main)

{breakdown_table(runs, "main", "length=", bins)}

## 7. 생성기별 / 프롬프트별 (main)

{breakdown_table(runs, "main", "generator=")}

{breakdown_table(runs, "main", "prompt=")}

## 8. 데이터

{_table(["split"] + list(counts.columns), [[i] + [str(v) for v in r] for i, r in zip(counts.index, counts.values)])}

{_table(["source"] + list(src.columns), [[i] + [str(v) for v in r] for i, r in zip(src.index, src.values)])}

## 9. 오류 직접 읽기

사람 리뷰 중 AI 점수가 가장 높은 30건 + LLM 리뷰 중 가장 낮은 30건. 실제로 틀린 건 `wrong_at_fpr5_threshold=True` 인 행입니다:
{chr(10).join(f"- `{(p.relative_to(cfg.base_dir) if p.is_relative_to(cfg.base_dir) else p).as_posix()}`" for p in err_paths) or "- (없음)"}

## 10. 이번 베타의 한계

- 생성기 {df.loc[df['label'] == 1, 'family'].nunique()}개 계열로는 "미학습 LLM 전반"으로의 일반화를 주장하기 어렵습니다.
- 사람 리뷰는 KR3 (2022년 3월 이전 공개, 5개 사이트 크롤링) 한 출처입니다. 네이버 플레이스 등 다른 플랫폼 스타일과 실제 서비스 환경의 LLM 리뷰 비율은 검증하지 않았습니다.
- 사람이 쓴 리뷰를 LLM이 다듬은 경우와 사람·LLM 혼합 글은 평가하지 않았습니다.
"""
    path = results / "REPORT.md"
    path.write_text(md, encoding="utf-8")
    print(f"보고서 -> {path}")
    for p_ in err_paths:
        print(f"오류 사례 -> {p_}")


if __name__ == "__main__":
    main()
