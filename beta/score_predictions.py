"""Score a teammate's predictions with the shared protocol.

predictions/<model>/<experiment>.csv  ->  results/<model>/<experiment>/metrics.json

Usage:
    python -m beta.score_predictions --model e5base_lr_홍길동 --dir predictions/e5base_lr_홍길동
    python -m beta.score_predictions --model ... --dir ... --dataset benchmark/dataset.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_dataset import test_fingerprint
from .config import base_parser, load_config, setup_console
from .evaluate import evaluate, experiment_frames, save_run, summary_line


def score_one(df: pd.DataFrame, csv: Path, model: str, cfg, fp: str) -> dict:
    exp = csv.stem
    _, val, test, _ = experiment_frames(df, exp, cfg["seed"])
    pred = pd.read_csv(csv, dtype={"id": str})
    if not {"id", "score"} <= set(pred.columns):
        raise SystemExit(f"{csv}: 'id,score' 컬럼이 필요합니다. 실제 컬럼: {list(pred.columns)}")
    if pred["id"].duplicated().any():
        raise SystemExit(f"{csv}: 중복 id가 있습니다.")
    scores = dict(zip(pred["id"], pd.to_numeric(pred["score"], errors="coerce")))
    need = pd.concat([val, test])["id"]
    missing = need[~need.isin(scores.keys())]
    if len(missing):
        raise SystemExit(f"{csv}: val/test 행 {len(missing):,}개에 점수가 없습니다. 예: {missing.head(5).tolist()}")
    val = val.assign(score=val["id"].map(scores).astype(float))
    test = test.assign(score=test["id"].map(scores).astype(float))
    if not (np.isfinite(val["score"]).all() and np.isfinite(test["score"]).all()):
        raise SystemExit(f"{csv}: 점수에 NaN/inf 가 있습니다.")
    res = evaluate(val, test, cfg, cfg["seed"])
    save_run(cfg.path("results_dir") / model / exp, test, res,
             {"model": model, "experiment": exp, "submitted_from": str(csv), "test_sha256": fp})
    print(summary_line(model, exp, res))
    return res


def main(argv: list[str] | None = None) -> None:
    setup_console()
    p = base_parser("팀원 예측 파일 채점")
    p.add_argument("--model", required=True, help="결과 이름 (본인 이름을 넣어 겹치지 않게)")
    p.add_argument("--dir", required=True, help="<실험이름>.csv 파일들이 있는 폴더")
    p.add_argument("--dataset", default=None, help="dataset.parquet 경로 (기본: data/processed)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    ds_path = Path(args.dataset) if args.dataset else cfg.path("processed_dir") / "dataset.parquet"
    df = pd.read_parquet(ds_path)
    fp = test_fingerprint(df)
    freeze = ds_path.with_name("test_freeze.json")
    if freeze.exists() and json.loads(freeze.read_text(encoding="utf-8"))["sha256"] != fp:
        raise SystemExit("dataset.parquet 이 test_freeze.json 과 다릅니다. 공유받은 벤치마크 파일을 그대로 쓰세요.")
    print(f"테스트셋 sha256 {fp[:16]}")

    files = sorted(Path(args.dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"{args.dir} 에 <실험이름>.csv 가 없습니다.")
    for csv in files:
        try:
            score_one(df, csv, args.model, cfg, fp)
        except ValueError as e:
            print(f"[건너뜀] {csv.name}: 알 수 없는 실험 이름입니다 ({e}).")


if __name__ == "__main__":
    main()
