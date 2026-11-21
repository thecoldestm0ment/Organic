"""Package the frozen dataset so every teammate trains and scores on exactly
the same rows, whatever model they use.

Writes benchmark/:
    reviews.csv          id, text (+ ablation views), label, metadata, split
    splits.csv           one column per experiment: train / val / test / (blank = unused)
    llm_generated_all.csv every successful LLM generation (reference only)
    dataset.parquet, test_freeze.json   for `python -m beta.score_predictions`
    PROTOCOL.md          rules + submission format

Usage:
    python -m beta.export_benchmark
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .build_dataset import load_generated, test_fingerprint
from .config import base_parser, load_config, setup_console
from .evaluate import default_experiments, experiment_frames, load_dataset
from .normalize import strip_surface, strip_surface_nospace

TEXT_COLUMN = {"none": "text", "surface": "text_surface", "nospace": "text_nospace"}


def build_splits(df: pd.DataFrame, exps: list[str], seed: int) -> tuple[pd.DataFrame, dict[str, str]]:
    splits = pd.DataFrame({"id": df["id"].values})
    text_col = {}
    for exp in exps:
        tr, va, te, view = experiment_frames(df, exp, seed)
        role = pd.Series("", index=df["id"].values, dtype=object)
        role[tr["id"].values] = "train"
        role[va["id"].values] = "val"
        role[te["id"].values] = "test"
        splits[exp] = role.values
        text_col[exp] = TEXT_COLUMN[view]
    return splits, text_col


def protocol_md(exps: list[str], text_col: dict[str, str], fp: str, counts: pd.DataFrame) -> str:
    rows = "\n".join(f"| `{e}` | `{text_col[e]}` | {_desc(e)} |" for e in exps)
    table = "\n".join(
        ["| source | " + " | ".join(map(str, counts.columns)) + " |", "|---" * (len(counts.columns) + 1) + "|"]
        + [f"| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in zip(counts.index, counts.values)]
    )
    return f"""# 팀 공통 벤치마크 규칙

테스트셋 sha256: `{fp}`
점수를 공유할 때 이 값이 같아야 같은 시험지로 본 점수입니다.

## 파일

- `reviews.csv`: 리뷰 본문과 정답. `label` 1 = LLM, 0 = 사람. `text_surface` / `text_nospace` 는 ablation용 텍스트
- `splits.csv`: 실험마다 각 행이 train / val / test 중 어디에 쓰이는지. 빈칸은 그 실험에서 쓰지 않는 행
- `llm_generated_all.csv`: 생성된 LLM 리뷰 전체 (참고용. 점수 공유는 반드시 위 두 파일 기준)
- `dataset.parquet`, `test_freeze.json`: 공통 채점 스크립트용

## 실험

| 실험 | 쓸 텍스트 컬럼 | 내용 |
|---|---|---|
{rows}

필수: `main`, `logo_*` 전부. 권장: `ablation_surface`. 선택: 나머지.

## 규칙

1. 실험마다 `splits.csv` 에서 그 실험 컬럼이 `train` 인 행으로만 학습합니다.
2. 하이퍼파라미터, 에폭, 임베딩 모델 선택은 `val` 로만 합니다. **`test` 정답을 보고 고르면 안 됩니다.**
3. `val` 과 `test` 행 전부에 점수를 매깁니다. 점수는 클수록 LLM다운 쪽이면 아무 값이나 됩니다 (확률, 로짓, 거리 등).
4. 임계값은 직접 정하지 않습니다. 채점 스크립트가 val에서 FPR 1%/5% 임계값을 정해 test에 적용합니다.

## 제출 형식

`predictions/<모델이름>/<실험이름>.csv`, 컬럼은 `id,score` 두 개.

```
id,score
h000123,-2.31
g000456,1.87
```

## 채점

```
python -m beta.score_predictions --model e5base_lr_홍길동 --dir predictions/e5base_lr_홍길동 --dataset benchmark/dataset.parquet
python -m beta.report
```

모델 이름에 본인 이름을 넣어 겹치지 않게 하세요. 각자의 `results/<모델이름>/` 폴더를 한 곳에 모으고 `python -m beta.report` 를 돌리면 전원 점수가 한 보고서에 나옵니다.

## 라이선스

사람 리뷰는 KR3 (https://huggingface.co/datasets/leey4n/KR3, CC BY-NC-SA 4.0)에서 가져와 정규화·필터링했습니다.
이 벤치마크도 같은 **CC BY-NC-SA 4.0** 을 따릅니다: 출처(KR3) 표시, 비상업 용도만, 수정본도 같은 라이선스로 배포.
LLM 리뷰는 연구용으로 생성한 합성 데이터이며 실제 리뷰로 게시하는 데 쓰면 안 됩니다.

## 데이터 구성

{table}
"""


def _desc(exp: str) -> str:
    if exp == "main":
        return "전체 생성기·프롬프트로 학습/평가"
    if exp == "ablation_surface":
        return "main과 같은 분할, 구두점·이모지·자모 제거 텍스트"
    if exp == "ablation_nospace":
        return "main과 같은 분할, 띄어쓰기까지 제거"
    if exp.startswith("logo_"):
        return f"`{exp[5:]}` 계열을 학습에서 빼고, 그 계열 리뷰 전부로 평가 (미학습 생성기)"
    if exp.startswith("lopo_"):
        return f"프롬프트 `{exp[5:]}` 를 학습에서 빼고 평가 (미학습 프롬프트)"
    return ""


def main(argv: list[str] | None = None) -> None:
    setup_console()
    p = base_parser("팀 공유용 벤치마크 내보내기")
    p.add_argument("--out", default=None, help="출력 폴더 (기본: benchmark/)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else cfg.base_dir / "benchmark"
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg)
    exps = default_experiments(df)
    splits, text_col = build_splits(df, exps, cfg["seed"])

    reviews = df[["id", "text", "label", "source", "family", "prompt_type", "rating", "category", "length", "length_bin", "split"]].copy()
    reviews.insert(2, "text_surface", reviews["text"].map(strip_surface))
    reviews.insert(3, "text_nospace", reviews["text"].map(strip_surface_nospace))
    reviews.to_csv(out / "reviews.csv", index=False, encoding="utf-8-sig")
    splits.to_csv(out / "splits.csv", index=False, encoding="utf-8-sig")

    interim = cfg.path("interim_dir")
    gen = load_generated(interim / "generated.jsonl")
    conds = pd.read_parquet(interim / "conditions.parquet")
    gen = gen.merge(conds[["cond_id", "rating", "category", "target_chars"]], on="cond_id", how="left")
    gen[["cond_id", "generator", "family", "model", "prompt_type", "temperature", "rating", "category", "target_chars", "text"]].to_csv(
        out / "llm_generated_all.csv", index=False, encoding="utf-8-sig"
    )

    processed = cfg.path("processed_dir")
    shutil.copy2(processed / "dataset.parquet", out / "dataset.parquet")
    shutil.copy2(processed / "test_freeze.json", out / "test_freeze.json")
    fp = json.loads((out / "test_freeze.json").read_text(encoding="utf-8"))["sha256"]
    assert fp == test_fingerprint(df), "dataset.parquet 과 test_freeze.json 이 맞지 않습니다. build_dataset 을 다시 확인하세요."

    counts = pd.crosstab(df["source"], df["split"])
    (out / "PROTOCOL.md").write_text(protocol_md(exps, text_col, fp, counts), encoding="utf-8")
    print(f"벤치마크 -> {out}")
    print(f"  리뷰 {len(reviews):,}건, 실험 {len(exps)}개: {', '.join(exps)}")
    print(f"  LLM 생성 리뷰 전체 {len(gen):,}건 -> llm_generated_all.csv")
    print(f"  테스트셋 sha256 {fp[:16]}")


if __name__ == "__main__":
    main()
