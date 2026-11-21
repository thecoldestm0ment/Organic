"""Day 1 morning: build the human side from KR3 raw.

- KR3 was released in Feb-Mar 2022, before ChatGPT, so these count as human.
- Uses kr3_raw (not the preprocessed kr3): the preprocessed version removed
  emoji, ㅋ/ㅠ, ~ and ran a spell checker, which would hand the classifier a
  free shortcut ("has ㅋㅋ -> human").
- Outputs a length-balanced human sample plus a disjoint few-shot pool used by
  the style-mimicking prompt.

Usage:
    python -m beta.prepare_human              # downloads kr3_raw.tsv (~218MB) if missing
    python -m beta.prepare_human --input path/to/kr3_raw.tsv
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import base_parser, load_config, set_seed, setup_console
from .normalize import bin_labels, length_bin, near_dup_key, normalize


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"다운로드: {url}\n -> {dest}")

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0 and blocks % 400 == 0:
            print(f"  {min(blocks * block_size / total, 1.0):6.1%}", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(dest)
    return dest


def read_kr3(path: Path, expected_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, on_bad_lines="warn")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    cols = {str(c).strip().lower(): c for c in df.columns}
    missing = {"rating", "review"} - cols.keys()
    if missing:
        raise ValueError(f"KR3 파일에 {missing} 컬럼이 없습니다. 실제 컬럼: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "rating": df[cols["rating"]].astype(str).str.strip(),
            "raw_text": df[cols["review"]].astype(str),
            # Region is often the crawl query, not the real location (KR3 README) -> not used.
            "category": df[cols["category"]].astype(str).str.strip() if "category" in cols else "",
        }
    )
    print(f"읽은 행 수: {len(out):,}")
    if expected_rows and abs(len(out) - expected_rows) > 0.01 * expected_rows:
        print(
            f"[경고] 예상 행 수 {expected_rows:,}와 1% 이상 차이 납니다. "
            "줄바꿈/따옴표 때문에 TSV 파싱이 깨졌을 수 있으니 확인하세요."
        )
    print("Rating 분포:", out["rating"].value_counts().to_dict())
    return out


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h = cfg["human"]
    n0 = len(df)
    df = df[df["rating"].isin(["0", "1", "2"])].copy()
    df["rating"] = df["rating"].astype(int)
    df["text"] = df["raw_text"].map(normalize)
    df["length"] = df["text"].str.len()
    df = df[(df["length"] >= h["min_chars"]) & (df["length"] <= h["max_chars"])]
    # Some KR3 sources were crawled from previews cut at ~200 chars + "...".
    # LLM text never looks like that, so it would be a free "human" signal.
    lo, hi = h.get("truncated_len_range", [190, 205])
    truncated = df["text"].str.endswith("...") & df["length"].between(lo, hi)
    df = df[~truncated]
    print(f"크롤링 미리보기로 잘린 리뷰(~200자 + '...') 제거: {int(truncated.sum()):,}건")
    df = df.drop_duplicates("text")
    df["dup_key"] = df["text"].map(near_dup_key)
    df = df.drop_duplicates("dup_key")
    df["length_bin"] = df["length"].map(lambda n: length_bin(int(n), h["length_edges"]))
    print(f"정제 후: {len(df):,} / {n0:,} (길이 필터 + 중복/준중복 제거)")
    return df.drop(columns=["raw_text"]).reset_index(drop=True)


def sample_human(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    h = cfg["human"]
    n = h["n_samples"]
    if not h.get("balance_bins", True):
        idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
        return df.iloc[np.sort(idx)]
    labels = bin_labels(h["length_edges"])
    quota = n // len(labels)
    parts = []
    for label in labels:
        pool = df[df["length_bin"] == label]
        take = min(quota, len(pool))
        if take < quota:
            print(f"[경고] 길이 구간 {label}: {len(pool)}건뿐이라 {quota}건을 못 채웁니다.")
        parts.append(pool.iloc[rng.choice(len(pool), size=take, replace=False)])
    return pd.concat(parts)


def main(argv: list[str] | None = None) -> None:
    setup_console()
    p = base_parser("KR3 raw에서 사람 리뷰 샘플과 few-shot 풀 만들기")
    p.add_argument("--input", default=None, help="이미 받은 kr3_raw.tsv 경로")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    if args.input:
        path = Path(args.input)
    else:
        path = download(cfg["kr3"]["url"], cfg.path("raw_dir") / cfg["kr3"]["filename"])

    df = clean(read_kr3(path, cfg["kr3"].get("expected_rows")), cfg)
    human = sample_human(df, cfg, rng)
    rest = df.drop(index=human.index)
    pool_n = min(cfg["human"]["fewshot_pool_size"], len(rest))
    pool = rest.iloc[rng.choice(len(rest), size=pool_n, replace=False)]

    human = human.reset_index(drop=True)
    human.insert(0, "id", [f"h{i:06d}" for i in range(len(human))])
    pool = pool.reset_index(drop=True)
    pool.insert(0, "id", [f"p{i:06d}" for i in range(len(pool))])

    out = cfg.path("interim_dir")
    cols = ["id", "text", "rating", "category", "length", "length_bin"]
    human[cols].to_parquet(out / "human.parquet", index=False)
    pool[cols].to_parquet(out / "fewshot_pool.parquet", index=False)

    print(f"\n사람 리뷰 {len(human):,}건 -> {out / 'human.parquet'}")
    print(human["length_bin"].value_counts().reindex(bin_labels(cfg["human"]["length_edges"])).to_string())
    print(f"few-shot 풀 {len(pool):,}건 -> {out / 'fewshot_pool.parquet'}")


if __name__ == "__main__":
    main()
