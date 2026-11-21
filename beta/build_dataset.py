"""Day 1 evening: merge, normalize, length-match, split, and freeze the test set.

Usage:
    python -m beta.build_dataset            # 처음 만들 때
    python -m beta.build_dataset --force    # 테스트셋이 바뀌어도 덮어쓰기 (결과 보고 나서는 쓰지 말 것)
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from .config import base_parser, load_config, setup_console
from .normalize import bin_labels, is_korean_text, length_bin, near_dup_key, normalize

SPLITS = ("train", "val", "test")


def load_generated(path) -> pd.DataFrame:
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(recs)
    df = df[df["status"] == "ok"]
    # last successful record per condition wins; sort so async completion order
    # doesn't change the sampled split on a rebuild
    return df.drop_duplicates("cond_id", keep="last").sort_values("cond_id").reset_index(drop=True)


def assemble(human: pd.DataFrame, gen: pd.DataFrame, conds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    edges = cfg["human"]["length_edges"]
    gen = gen.merge(conds[["cond_id", "rating", "category", "target_chars"]], on="cond_id", how="left")
    llm = pd.DataFrame(
        {
            "id": gen["cond_id"],
            "group_id": gen["cond_id"],
            "text": gen["text"],
            "label": 1,
            "source": gen["generator"],
            "family": gen["family"],
            "prompt_type": gen["prompt_type"],
            "rating": gen["rating"],
            "category": gen["category"],
        }
    )
    hum = pd.DataFrame(
        {
            "id": human["id"],
            "group_id": human["id"],
            "text": human["text"],
            "label": 0,
            "source": "human",
            "family": "human",
            "prompt_type": "human",
            "rating": human["rating"],
            "category": human["category"],
        }
    )
    df = pd.concat([hum, llm], ignore_index=True)
    df["text"] = df["text"].map(normalize)  # identical normalization for both classes
    korean = df["text"].map(is_korean_text)  # same language filter for both classes
    print("한국어 필터로 제외 (사람 / LLM):", int((~korean & (df["label"] == 0)).sum()), "/", int((~korean & (df["label"] == 1)).sum()))
    df = df[korean]
    df["length"] = df["text"].str.len()
    h = cfg["human"]
    df = df[(df["length"] >= h["min_chars"]) & (df["length"] <= h["max_chars"])]
    # Exact and near duplicates: keep the first (humans come first).
    df = df.drop_duplicates("text")
    df = df[~df["text"].map(near_dup_key).duplicated()]
    df["length_bin"] = df["length"].map(lambda n: length_bin(int(n), edges))
    return df.reset_index(drop=True)


def balance_by_length(df: pd.DataFrame, edges: list[int], rng: np.random.Generator) -> pd.DataFrame:
    """Per length bin, keep the same number of human and LLM reviews."""
    parts = []
    print("\n길이 구간별 매칭 (사람 / LLM -> 유지):")
    for b in bin_labels(edges):
        hb = df[(df["length_bin"] == b) & (df["label"] == 0)]
        lb = df[(df["length_bin"] == b) & (df["label"] == 1)]
        n = min(len(hb), len(lb))
        flag = "  [경고] 샘플이 너무 적음" if n < 100 else ""
        print(f"  {b:>8}: {len(hb):>6,} / {len(lb):>6,} -> {n:,}{flag}")
        for part in (hb, lb):
            if n:
                parts.append(part.iloc[np.sort(rng.choice(len(part), size=n, replace=False))])
    return pd.concat(parts).reset_index(drop=True)


def assign_splits(df: pd.DataFrame, ratios: list[float], rng: np.random.Generator) -> pd.Series:
    """Group-aware stratified split: a group never crosses splits.

    Strata = label x length bin x source x prompt type, so every generator and
    prompt type shows up in every split in the same proportion.
    """
    strata = df["label"].astype(str) + "|" + df["length_bin"] + "|" + df["source"] + "|" + df["prompt_type"]
    group_stratum = pd.Series(strata.values, index=df["group_id"].values)
    group_stratum = group_stratum[~group_stratum.index.duplicated()]
    split_of_group: dict[str, str] = {}
    cum = np.cumsum(ratios) / np.sum(ratios)
    for _, groups in group_stratum.groupby(group_stratum):
        ids = groups.index.to_numpy()
        ids = ids[rng.permutation(len(ids))]
        bounds = np.round(cum * len(ids)).astype(int)
        for i, gid in enumerate(ids):
            split_of_group[gid] = SPLITS[int(np.searchsorted(bounds, i, side="right"))]
    return df["group_id"].map(split_of_group)


def test_fingerprint(df: pd.DataFrame) -> str:
    test = df[df["split"] == "test"].sort_values("id")
    h = hashlib.sha256()
    for i, t in zip(test["id"], test["text"]):
        h.update(f"{i}\t{t}\n".encode("utf-8"))
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    setup_console()
    p = base_parser("사람+LLM 리뷰 합치기, 길이 매칭, 분할, 테스트셋 고정")
    p.add_argument("--force", action="store_true", help="고정된 테스트셋이 바뀌어도 덮어쓰기")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["seed"])
    interim, processed = cfg.path("interim_dir"), cfg.path("processed_dir")

    human = pd.read_parquet(interim / "human.parquet")
    conds = pd.read_parquet(interim / "conditions.parquet")
    gen = load_generated(interim / "generated.jsonl")
    print(f"사람 {len(human):,}건, LLM {len(gen):,}건 (status=ok)")

    df = assemble(human, gen, conds, cfg)
    if cfg["human"].get("balance_bins", True):
        df = balance_by_length(df, cfg["human"]["length_edges"], rng)
    df["split"] = assign_splits(df, cfg["split"]["ratios"], rng)

    fp = test_fingerprint(df)
    freeze_path = processed / "test_freeze.json"
    if freeze_path.exists():
        old = json.loads(freeze_path.read_text(encoding="utf-8"))
        if old["sha256"] != fp and not args.force:
            raise SystemExit(
                "테스트셋이 고정본과 다릅니다. 결과를 이미 봤다면 테스트셋을 바꾸면 안 됩니다.\n"
                "정말 다시 만들려면 --force 를 붙이세요."
            )
    df.to_parquet(processed / "dataset.parquet", index=False)
    freeze_path.write_text(
        json.dumps(
            {"sha256": fp, "n_test": int((df["split"] == "test").sum()), "created_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n데이터셋 {len(df):,}건 -> {processed / 'dataset.parquet'}")
    print(pd.crosstab(df["split"], df["label"].map({0: "human", 1: "llm"})).to_string())
    print("\n출처 x 프롬프트:")
    print(pd.crosstab(df["source"], df["prompt_type"]).to_string())
    print(f"\n테스트셋 고정: {freeze_path} (sha256 {fp[:12]}...)")


if __name__ == "__main__":
    main()
