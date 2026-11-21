"""Offline end-to-end run: synthetic KR3 -> mock generation -> dataset ->
TF-IDF, tiny KcELECTRA stand-in, tiny zero-shot LM -> report."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from beta import (build_dataset, embed_baseline, export_benchmark, generate, prepare_human, report, score_predictions,
                  tfidf_baseline, train_electra, zeroshot)


@pytest.fixture(scope="module")
def built(project):
    c = project["config"]
    prepare_human.main(["--config", c, "--input", str(project["tsv"])])
    generate.main(["--config", c, "--mock"])
    build_dataset.main(["--config", c])
    return project


def _p(project, *parts) -> Path:
    return Path(project["root"]).joinpath(*parts)


def test_human_side(built):
    human = pd.read_parquet(_p(built, "data/interim/human.parquet"))
    pool = pd.read_parquet(_p(built, "data/interim/fewshot_pool.parquet"))
    assert len(human) == 400
    assert not set(human["text"]) & set(pool["text"]), "few-shot pool must not overlap the human set"
    assert not human["text"].str.contains("\n").any()


def test_generation_records(built):
    recs = [json.loads(l) for l in open(_p(built, "data/interim/generated.jsonl"), encoding="utf-8")]
    statuses = {r["status"] for r in recs}
    assert "ok" in statuses and "rejected" in statuses  # mock emits some "[가게 이름]" outputs
    conds = pd.read_parquet(_p(built, "data/interim/conditions.parquet"))
    cells = conds.groupby(["generator", "prompt_type"]).size()
    assert cells.max() - cells.min() <= 1, "generator x prompt cells should be balanced"
    assert (conds.loc[conds["prompt_type"] == "C", "fewshot_ids"].str.count(",") == 3).all()


def test_rerun_generation_resumes(built):
    before = sum(1 for _ in open(_p(built, "data/interim/generated.jsonl"), encoding="utf-8"))
    generate.main(["--config", built["config"], "--mock"])
    after_lines = [json.loads(l) for l in open(_p(built, "data/interim/generated.jsonl"), encoding="utf-8")]
    # only previously rejected conditions are retried
    assert before < len(after_lines) < before * 1.5


def test_dataset_and_freeze(built):
    df = pd.read_parquet(_p(built, "data/processed/dataset.parquet"))
    assert set(df["split"]) == {"train", "val", "test"}
    assert df.groupby("group_id")["split"].nunique().max() == 1
    assert not df["text"].str.contains(r"\*\*|^리뷰:", regex=True).any()
    per_bin = df.groupby(["length_bin", "label"]).size().unstack()
    assert (per_bin[0] == per_bin[1]).all(), "length bins must be matched between classes"
    # same inputs -> same test set, so a rebuild is allowed
    build_dataset.main(["--config", built["config"]])


def test_freeze_blocks_changed_testset(built, tmp_path):
    freeze = _p(built, "data/processed/test_freeze.json")
    saved = freeze.read_text(encoding="utf-8")
    data = json.loads(saved)
    data["sha256"] = "0" * 64
    freeze.write_text(json.dumps(data), encoding="utf-8")
    try:
        with pytest.raises(SystemExit):
            build_dataset.main(["--config", built["config"]])
    finally:
        freeze.write_text(saved, encoding="utf-8")


def test_models_and_report(built):
    c = built["config"]
    tfidf_baseline.main(["--config", c])
    train_electra.main(["--config", c, "--exp", "main,logo_exaone"])
    zeroshot.main(["--config", c])
    report.main(["--config", c])

    res = _p(built, "results")
    for model, exp in [("tfidf_lr", "main"), ("tfidf_lr", "ablation_surface"), ("tfidf_lr", "logo_kanana"),
                       ("tfidf_lr", "lopo_C"), ("kcelectra", "main"), ("kcelectra", "logo_exaone"),
                       ("zeroshot_fastdetect", "main"), ("zeroshot_loglik", "main")]:
        m = json.loads((res / model / exp / "metrics.json").read_text(encoding="utf-8"))
        assert 0.0 <= m["subsets"]["overall"]["auc"] <= 1.0, (model, exp)

    # generator hold-out test = human test + every row of the held-out family
    logo = pd.read_parquet(res / "tfidf_lr" / "logo_kanana" / "predictions.parquet")
    assert set(logo.loc[logo["label"] == 1, "family"]) == {"kanana"}
    assert set(logo.loc[logo["label"] == 0, "split"]) == {"test"}

    md = (res / "REPORT.md").read_text(encoding="utf-8")
    for heading in ("진행 여부 기준", "정규화 ablation", "미학습 생성기", "길이 구간별"):
        assert heading in md
    err = pd.read_csv(res / "error_analysis_tfidf_lr.csv")
    assert len(err) == 60 and err["text"].notna().all()
    feats = pd.read_csv(res / "tfidf_lr" / "main" / "top_features.csv")
    assert len(feats) == 50


def test_embedding_baseline(built):
    cls_dir = built["cfg"]["electra"]["model_name"]
    args = ["--config", built["config"], "--model-name", cls_dir, "--name", "emb_tiny_lr", "--exp", "main,logo_qwen"]
    embed_baseline.main(args)
    embed_baseline.main(args)  # second run reuses the cached embeddings
    m = json.loads(_p(built, "results/emb_tiny_lr/logo_qwen/metrics.json").read_text(encoding="utf-8"))
    assert m["subsets"]["overall"]["n_llm"] > 0
    assert list(_p(built, "data/embeddings").glob("*.npz"))


def test_team_benchmark_roundtrip(built, tmp_path):
    c = built["config"]
    out = tmp_path / "benchmark"
    export_benchmark.main(["--config", c, "--out", str(out)])
    reviews = pd.read_csv(out / "reviews.csv", dtype={"id": str})
    splits = pd.read_csv(out / "splits.csv", dtype={"id": str}, keep_default_na=False)
    assert {"main", "logo_exaone", "ablation_surface"} <= set(splits.columns)
    assert (splits["main"] == splits["ablation_surface"]).all()
    assert "test" in set(splits["logo_exaone"]) and "" in set(splits["logo_exaone"])
    assert (out / "llm_generated_all.csv").exists() and "sha256" in (out / "PROTOCOL.md").read_text(encoding="utf-8")

    # a teammate's "model": score = number of periods, written for every row
    pred_dir = tmp_path / "predictions" / "periods_test"
    pred_dir.mkdir(parents=True)
    pred = pd.DataFrame({"id": reviews["id"], "score": reviews["text"].str.count(r"\.")})
    for exp in ("main", "logo_exaone"):
        pred.to_csv(pred_dir / f"{exp}.csv", index=False)
    score_predictions.main(["--config", c, "--model", "periods_test", "--dir", str(pred_dir),
                            "--dataset", str(out / "dataset.parquet")])
    m = json.loads(_p(built, "results/periods_test/logo_exaone/metrics.json").read_text(encoding="utf-8"))
    assert m["test_sha256"] == json.loads((out / "test_freeze.json").read_text(encoding="utf-8"))["sha256"]

    pred.iloc[: len(pred) // 2].to_csv(pred_dir / "main.csv", index=False)  # missing rows must be rejected
    with pytest.raises(SystemExit):
        score_predictions.main(["--config", c, "--model", "periods_test", "--dir", str(pred_dir),
                                "--dataset", str(out / "dataset.parquet")])


def test_replan_backs_up_old_generations(built):
    interim = _p(built, "data/interim")
    generate.main(["--config", built["config"], "--mock", "--replan", "--pilot", "1"])
    assert list(interim.glob("generated.*.jsonl.bak")), "old generations must be moved aside, not mixed in"


def test_fastdetect_prefers_greedy_text():
    """Greedy tokens are, by construction, at least as likely as the model's
    expectation; uniform random tokens are less likely in expectation."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_embd=32, n_layer=1, n_head=2, n_positions=64)).eval()
    start = torch.tensor([[1]])
    greedy = model.generate(start, max_new_tokens=30, do_sample=False, pad_token_id=0)
    rand = torch.cat([start, torch.randint(0, 50, (1, 30))], dim=1)
    ids = torch.cat([greedy, rand])
    fast, ll = zeroshot.score_batch(model, ids, torch.ones_like(ids))
    assert fast[0] > fast[1] and ll[0] > ll[1]
