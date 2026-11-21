"""Frozen embedding model + Logistic Regression. A template for teammates
trying several embedding models under the same protocol.

Embeddings are cached per (model, text view) in data/embeddings/, so trying a
new classifier on top does not re-encode anything.

Usage:
    python -m beta.embed_baseline --preset e5-base
    python -m beta.embed_baseline --preset bge-m3 --exp main,logo_openai
    python -m beta.embed_baseline --model-name jhgan/ko-sroberta-multitask --pooling mean --name koSRoBERTa_lr
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from . import metrics as M
from .config import base_parser, load_config, set_seed, setup_console
from .evaluate import TEXT_VIEWS, default_experiments, evaluate, experiment_frames, load_dataset, save_run, summary_line

# Pooling/prefix follow each model card.
PRESETS = {
    "bge-m3": {"model_name": "BAAI/bge-m3", "pooling": "cls", "prefix": ""},
    "e5-base": {"model_name": "intfloat/multilingual-e5-base", "pooling": "mean", "prefix": "query: "},
    "e5-large": {"model_name": "intfloat/multilingual-e5-large", "pooling": "mean", "prefix": "query: "},
    "ko-sroberta": {"model_name": "jhgan/ko-sroberta-multitask", "pooling": "mean", "prefix": ""},
    "kr-sbert": {"model_name": "snunlp/KR-SBERT-V40K-klueNLI-augSTS", "pooling": "mean", "prefix": ""},
    "kcelectra-frozen": {"model_name": "beomi/KcELECTRA-base-v2022", "pooling": "mean", "prefix": ""},
}
C_GRID = [0.1, 1.0, 10.0]


@torch.no_grad()
def embed(texts: list[str], tok, model, pooling: str, prefix: str, max_length: int, batch_size: int,
          device: torch.device) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])
    out = None
    amp = device.type == "cuda"
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        enc = tok([prefix + texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            h = model(**enc).last_hidden_state.float()
        if pooling == "cls":
            v = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        if out is None:
            out = np.zeros((len(texts), v.shape[1]), dtype=np.float32)
        out[idx] = v
        done = start + len(idx)
        if done % (batch_size * 50) < batch_size or done == len(texts):
            print(f"  임베딩 {done:,}/{len(texts):,} ({time.time() - t0:.0f}s)", flush=True)
    return out


def cached_embeddings(df: pd.DataFrame, view: str, spec: dict, cfg, args, device) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    cache_dir = cfg.base_dir / "data" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^0-9A-Za-z._-]+", "_", spec["model_name"])
    path = cache_dir / f"{key}__{spec['pooling']}__{view}.npz"
    ids = df["id"].to_numpy().astype(str)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if np.array_equal(z["ids"], ids):
            return z["emb"]
    tok = AutoTokenizer.from_pretrained(spec["model_name"])
    model = AutoModel.from_pretrained(spec["model_name"]).to(device).eval()
    max_len = min(args.max_length, tok.model_max_length)
    pos = getattr(model.config, "max_position_embeddings", None)
    if pos:  # RoBERTa-style models reserve 2 positions for the padding offset
        max_len = min(max_len, pos - 2 if "roberta" in model.config.model_type else pos)
    texts = df["text"].map(TEXT_VIEWS[view]).tolist()
    emb = embed(texts, tok, model, spec["pooling"], spec["prefix"], max_len, args.batch_size, device)
    np.savez(path, ids=ids, emb=emb)
    del model
    return emb


def main(argv: list[str] | None = None) -> None:
    setup_console()
    p = base_parser("임베딩 모델 + Logistic Regression")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument("--model-name", default=None, help="Hugging Face 모델 이름 (preset 대신)")
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument("--prefix", default="", help="입력 앞에 붙일 문자열 (e5는 'query: ')")
    p.add_argument("--name", default=None, help="결과 폴더 이름 (기본: emb_<preset>_lr)")
    p.add_argument("--exp", default="", help="쉼표로 구분 (기본: main, ablation_surface, 생성기 hold-out)")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    if not (args.preset or args.model_name):
        p.error("--preset 이나 --model-name 중 하나가 필요합니다")
    spec = dict(PRESETS[args.preset]) if args.preset else {"model_name": args.model_name, "pooling": args.pooling, "prefix": args.prefix}
    name = args.name or f"emb_{args.preset or re.sub(r'[^0-9A-Za-z]+', '_', args.model_name.split('/')[-1])}_lr"

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_dataset(cfg).reset_index(drop=True)
    row = pd.Series(np.arange(len(df)), index=df["id"].values)
    if args.exp:
        exps = args.exp.split(",")
    else:
        exps = [e for e in default_experiments(df) if e in ("main", "ablation_surface") or e.startswith("logo_")]

    embs: dict[str, np.ndarray] = {}
    for exp in exps:
        train, val, test, view = experiment_frames(df, exp, cfg["seed"])
        if view not in embs:
            print(f"{spec['model_name']} ({spec['pooling']}) 로 '{view}' 텍스트 임베딩")
            embs[view] = cached_embeddings(df, view, spec, cfg, args, device)
        X = embs[view]
        Xtr, Xva, Xte = (X[row[d["id"]].values] for d in (train, val, test))
        best, best_auc = None, -1.0
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, train["label"])
            a = M.auc(val["label"].values, clf.decision_function(Xva))
            if a > best_auc:
                best, best_auc = clf, a
        val = val.assign(score=best.decision_function(Xva))
        test = test.assign(score=best.decision_function(Xte))
        res = evaluate(val, test, cfg, cfg["seed"])
        save_run(cfg.path("results_dir") / name / exp, test, res,
                 {"model": name, "experiment": exp, "text_view": view, **spec, "C": best.C})
        print(summary_line(name, exp, res))


if __name__ == "__main__":
    main()
