"""Day 2 afternoon: KcELECTRA fine-tuning (GPU recommended, e.g. Colab T4).

Each experiment trains a fresh model, keeps the epoch with the best
validation AUC, and scores the test set with the logit margin (not softmax,
which saturates and makes low-FPR thresholds coarse).

Usage:
    python -m beta.train_electra                  # main + 생성기 hold-out 전부 (기본)
    python -m beta.train_electra --exp main
    python -m beta.train_electra --exp all        # ablation, 프롬프트 hold-out까지
"""
from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .config import base_parser, load_config, set_seed, setup_console
from .evaluate import TEXT_VIEWS, default_experiments, evaluate, experiment_frames, load_dataset, save_run, summary_line

MODEL = "kcelectra"


def make_loader(texts, labels, tok, max_length: int, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    data = list(zip(texts, labels))

    def collate(batch):
        t, y = zip(*batch)
        enc = tok(list(t), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.long)
        return enc

    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=gen)


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(**batch).logits.float()
        out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_one(train: pd.DataFrame, val: pd.DataFrame, cfg, device: torch.device, seed: int):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    e = cfg["electra"]
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(e["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(e["model_name"], num_labels=2).to(device)
    amp = device.type == "cuda"
    tl = make_loader(train["text"].tolist(), train["label"].tolist(), tok, e["max_length"], e["batch_size"], True, seed)
    vl = make_loader(val["text"].tolist(), val["label"].tolist(), tok, e["max_length"], e["batch_size"] * 2, False, seed)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": e["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=float(e["lr"]))
    steps = len(tl) * e["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(steps * e["warmup_ratio"]), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_auc, best_state = -1.0, None
    for epoch in range(e["epochs"]):
        model.train()
        t0, total = time.time(), 0.0
        for batch in tl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        val_auc = M.auc(val["label"].values, predict(model, vl, device, amp))
        print(f"  epoch {epoch + 1}/{e['epochs']}: loss {total / max(len(tl), 1):.4f}, val AUC {val_auc:.4f} ({time.time() - t0:.0f}s)")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tok, amp


def run_experiment(df: pd.DataFrame, exp: str, cfg, device: torch.device, seed: int, save_model: bool) -> dict:
    e = cfg["electra"]
    train, val, test, view = experiment_frames(df, exp, seed)
    tf = TEXT_VIEWS[view]
    train, val, test = (d.assign(text=d["text"].map(tf)) for d in (train, val, test))
    print(f"\n== {MODEL}/{exp}: train {len(train):,} / val {len(val):,} / test {len(test):,}")
    model, tok, amp = train_one(train, val, cfg, device, seed)
    score = lambda d: predict(model, make_loader(d["text"].tolist(), d["label"].tolist(), tok, e["max_length"], e["batch_size"] * 2, False, seed), device, amp)  # noqa: E731
    val = val.assign(score=score(val))
    test = test.assign(score=score(test))
    res = evaluate(val, test, cfg, seed)
    out = cfg.path("results_dir") / MODEL / exp
    save_run(out, test, res, {"model": MODEL, "experiment": exp, "text_view": view, "base_model": e["model_name"],
                              "n_train": len(train), "n_val": len(val), "n_test": len(test)})
    if save_model:
        model.save_pretrained(out / "model")
        tok.save_pretrained(out / "model")
    print(summary_line(MODEL, exp, res))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main(argv: list[str] | None = None) -> None:
    setup_console()
    p = base_parser("KcELECTRA 미세조정")
    p.add_argument("--exp", default="", help="쉼표로 구분한 실험 이름, 'all' 이면 전부 (기본: main + 생성기 hold-out)")
    p.add_argument("--save-model", action="store_true", help="학습한 모델 저장 (main은 서비스용으로 저장 권장)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    df = load_dataset(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[경고] GPU가 없습니다. CPU로는 매우 느립니다. Colab 등 GPU 환경을 권장합니다.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.exp == "all":
        exps = default_experiments(df)
    elif args.exp:
        exps = args.exp.split(",")
    else:
        exps = [x for x in default_experiments(df) if x == "main" or x.startswith("logo_")]
    for exp in exps:
        run_experiment(df, exp, cfg, device, cfg["seed"], args.save_model)


if __name__ == "__main__":
    main()
