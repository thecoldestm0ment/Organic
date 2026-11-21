"""Day 2 evening (optional): zero-shot detection with a small open LM.

Two scores, no training:
- fastdetect : Fast-DetectGPT analytic sampling discrepancy (Bao et al., ICLR 2024),
               same model for sampling and scoring.
               (log p(x) - E[log p]) / sqrt(Var[log p]) summed over tokens.
- loglik     : mean token log-likelihood (the classic perplexity baseline).
Higher = more LLM-like. Thresholds still come from the validation split.

No generator is ever "seen" here, so per-generator results in `main` are the
zero-shot counterpart of the supervised generator hold-out.

Usage:
    python -m beta.zeroshot
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .config import base_parser, load_config, set_seed, setup_console
from .evaluate import evaluate, load_dataset, save_run, summary_line


@torch.no_grad()
def score_batch(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()
    lprobs = torch.log_softmax(logits, dim=-1)
    probs = lprobs.exp()
    ll = lprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mean_ref = (probs * lprobs).sum(-1)
    var_ref = (probs * lprobs.pow(2)).sum(-1) - mean_ref.pow(2)
    num = ((ll - mean_ref) * mask).sum(-1)
    den = (var_ref * mask).sum(-1).clamp_min(1e-8).sqrt()
    n_tok = mask.sum(-1).clamp_min(1.0)
    return (num / den).cpu().numpy(), ((ll * mask).sum(-1) / n_tok).cpu().numpy()


def score_texts(texts: list[str], model, tok, max_length: int, batch_size: int, device: torch.device):
    order = np.argsort([len(t) for t in texts])  # similar lengths per batch -> less padding
    fast = np.zeros(len(texts))
    loglik = np.zeros(len(texts))
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        f, l = score_batch(model, enc["input_ids"], enc["attention_mask"])
        fast[idx], loglik[idx] = f, l
        done = start + len(idx)
        if done % (batch_size * 50) < batch_size or done == len(texts):
            print(f"  {done:,}/{len(texts):,} ({time.time() - t0:.0f}s)", flush=True)
    return fast, loglik


def load_lm(name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok


def main(argv: list[str] | None = None) -> None:
    setup_console()
    p = base_parser("zero-shot 탐지 (Fast-DetectGPT, log-likelihood)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    z = cfg["zeroshot"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[경고] GPU가 없습니다. 1B급 모델은 CPU에서 매우 느립니다.")

    df = load_dataset(cfg)
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    model, tok = load_lm(z["model_name"], device)
    print(f"채점 모델: {z['model_name']} ({device})")
    both = pd.concat([val, test])
    fast, loglik = score_texts(both["text"].tolist(), model, tok, z["max_length"], z["batch_size"], device)

    n_val = len(val)
    for method, scores in (("fastdetect", fast), ("loglik", loglik)):
        name = f"zeroshot_{method}"
        v = val.assign(score=scores[:n_val])
        t = test.assign(score=scores[n_val:])
        res = evaluate(v, t, cfg, cfg["seed"])
        save_run(cfg.path("results_dir") / name / "main", t, res,
                 {"model": name, "experiment": "main", "scoring_model": z["model_name"], "n_val": len(v), "n_test": len(t)})
        print(summary_line(name, "main", res))


if __name__ == "__main__":
    main()
