"""Synthetic KR3-like data and a throwaway project dir, so the whole pipeline
can run offline in seconds (mock generator, tiny random models)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

HUMAN_PHRASES = [
    "존맛 ㅋㅋ", "또 갈듯", "웨이팅 넘 길었음ㅠ", "사장님 친절하심", "가성비 굿",
    "국물이 진짜 진함", "양 많아요👍", "주차는 좀 불편", "맛 집 인정", "재방문 각",
    "고기 질 좋음", "반찬 리필 됨", "좀 짰어요ㅠㅠ", "분위기 짱", "가격은 쫌 있음",
    "점심에 줄 섬", "김치 맛있음", "서비스로 계란찜 주심", "면이 쫄깃", "다음엔 포장해야지~",
]
CATEGORIES = ["한식", "중식", "일식", "고기", "카페"]


def make_kr3_tsv(path: Path, n: int = 3000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        k = int(rng.choice([1, 2, 3, 5, 8, 12, 20, 30]))  # covers every length bin
        text = " ".join(rng.choice(HUMAN_PHRASES, size=k))
        if rng.random() < 0.1:
            text = text.replace(" ", "\n", 1)  # raw data still has line breaks
        rows.append({"Rating": str(rng.choice(["0", "1", "1", "1", "2"])), "Review": text,
                     "Region": "서울", "Category": str(rng.choice(CATEGORIES))})
    rows += rows[:50]  # exact duplicates
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def make_tiny_models(root: Path, texts: list[str]) -> tuple[Path, Path]:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, GPT2Config, GPT2LMHeadModel

    chars = sorted({c for t in texts for c in t if not c.isspace()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + [f"##{c}" for c in chars]
    vocab_file = root / "vocab.txt"
    vocab_file.write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False, strip_accents=False)

    cls_dir, lm_dir = root / "tiny-cls", root / "tiny-lm"
    BertForSequenceClassification(
        BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                   intermediate_size=64, max_position_embeddings=128, num_labels=2)
    ).save_pretrained(cls_dir)
    tok.save_pretrained(cls_dir)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_embd=32, n_layer=1, n_head=2, n_positions=128)).save_pretrained(lm_dir)
    tok.save_pretrained(lm_dir)
    return cls_dir, lm_dir


@pytest.fixture(scope="session")
def project(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("proj")
    tsv = make_kr3_tsv(root / "kr3_raw.tsv")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["kr3"]["expected_rows"] = 3050
    cfg["human"].update(n_samples=400, fewshot_pool_size=100)
    cfg["eval"]["n_bootstrap"] = 30
    cfg["tfidf"]["C_grid"] = [1.0]
    texts = pd.read_csv(tsv, sep="\t")["Review"].tolist() + [
        "전반적으로 솔직히 말하면 또한 음식이 정말 맛있었어요. 직원분들이 친절하셨습니다. 가격 대비 양이 푸짐해요. "
        "재방문 의사 있습니다. 맛은 무난한 편이었어요. 웨이팅이 조금 있었습니다. 분위기는 괜찮았지만 가격이 아쉬워요. "
        "한 번쯤 가볼 만해요. 너무 짜서 먹기 힘들었어요. 응대가 아쉬웠습니다. 적네요. 다시 방문할 것 같지는 않아요. ㅎ"
    ]
    cls_dir, lm_dir = make_tiny_models(root, texts)
    cfg["electra"].update(model_name=str(cls_dir), max_length=64, batch_size=16, epochs=1)
    cfg["zeroshot"].update(model_name=str(lm_dir), max_length=64, batch_size=16)
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return {"root": root, "tsv": tsv, "config": str(cfg_path), "cfg": cfg}
