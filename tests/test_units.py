import math

import numpy as np
import pandas as pd
import pytest

from beta import metrics as M
from beta.build_dataset import assign_splits
from beta.generate import clean_generation
from beta.normalize import bin_labels, length_bin, normalize, strip_surface, strip_surface_nospace


# ------------------------------------------------------------ normalize

def test_normalize_keeps_jamo_and_emoji():
    t = normalize("맛있어요ㅋㅋ 👍")
    assert t == "맛있어요ㅋㅋ 👍"
    assert "ㅋ" in t  # compatibility jamo ㅋ, not the conjoining form NFKC would produce


def test_normalize_removes_chat_artifacts():
    raw = "**정말 맛있었어요!**\n\n- 직원분이 친절했어요\n- “재방문” 의사 있음…"
    assert normalize(raw) == "정말 맛있었어요! 직원분이 친절했어요 \"재방문\" 의사 있음..."


def test_normalize_idempotent_and_safe():
    s = "  국물이   진함\t\n최고  "
    assert normalize(normalize(s)) == normalize(s) == "국물이 진함 최고"
    assert normalize(None) == "" and normalize(float("nan")) == ""


def test_strip_surface_views():
    s = "존맛ㅋㅋ!! 또 갈듯~ 😋 10점"
    assert strip_surface(s) == "존맛 또 갈듯 10점"
    assert strip_surface_nospace(s) == "존맛또갈듯10점"


def test_length_bins():
    edges = [20, 50, 150]
    assert bin_labels(edges) == ["<=20", "21-50", "51-150", ">150"]
    assert [length_bin(n, edges) for n in (5, 20, 21, 50, 51, 150, 151)] == [
        "<=20", "<=20", "21-50", "21-50", "51-150", "51-150", ">150"]


# ------------------------------------------------------------ generation cleaning

@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"국물이 진하고 맛있어요."', "국물이 진하고 맛있어요."),
        ("리뷰: 국물이 진하고 맛있어요.", "국물이 진하고 맛있어요."),
        ("물론입니다! 리뷰를 작성해 드릴게요:\n국물이 진하고 맛있어요.", "국물이 진하고 맛있어요."),
        ("직원분이 죄송하다고 하셨지만 음식은 늦게 나왔어요.", "직원분이 죄송하다고 하셨지만 음식은 늦게 나왔어요."),
        # special tokens leaking from GGUF conversions
        ("딤섬 만두 모두 맛있고 분위기 좋음<|eot_id|>", "딤섬 만두 모두 맛있고 분위기 좋음"),
        ("맛있어요<|eot_id|>\nassistant\n또 올게요", "맛있어요"),
        ("국물이 진해요[|endofturn|]", "국물이 진해요"),
        ("가성비 좋아요<|im_end|>", "가성비 좋아요"),
    ],
)
def test_clean_generation_accepts(raw, expected):
    text, reason = clean_generation(raw)
    assert reason is None and text == expected


@pytest.mark.parametrize(
    "raw,reason",
    [
        ("죄송하지만 그 요청은 도와드릴 수 없습니다.", "refusal"),
        ("[가게 이름]에서 먹은 짬뽕이 맛있었어요.", "placeholder"),
        ("OOO 식당 최고", "placeholder"),
        ("", "empty"),
        (None, "empty"),
    ],
)
def test_clean_generation_rejects(raw, reason):
    assert clean_generation(raw)[1] == reason


# ------------------------------------------------------------ metrics

def test_threshold_for_fpr_respects_target():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
    s = np.r_[rng.normal(0, 1, 1000), rng.normal(2, 1, 1000)]
    for target in (0.01, 0.05, 0.2):
        thr = M.threshold_for_fpr(y, s, target)
        assert M.at_threshold(y, s, thr)["fpr"] <= target


def test_tpr_at_fpr_and_auc_perfect():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert M.auc(y, s) == 1.0
    assert M.tpr_at_fpr(y, s, 0.0) == 1.0


def test_precision_at_prevalence_matches_hand_calc():
    # TPR 90%, FPR 5%, 5% of reviews are LLM -> about half of the flags are wrong
    assert M.precision_at_prevalence(0.9, 0.05, 0.05) == pytest.approx(0.486, abs=1e-3)


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)
    s = np.r_[rng.normal(0, 1, 300), rng.normal(1, 1, 300)]
    lo, hi = M.bootstrap_ci(M.auc, y, s, n=200, seed=0)
    assert lo <= M.auc(y, s) <= hi


# ------------------------------------------------------------ split

def test_assign_splits_groups_and_ratios():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(
        {
            "group_id": [f"g{i // 2}" for i in range(n)],  # pairs share a group
            "label": rng.integers(0, 2, n),
            "length_bin": rng.choice(["<=20", "21-50"], n),
            "source": rng.choice(["human", "gpt"], n),
            "prompt_type": rng.choice(["A", "C"], n),
        }
    )
    split = assign_splits(df, [0.7, 0.15, 0.15], np.random.default_rng(0))
    assert split.notna().all()
    assert (df.assign(split=split).groupby("group_id")["split"].nunique() == 1).all()
    frac = split.value_counts(normalize=True)
    assert math.isclose(frac["train"], 0.7, abs_tol=0.05)
    assert math.isclose(frac["test"], 0.15, abs_tol=0.05)
