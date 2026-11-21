"""Day 1: generate LLM reviews through OpenAI-compatible APIs.

Usage:
    python -m beta.generate --pilot 20     # 생성기x프롬프트마다 20개 시험 생성 후 출력 (눈으로 확인)
    python -m beta.generate                # 전체 생성. 중단돼도 다시 실행하면 이어서 함
    python -m beta.generate --only gpt     # 특정 생성기만
    python -m beta.generate --mock         # API 없이 가짜 생성기로 파이프라인 점검
    python -m beta.generate --replan       # 생성 조건 다시 뽑기 (생성기 목록을 바꿨을 때)

Every generator is called through the OpenAI SDK with a custom base_url, which
covers OpenAI, Gemini's OpenAI-compatible endpoint, OpenRouter/Together, and a
local vLLM server.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import urllib.request
import zlib
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .config import base_parser, load_config, load_env, set_seed, setup_console
from .normalize import length_bin
from .prompts import build_messages

# ---------------------------------------------------------------- cleaning

_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
_PREAMBLE_LINE = re.compile(r"^.{0,60}(리뷰|작성|물론|다음은|여기).{0,30}[:：]\s*$")
_LABEL_PREFIX = re.compile(r"^\s*\[?리뷰\]?\s*\d*\s*[:：]\s*")
_PLACEHOLDER = re.compile(
    r"\[[^\]]{0,20}(이름|가게|식당|상호|메뉴|지역)[^\]]{0,20}\]"
    r"|OOO|○○|◯◯|XX식당|XX집|\{[^}]*\}"
)
_REFUSAL = re.compile(r"^(죄송|저는 AI|AI로서|인공지능으로서)|언어 ?모델로서|AI 언어 ?모델")
_WRAPPERS = [('"', '"'), ("“", "”"), ("'", "'"), ("「", "」"), ("『", "』")]


def clean_generation(raw: str | None) -> tuple[str, str | None]:
    """Strip chat-style wrapping. Returns (text, reject_reason or None)."""
    if not raw or not raw.strip():
        return "", "empty"
    t = _THINK.sub("", raw).strip()  # reasoning models may prepend <think>...</think>
    if not t:
        return "", "empty"
    lines = t.splitlines()
    if len(lines) > 1 and _PREAMBLE_LINE.match(lines[0].strip()):
        t = "\n".join(lines[1:]).strip()
    t = _LABEL_PREFIX.sub("", t, count=1).strip()
    for left, right in _WRAPPERS:
        if len(t) > 2 and t.startswith(left) and t.endswith(right):
            t = t[1:-1].strip()
    if _REFUSAL.search(t):
        return t, "refusal"
    if _PLACEHOLDER.search(t):
        return t, "placeholder"
    if len(t) < 3:
        return t, "too_short"
    return t, None


# ---------------------------------------------------------------- planning


def plan_conditions(cfg: dict, human: pd.DataFrame, pool: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    g = cfg["generation"]
    gens = g["generators"]
    prompt_types = g["prompt_types"]
    edges = cfg["human"]["length_edges"]
    n = math.ceil(len(human) * g["oversample"])
    src = human.iloc[rng.choice(len(human), size=n, replace=n > len(human))].reset_index(drop=True)
    cells = list(product(range(len(gens)), prompt_types))
    order = rng.permutation(n)

    by_rating_bin: dict[tuple[int, str], list[str]] = defaultdict(list)
    by_rating: dict[int, list[str]] = defaultdict(list)
    for r in pool.itertuples():
        by_rating_bin[(int(r.rating), r.length_bin)].append(r.id)
        by_rating[int(r.rating)].append(r.id)
    all_ids = pool["id"].tolist()
    k = g["fewshot_k"]

    def pick_fewshot(rating: int, lbin: str) -> str:
        for cand in (by_rating_bin[(rating, lbin)], by_rating[rating], all_ids):
            if len(cand) >= k:
                return ",".join(rng.choice(cand, size=k, replace=False))
        return ",".join(all_ids[:k])

    rows = []
    for i, r in enumerate(src.itertuples()):
        gi, pt = cells[order[i] % len(cells)]
        gen = gens[gi]
        target = int(r.length)
        rows.append(
            {
                "cond_id": f"g{i:06d}",
                "generator": gen["name"],
                "family": gen["family"],
                "prompt_type": pt,
                "rating": int(r.rating),
                "category": r.category,
                "target_chars": target,
                "target_bin": length_bin(target, edges),
                "temperature": float(rng.choice(g["temperatures"])),
                "fewshot_ids": pick_fewshot(int(r.rating), r.length_bin) if pt == "C" else "",
                "source_human_id": r.id,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- mock

_MOCK = {
    0: ["음식이 너무 짜서 먹기 힘들었어요.", "직원분 응대가 아쉬웠습니다.", "가격에 비해 양이 적네요.", "다시 방문할 것 같지는 않아요."],
    1: ["음식이 정말 맛있었어요.", "직원분들이 친절하셨습니다.", "가격 대비 양이 푸짐해요.", "재방문 의사 있습니다."],
    2: ["맛은 무난한 편이었어요.", "웨이팅이 조금 있었습니다.", "분위기는 괜찮았지만 가격이 아쉬워요.", "한 번쯤 가볼 만해요."],
}
_MOCK_FLAVOR = {"qwen": "또한 ", "exaone": "전반적으로 ", "kanana": "솔직히 말하면 ", "gpt": "무엇보다 ", "gemini": "개인적으로 "}


def mock_generate(cond: dict) -> str:
    """Deterministic fake LLM output with typical chat artifacts, for dry runs."""
    rng = np.random.default_rng(zlib.crc32(cond["cond_id"].encode()))
    phrases = _MOCK[int(cond["rating"])]
    flavor = _MOCK_FLAVOR.get(cond["generator"], "")
    parts: list[str] = []
    while sum(len(p) + 1 for p in parts) < max(10, int(cond["target_chars"]) * 0.9):
        parts.append(str(rng.choice(phrases)))
    text = flavor + " ".join(parts)
    if cond["prompt_type"] == "C" and rng.random() < 0.5:
        text += " ㅎㅎ"
    roll = rng.random()
    if roll < 0.15:
        text = f"**{text}**"
    elif roll < 0.25:
        text = "리뷰: " + text
    elif roll < 0.30:
        text = "[가게 이름]에 다녀왔어요. " + text
    return text


# ---------------------------------------------------------------- running


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    done.add(rec["cond_id"])
    return done


async def _call_api(client, gen: dict, messages: list[dict], temperature: float) -> str:
    kwargs: dict = {"model": gen["model"], "messages": messages}
    if gen.get("supports_temperature", True):
        kwargs["temperature"] = temperature
    if gen.get("extra_body"):
        kwargs["extra_body"] = gen["extra_body"]
    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


async def run(cfg: dict, conds: pd.DataFrame, pool_text: dict[str, str], out_path: Path, mock: bool) -> Counter:
    g = cfg["generation"]
    gens = {x["name"]: x for x in g["generators"]}
    clients, sems = {}, {}
    for name in conds["generator"].unique():
        gen = gens[name]
        sems[name] = asyncio.Semaphore(int(gen.get("concurrency", 4)))
        if not mock:
            from openai import AsyncOpenAI

            clients[name] = AsyncOpenAI(
                base_url=gen["base_url"],
                # local vLLM/Ollama servers accept any key
                api_key=os.environ[gen["api_key_env"]] if gen.get("api_key_env") else "EMPTY",
                max_retries=g["max_retries"],
                timeout=g["request_timeout"],
            )

    stats: Counter = Counter()
    consecutive_errors: Counter = Counter()
    disabled: set[str] = set()
    total = len(conds)
    finished = 0
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as f:

        async def one(cond: dict) -> None:
            nonlocal finished
            name = cond["generator"]
            async with sems[name]:
                if name in disabled:
                    return
                fewshot = [pool_text[i] for i in cond["fewshot_ids"].split(",") if i] or None
                messages = build_messages(cond, fewshot)
                try:
                    if mock:
                        raw = mock_generate(cond)
                    else:
                        raw = await _call_api(clients[name], gens[name], messages, cond["temperature"])
                    text, reason = clean_generation(raw)
                    status = "ok" if reason is None else "rejected"
                    consecutive_errors[name] = 0
                except Exception as e:  # noqa: BLE001 - record and keep going
                    raw, text, status = None, "", "error"
                    reason = f"{type(e).__name__}: {e}"[:300]
                    consecutive_errors[name] += 1
                    if consecutive_errors[name] >= 10 and name not in disabled:
                        disabled.add(name)
                        print(f"\n[중단] {name}: 연속 10회 오류. 서버가 떠 있는지, 키/모델 이름/요금 한도를 확인하세요. 마지막 오류: {reason}")
                rec = {
                    **{k: cond[k] for k in ("cond_id", "generator", "family", "prompt_type", "temperature")},
                    "model": "mock" if mock else gens[name]["model"],
                    "status": status,
                    "reason": reason,
                    "raw_text": raw,
                    "text": text,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                stats[(name, status)] += 1
                finished += 1
                if finished % 100 == 0 or finished == total:
                    rate = finished / max(time.time() - t0, 1e-6)
                    print(f"  {finished:,}/{total:,} ({rate:.1f}/s)", flush=True)

        await asyncio.gather(*(one(c) for c in conds.to_dict("records")))
    return stats


def check_local_server(gen: dict) -> None:
    """For a localhost endpoint, make sure the server is up AND serving this
    generator's model, so outputs are never labelled with the wrong family."""
    if urlparse(gen["base_url"]).hostname not in ("localhost", "127.0.0.1", "0.0.0.0"):
        return
    url = gen["base_url"].rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            served = [m["id"] for m in json.load(r)["data"]]
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"{gen['name']}: 로컬 서버({url})에 연결할 수 없습니다 ({type(e).__name__}). "
            "`bash scripts/generate_local.sh` 로 실행하거나 vLLM 서버를 먼저 띄우세요."
        )
    if gen["model"] not in served:
        raise SystemExit(
            f"{gen['name']}: 지금 서버에 떠 있는 모델은 {served} 인데 config의 model은 {gen['model']} 입니다. "
            "--only 로 떠 있는 모델의 생성기만 실행하세요."
        )


def print_pilot(path: Path, conds: pd.DataFrame) -> None:
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    target = conds.set_index("cond_id")["target_chars"].to_dict()
    for rec in sorted(recs, key=lambda r: (r["generator"], r["prompt_type"])):
        t = rec["text"] or ""
        print(
            f"[{rec['generator']}/{rec['prompt_type']}] {rec['status']:<8} "
            f"{len(t):>4}자 (목표 {target[rec['cond_id']]}) | {t[:160]}"
            + (f"  <- {rec['reason']}" if rec["reason"] else "")
        )


def main(argv: list[str] | None = None) -> None:
    setup_console()
    p = base_parser("LLM 리뷰 생성")
    p.add_argument("--mock", action="store_true", help="API 대신 가짜 생성기 사용")
    p.add_argument("--pilot", type=int, default=0, help="생성기x프롬프트마다 N개만 시험 생성")
    p.add_argument("--only", default="", help="쉼표로 구분한 생성기 이름만 실행")
    p.add_argument("--replan", action="store_true", help="생성 조건 새로 뽑기")
    p.add_argument("--plan-only", action="store_true", help="생성 조건만 만들고 종료")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    load_env()
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    interim = cfg.path("interim_dir")
    human = pd.read_parquet(interim / "human.parquet")
    pool = pd.read_parquet(interim / "fewshot_pool.parquet")
    pool_text = dict(zip(pool["id"], pool["text"]))

    plan_path = interim / "conditions.parquet"
    if args.replan or not plan_path.exists():
        old = interim / "generated.jsonl"
        if old.exists():
            # cond_ids are reused by the new plan, so old outputs would attach to the wrong conditions
            backup = old.with_name(f"generated.{time.strftime('%Y%m%d-%H%M%S')}.jsonl.bak")
            old.rename(backup)
            print(f"조건을 새로 뽑으므로 기존 생성 결과를 옮겨 둡니다: {backup.name}")
        conds = plan_conditions(cfg, human, pool, rng)
        conds.to_parquet(plan_path, index=False)
        print(f"생성 조건 {len(conds):,}개 -> {plan_path}")
    conds = pd.read_parquet(plan_path)
    if args.plan_only:
        print(pd.crosstab(conds["generator"], conds["prompt_type"]).to_string())
        return
    known = {x["name"] for x in cfg["generation"]["generators"]}
    unknown = set(conds["generator"]) - known
    if unknown:
        raise SystemExit(f"조건 파일에 config에 없는 생성기 {unknown}가 있습니다. --replan 으로 다시 뽑으세요.")
    if args.only:
        conds = conds[conds["generator"].isin(args.only.split(","))]

    if not args.mock:
        active = [x for x in cfg["generation"]["generators"] if x["name"] in set(conds["generator"])]
        missing = sorted({x["api_key_env"] for x in active if x.get("api_key_env") and not os.environ.get(x["api_key_env"])})
        if missing:
            raise SystemExit(f"API 키가 없습니다: {missing}. .env 를 채우거나 --only / --mock 을 쓰세요.")
        for gen in active:
            check_local_server(gen)

    if args.pilot:
        conds = conds.groupby(["generator", "prompt_type"], sort=False).head(args.pilot)
        out = interim / f"generated_pilot_{'_'.join(sorted(conds['generator'].unique()))}.jsonl"
        out.unlink(missing_ok=True)
        asyncio.run(run(cfg, conds, pool_text, out, args.mock))
        print_pilot(out, conds)
        return

    out = interim / "generated.jsonl"
    done = load_done(out)
    todo = conds[~conds["cond_id"].isin(done)]
    print(f"생성 대상 {len(todo):,}개 (이미 완료 {len(done):,}개)")
    stats = asyncio.run(run(cfg, todo, pool_text, out, args.mock))
    print("\n결과:")
    for (name, status), n in sorted(stats.items()):
        print(f"  {name:<10} {status:<9} {n:,}")
    reasons = Counter()
    with open(out, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["status"] != "ok":
                reasons[(rec["generator"], (rec["reason"] or "").split(":")[0])] += 1
    if reasons:
        print("실패/거절 사유 (누적):")
        for (name, reason), n in reasons.most_common(15):
            print(f"  {name:<10} {reason:<20} {n:,}")
        print("거절·오류 건은 다시 실행하면 재시도합니다.")


if __name__ == "__main__":
    main()
