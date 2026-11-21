"""Generation prompts.

A  basic        : what a lazy review farm would send.
B  constrained  : casual tone, slang, typos allowed (defined, off by default).
C  style mimic  : few-shot with real reviews from a pool disjoint from the
                  human set, the hardest and most realistic setting.

Conditions (rating, category, target length) are copied from a human sample so
the two classes match in distribution without leaking any human text.
"""
from __future__ import annotations

SENTIMENT = {
    0: "불만족스러웠던 경험 (별점 1~2점 정도의 부정적인 평가)",
    1: "만족스러웠던 경험 (별점 4~5점 정도의 긍정적인 평가)",
    2: "그저 그랬던 경험 (별점 3점 정도, 좋은 점과 아쉬운 점이 섞인 평가)",
}

_TAIL = (
    "한국어로만 쓰고, 중국어·일본어 글자나 영어 문장을 섞지 마세요.\n"
    "가게 이름은 쓰지 마세요.\n"
    "리뷰 본문만 출력하고, 제목·따옴표·설명·인사말은 붙이지 마세요."
)


def _length_rule(n: int) -> str:
    # LLMs overshoot short targets; without a hard cap the short length bins
    # end up nearly empty after length matching.
    if n <= 20:
        return f"공백 포함 {n}자 안팎의 아주 짧은 한 줄. 절대 {n + 10}자를 넘기지 마세요"
    if n <= 50:
        return f"공백 포함 약 {n}자, 한두 문장. {n + 15}자를 넘기지 마세요"
    # the pilot showed long targets coming back at 0.6-0.7x length
    return f"공백 포함 약 {n}자. 너무 짧게 줄이지 말고 최소 {int(n * 0.8)}자 이상 쓰세요"


def _common(category: str, rating: int, target_chars: int) -> str:
    place = f"{category} 음식점" if category else "음식점"
    return (
        f"- 가게 종류: {place}\n"
        f"- 방문 경험: {SENTIMENT[int(rating)]}\n"
        f"- 길이: {_length_rule(int(target_chars))}"
    )


def prompt_a(category: str, rating: int, target_chars: int) -> str:
    return (
        "음식점에 다녀온 손님 입장에서 온라인 리뷰를 하나 작성해 주세요.\n"
        f"{_common(category, rating, target_chars)}\n\n{_TAIL}"
    )


def prompt_b(category: str, rating: int, target_chars: int) -> str:
    return (
        "음식점에 다녀온 손님이 휴대폰으로 대충 남기는 리뷰를 하나 써 주세요.\n"
        f"{_common(category, rating, target_chars)}\n"
        "- 반말·구어체를 섞고, ㅋㅋ/ㅎㅎ/ㅠㅠ나 이모지를 자연스럽게 써도 됩니다.\n"
        "- 맞춤법이나 띄어쓰기가 조금 틀려도 괜찮습니다.\n\n"
        f"{_TAIL}"
    )


def prompt_c(category: str, rating: int, target_chars: int, examples: list[str]) -> str:
    shots = "\n".join(f"- {e}" for e in examples)
    return (
        "아래는 실제 손님들이 음식점 리뷰 사이트에 남긴 리뷰입니다.\n\n"
        f"{shots}\n\n"
        "이 사람들과 같은 사람이 쓴 것처럼 말투, 띄어쓰기와 맞춤법 습관, 문장 길이, "
        "표현 방식을 그대로 따라서 새 리뷰를 하나 써 주세요.\n"
        f"{_common(category, rating, target_chars)}\n"
        "- 예시에 ㅋㅋ, ㅠㅠ, ~, 이모티콘, 맞춤법 실수, 띄어쓰기 습관이 있으면 그런 것도 비슷하게 따라 하세요.\n"
        "- 예시 문장을 베끼지 말고 새로운 내용으로 쓰세요.\n\n"
        f"{_TAIL}"
    )


def build_messages(cond: dict, fewshot_texts: list[str] | None = None) -> list[dict]:
    pt = cond["prompt_type"]
    args = (cond.get("category") or "", cond["rating"], cond["target_chars"])
    if pt == "A":
        user = prompt_a(*args)
    elif pt == "B":
        user = prompt_b(*args)
    elif pt == "C":
        if not fewshot_texts:
            raise ValueError("prompt C needs few-shot examples")
        user = prompt_c(*args, examples=fewshot_texts)
    else:
        raise ValueError(f"unknown prompt type {pt!r}")
    return [{"role": "user", "content": user}]
