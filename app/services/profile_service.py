from __future__ import annotations

from collections import OrderedDict


PROFILE_CATEGORIES = OrderedDict(
    [
        (
            "전반적 스타일/무드",
            ["청순한", "매력적인", "귀여운", "차분한", "지적인", "시크한", "밝은", "자연스러운"],
        ),
        ("얼굴형", ["계란형", "둥근형", "긴형", "각진형", "하트형", "V라인"]),
        ("머리 길이", ["숏컷", "단발", "중단발", "긴머리"]),
        ("머리 스타일", ["생머리", "웨이브", "앞머리 있음", "앞머리 없음", "묶은 머리"]),
        ("눈", ["큰 눈", "작은 눈", "가는 눈", "쌍꺼풀", "또렷한 눈매", "무쌍"]),
        ("코/입", ["높은 코", "작은 코", "도톰한 입술", "입꼬리 올라감", "미소가 예쁨"]),
        ("인상", ["순한 인상", "차가운 인상", "성숙한 인상", "어려 보이는 인상", "카리스마", "귀여운 인상"]),
        ("분위기/라이프스타일", ["운동을 좋아함", "문화생활을 좋아함", "여행을 좋아함", "패션 관심이 많음", "차분한 취미", "활발한 성격"]),
    ]
)


ENGLISH_TERMS = {
    "청순한": "innocent and natural",
    "매력적인": "attractive",
    "귀여운": "cute but adult",
    "차분한": "calm",
    "지적인": "intellectual",
    "시크한": "chic",
    "밝은": "bright and friendly",
    "자연스러운": "natural",
    "계란형": "oval face",
    "둥근형": "round face",
    "긴형": "long face",
    "각진형": "angular face",
    "하트형": "heart-shaped face",
    "V라인": "V-shaped jawline",
    "숏컷": "short hair",
    "단발": "bob haircut",
    "중단발": "medium-length hair",
    "긴머리": "long hair",
    "생머리": "straight hair",
    "웨이브": "wavy hair",
    "앞머리 있음": "with bangs",
    "앞머리 없음": "without bangs",
    "묶은 머리": "tied-back hair",
    "큰 눈": "large eyes",
    "작은 눈": "small eyes",
    "가는 눈": "narrow eyes",
    "쌍꺼풀": "double eyelids",
    "또렷한 눈매": "clear defined eyes",
    "무쌍": "monolid eyes",
    "높은 코": "high nose bridge",
    "작은 코": "small nose",
    "도톰한 입술": "full lips",
    "입꼬리 올라감": "upturned mouth corners",
    "미소가 예쁨": "beautiful smile",
    "순한 인상": "gentle impression",
    "차가운 인상": "cool impression",
    "성숙한 인상": "mature impression",
    "어려 보이는 인상": "youthful adult impression",
    "카리스마": "charismatic",
    "귀여운 인상": "charming adult impression",
    "운동을 좋아함": "sporty lifestyle",
    "문화생활을 좋아함": "enjoys arts and culture",
    "여행을 좋아함": "enjoys travel",
    "패션 관심이 많음": "fashion-conscious",
    "차분한 취미": "calm hobbies",
    "활발한 성격": "outgoing personality",
}


def build_profile_prompt(target_gender: str | None, selected: list[str]) -> str:
    subject = {
        "male": "adult man",
        "female": "adult woman",
    }.get(target_gender, "adult person")
    attributes = [ENGLISH_TERMS.get(option, option) for option in selected]
    suffix = f", {', '.join(attributes)}" if attributes else ""
    return (
        f"high quality studio portrait photograph of an {subject}, "
        f"front-facing, realistic human face{suffix}"
    )

