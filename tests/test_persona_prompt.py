import pytest

from app.services.persona_service import PERSONA_CATEGORIES, build_persona_prompt_bundle


def test_persona_prompt_bundle_is_deterministic_and_normalizes_category_weights():
    answers = {category.key: ["no_preference"] for category in PERSONA_CATEGORIES}
    answers.update(
        {
            "overall_mood": ["overall_calm", "intellectual"],
            "hair_length": ["long_hair"],
            "hair_style": ["straight_hair", "no_bangs"],
            "expression": ["slight_smile"],
        }
    )
    first = build_persona_prompt_bundle(
        "female", answers, ["overall_calm", "long_hair"]
    )
    second = build_persona_prompt_bundle(
        "female", answers, ["overall_calm", "long_hair"]
    )

    assert first == second
    assert "adult woman aged 20 to 29" in first.base_prompt
    assert "East Asian" not in first.base_prompt
    assert "long hair" in first.full_prompt
    assert "성별" not in first.korean_summary
    assert sum(first.category_weights) == pytest.approx(1.0)
    assert first.category_weights[0] > first.category_weights[-1]
    assert first.raw_profile["fixed_age"] == "20_29"
    assert first.raw_profile["fixed_race"] == "unrestricted"
