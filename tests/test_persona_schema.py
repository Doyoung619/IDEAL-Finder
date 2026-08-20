import pytest

from app.services.persona_service import (
    PERSONA_CATEGORIES,
    PersonaValidationError,
    persona_categories_for_gender,
    validate_persona_profile,
)


def complete_answers():
    return {category.key: ["no_preference"] for category in PERSONA_CATEGORIES}


def test_persona_schema_accepts_complete_answers_and_selected_priorities():
    answers = complete_answers()
    answers["overall_mood"] = ["overall_calm", "intellectual"]
    result = validate_persona_profile("female", answers, ["overall_calm"])

    assert result["overall_mood"] == ["overall_calm", "intellectual"]
    assert result["priorities"] == ["overall_calm"]


@pytest.mark.parametrize(
    ("category", "values"),
    [
        ("overall_mood", ["overall_calm", "intellectual", "natural"]),
        ("hair_style", ["bangs", "no_bangs"]),
        ("eyes", ["double_eyelids", "monolid"]),
        ("eyes", ["large_eyes", "small_eyes"]),
        ("face_shape", ["oval", "no_preference"]),
    ],
)
def test_persona_schema_rejects_limits_conflicts_and_no_preference(category, values):
    answers = complete_answers()
    answers[category] = values
    with pytest.raises(PersonaValidationError):
        validate_persona_profile("female", answers, [])


def test_persona_priority_must_reference_an_actual_selection():
    with pytest.raises(PersonaValidationError):
        validate_persona_profile("male", complete_answers(), ["long_hair"])


def test_male_questionnaire_uses_gender_appropriate_hair_options():
    male = {category.key: category for category in persona_categories_for_gender("male")}
    female = {category.key: category for category in persona_categories_for_gender("female")}

    male_lengths = {option.key for option in male["hair_length"].options}
    male_styles = {option.key for option in male["hair_style"].options}
    female_lengths = {option.key for option in female["hair_length"].options}
    female_styles = {option.key for option in female["hair_style"].options}

    assert "bob_hair" not in male_lengths
    assert "tied_hair" not in male_styles
    assert "bob_hair" in female_lengths
    assert "tied_hair" in female_styles

    answers = complete_answers()
    answers["hair_length"] = ["bob_hair"]
    with pytest.raises(PersonaValidationError):
        validate_persona_profile("male", answers, [])
    validate_persona_profile("female", answers, [])
