import json

import pytest

from powerlifting_coach.llm_feedback import LocalModelConfig, generate_local_feedback


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def create_chat_completion(self, **request):
        self.calls += 1
        content = self.response
        if not isinstance(content, str):
            content = json.dumps(content)
        return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def analysis():
    return {
        "quality": {"analysis_valid": True, "confidence": "medium"},
        "reps_detected": 1,
        "rep": {
            "depth_assessment": "likely_sufficient_depth",
            "findings": [
                {
                    "priority": 7,
                    "name": "hip_first_ascent",
                    "status": "observed",
                    "confidence": "medium",
                    "evidence": "Hips rose before shoulders early in ascent.",
                },
                {
                    "priority": 12,
                    "name": "depth",
                    "status": "likely_sufficient_depth",
                    "confidence": "medium",
                    "evidence": "The 2D depth estimate was likely sufficient.",
                },
            ],
            "warnings": [],
        },
        "warnings": [],
    }


@pytest.fixture(autouse=True)
def model_ready(monkeypatch):
    monkeypatch.setattr(
        "powerlifting_coach.llm_feedback.check_local_model_status",
        lambda config: {
            "reachable": True,
            "configured_model_installed": True,
            "error": None,
        },
    )


def test_valid_local_feedback(monkeypatch, analysis):
    model = FakeModel(
        {
            "overall_assessment": "Your depth estimate was likely sufficient, with one ascent point to work on.",
            "cue_explanations": [
                {
                    "theme_id": "THEME_HIP_FIRST_ASCENT",
                    "what_was_observed": "Your hips rose before your shoulders at the start of the ascent.",
                    "why_focus_on_this": "Moving them together can make the ascent more coordinated and easier to finish smoothly.",
                }
            ],
        }
    )
    monkeypatch.setattr(
        "powerlifting_coach.llm_feedback._load_local_model", lambda config: model
    )

    result = generate_local_feedback(analysis, LocalModelConfig(model_path="unused.gguf"))

    assert model.calls == 1
    assert result["source"] == "local_model"
    assert result["feedback"]["priority_cues"][0]["rank"] == 1


def test_invalid_analysis_uses_fallback_without_loading_model(monkeypatch, analysis):
    analysis["quality"]["analysis_valid"] = False
    loaded = []
    monkeypatch.setattr(
        "powerlifting_coach.llm_feedback._load_local_model",
        lambda config: loaded.append(config),
    )

    result = generate_local_feedback(analysis)

    assert loaded == []
    assert result["source"] == "fallback"
    assert result["fallback_reason_code"] == "analysis_invalid"


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (
            {
                "reachable": False,
                "configured_model_installed": False,
                "error": "model_missing",
            },
            "local_model_missing",
        ),
        (
            {
                "reachable": False,
                "configured_model_installed": True,
                "error": "unavailable",
            },
            "local_model_unavailable",
        ),
    ],
)
def test_missing_or_unavailable_model_uses_fallback(
    monkeypatch, analysis, status, reason
):
    monkeypatch.setattr(
        "powerlifting_coach.llm_feedback.check_local_model_status",
        lambda config: status,
    )

    result = generate_local_feedback(analysis)

    assert result["source"] == "fallback"
    assert result["fallback_reason_code"] == reason


def test_malformed_json_uses_fallback(monkeypatch, analysis):
    model = FakeModel("not json")
    monkeypatch.setattr(
        "powerlifting_coach.llm_feedback._load_local_model", lambda config: model
    )

    result = generate_local_feedback(analysis)

    assert model.calls == 1
    assert result["source"] == "fallback"
    assert result["fallback_reason_code"] == "local_model_invalid_json"
