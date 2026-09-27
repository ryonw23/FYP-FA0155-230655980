
import pytest

from powerlifting_coach.llm_feedback import (
    build_template_feedback,
    _merge_llm_prose_with_brief,
)
from powerlifting_coach.coaching_rule_library import build_actionable_coaching_brief


def analysis(coordination, finish, measurement=None):
    return {
        "lift": "deadlift",
        "analysis_status": "valid",
        "quality": {"confidence": "high"},
        "metric_results": [
            {
                "metric_id": "deadlift_early_coordination_2d",
                "classification": coordination,
                "reliability": {"confidence": "high"},
            },
            {
                "metric_id": "deadlift_finish_geometry_2d",
                "classification": finish,
                "measurement": measurement
                or {
                    "projected_knee_angle_degrees": 172,
                    "projected_trunk_thigh_angle_degrees": 167,
                    "torso_inclination_from_vertical_degrees": 5,
                },
                "threshold": {
                    "knee_angle_degrees": 165,
                    "trunk_thigh_angle_degrees": 160,
                    "torso_inclination_degrees": 12,
                    "uncertainty_degrees": {"knee": 5, "trunk_thigh": 5, "torso": 5},
                },
                "reliability": {"confidence": "high"},
            },
        ],
    }


@pytest.mark.parametrize(
    ("coordination", "finish", "priorities", "positives", "no_issue"),
    [
        ("EARLY_HIP_RISE_NOT_DETECTED", "COMPLETE_FINISH", 0, 3, True),
        ("EARLY_HIP_RISE_DETECTED", "COMPLETE_FINISH", 1, 2, False),
        ("EARLY_HIP_RISE_NOT_DETECTED", "INCOMPLETE_FINISH", 1, 1, False),
        ("EARLY_HIP_RISE_DETECTED", "INCOMPLETE_FINISH", 2, 0, False),
        ("UNCERTAIN", "COMPLETE_FINISH", 0, 2, False),
        ("EARLY_HIP_RISE_DETECTED", "INSUFFICIENT_EVIDENCE", 1, 0, False),
    ],
)
def test_deadlift_feedback_result_combinations(
    coordination, finish, priorities, positives, no_issue
):
    feedback = build_template_feedback(analysis(coordination, finish))
    assert len(feedback["priority_cues"]) == priorities
    assert len(feedback["positive_observations"]) == positives
    assert ("No issue covered" in feedback["overall_assessment"]) is no_issue
    if coordination == "UNCERTAIN":
        assert any(
            "Early hip–shoulder coordination" in item
            for item in feedback["limitations"]
        )
    if finish == "INSUFFICIENT_EVIDENCE":
        assert any("Finish-position evidence" in item for item in feedback["limitations"])


@pytest.mark.parametrize(
    ("measurement", "expected"),
    [
        (
            {
                "projected_knee_angle_degrees": 150,
                "projected_trunk_thigh_angle_degrees": 167,
                "torso_inclination_from_vertical_degrees": 5,
            },
            "terminal knee angle measured 150° (below the provisional 160° boundary)",
        ),
        (
            {
                "projected_knee_angle_degrees": 172,
                "projected_trunk_thigh_angle_degrees": 145,
                "torso_inclination_from_vertical_degrees": 5,
            },
            "trunk–thigh extension measured 145° (below the provisional 155° boundary)",
        ),
        (
            {
                "projected_knee_angle_degrees": 172,
                "projected_trunk_thigh_angle_degrees": 167,
                "torso_inclination_from_vertical_degrees": 20,
            },
            "torso inclination measured 20° (above the provisional 17° boundary)",
        ),
    ],
)
def test_incomplete_finish_names_failed_component_and_boundary(measurement, expected):
    feedback = build_template_feedback(
        analysis("EARLY_HIP_RISE_NOT_DETECTED", "INCOMPLETE_FINISH", measurement)
    )
    assert expected in feedback["priority_cues"][0]["what_was_observed"]


def test_incomplete_finish_names_specific_combination_and_local_model_cannot_replace_it():
    measurement = {
        "projected_knee_angle_degrees": 150,
        "projected_trunk_thigh_angle_degrees": 145,
        "torso_inclination_from_vertical_degrees": 20,
    }
    brief = build_actionable_coaching_brief(
        analysis("EARLY_HIP_RISE_DETECTED", "INCOMPLETE_FINISH", measurement)
    )
    deterministic = brief["priority_themes"][1]["what_was_observed"]
    assert "terminal knee angle" in deterministic
    assert "trunk–thigh extension" in deterministic
    assert "torso inclination" in deterministic
    merged = _merge_llm_prose_with_brief(
        "Two priorities were detected.",
        {
            "DL-003-FINISH-GEOMETRY": {
                "what_was_observed": "The model generalized this.",
                "why_focus_on_this": "A sufficiently long rewritten explanation.",
            }
        },
        brief,
    )
    assert merged["priority_cues"][1]["what_was_observed"] == deterministic
    assert (
        merged["priority_cues"][1]["next_rep_cue"]
        == brief["priority_themes"][1]["next_rep_cue"]
    )


def test_incomplete_finish_wording_and_cue_refer_to_upright_top_position():
    brief = build_actionable_coaching_brief(
        analysis("EARLY_HIP_RISE_NOT_DETECTED", "INCOMPLETE_FINISH")
    )
    theme = brief["priority_themes"][0]

    assert theme["why_focus_on_this"] == (
        "The upright position did not reach the implemented extension range."
    )
    assert theme["next_rep_cue"].startswith("At the top of the pull, finish tall")


def test_unresolved_check_cannot_be_replaced_with_no_issue_summary():
    brief = build_actionable_coaching_brief(analysis("UNCERTAIN", "COMPLETE_FINISH"))
    merged = _merge_llm_prose_with_brief("No issue was detected.", {}, brief)
    assert (
        merged["overall_assessment"]
        == "One or more covered deadlift checks were inconclusive."
    )


def test_all_confidently_favorable_checks_are_presented_independently():
    feedback = build_template_feedback(
        analysis("EARLY_HIP_RISE_NOT_DETECTED", "COMPLETE_FINISH")
    )

    assert feedback["positive_observations"] == [
        "Early hip–shoulder rise stayed coordinated.",
        "Terminal knee extension reached the implemented range.",
        "Trunk–thigh extension reached the implemented range.",
    ]
    assert feedback["neutral_context"] == [
        "Keep pushing the floor away smoothly and repeat the same extended top position."
    ]


def test_uncertain_torso_preserves_favorable_finish_components_and_actual_range():
    feedback = build_template_feedback(
        analysis(
            "EARLY_HIP_RISE_NOT_DETECTED",
            "UNCERTAIN",
            {
                "projected_knee_angle_degrees": 178.24,
                "projected_trunk_thigh_angle_degrees": 167.19,
                "torso_inclination_from_vertical_degrees": 12.82,
            },
        )
    )

    assert feedback["overall_assessment"] == (
        "No corrective issue was detected in the checks that could be assessed confidently. "
        "One finish-position measurement was inconclusive."
    )
    assert feedback["positive_observations"] == [
        "Early hip–shoulder rise stayed coordinated.",
        "Terminal knee extension reached the implemented range.",
        "Trunk–thigh extension reached the implemented range.",
    ]
    assert any(
        limitation == "Torso inclination measured 12.82°, within the provisional uncertainty range (7° to 17°), so that component could not be classified confidently."
        for limitation in feedback["limitations"]
    )
    assert all("unavailable" not in text.casefold() for text in feedback["limitations"])


def test_genuinely_insufficient_finish_evidence_is_not_presented_as_uncertain_data():
    feedback = build_template_feedback(
        analysis("EARLY_HIP_RISE_NOT_DETECTED", "INSUFFICIENT_EVIDENCE")
    )

    assert "Finish-position evidence was insufficient for assessment." in feedback["limitations"]
    assert not any("Torso inclination measured" in text for text in feedback["limitations"])


def test_local_model_validation_failure_still_presents_deterministic_coaching():
    from powerlifting_coach.app import render_coaching_feedback_html

    feedback = build_template_feedback(
        analysis("EARLY_HIP_RISE_NOT_DETECTED", "COMPLETE_FINISH"),
        "invalid_llm_response",
    )
    rendered = render_coaching_feedback_html(
        feedback, "fallback", "local_model_schema_validation_failed"
    )

    assert "Deterministic coaching feedback." in rendered
    assert "feedback is unavailable" not in rendered
    assert "Local model did not produce validated feedback" not in rendered
