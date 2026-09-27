import pytest

from powerlifting_coach.deadlift_analysis import DeadliftAnalysisConfig
from powerlifting_coach.evaluation.deadlift_scale import (
    coordination_measurement,
    linked_chain_scale,
    transform_coordinates,
)


def frame(shoulder_y=0.2, hip_y=0.4, knee_y=0.6, ankle_y=0.8):
    return {
        f"left_{name}": {"x": 0.5, "y": y, "visibility": 1.0, "presence": 1.0}
        for name, y in (
            ("shoulder", shoulder_y),
            ("hip", hip_y),
            ("knee", knee_y),
            ("ankle", ankle_y),
        )
    }


def test_scale_and_coordination_use_linked_chain_and_production_numerator():
    frames = [frame(), frame(0.18, 0.35), frame(0.16, 0.30)]
    scale = linked_chain_scale(frames, "left", [0, 1, 2], 0.6)
    measurement = coordination_measurement(
        frames, "left", [0, 1, 2], 0.6, scale["value"], DeadliftAnalysisConfig()
    )

    assert scale == {"value": pytest.approx(0.62), "sample_count": 3}
    assert measurement[
        "framewise_hip_minus_shoulder_vertical_progress"
    ] == pytest.approx([0.0, 0.03 / 0.62, 0.06 / 0.62])
    assert measurement["classification"] == "UNCERTAIN"


def test_setup_scale_is_unavailable_without_complete_linked_chain_evidence():
    frames = [frame(), frame()]
    del frames[0]["left_knee"]
    frames[1]["left_hip"]["visibility"] = 0.1

    assert linked_chain_scale(frames, "left", [0, 1], 0.6) == {
        "value": None,
        "sample_count": 0,
    }


def test_uniform_scale_and_translation_preserve_normalised_measurement():
    frames = [frame(), frame(0.18, 0.35), frame(0.16, 0.30)]
    changed = transform_coordinates(frames, 2.3, 0.4, -0.2)
    original_scale = linked_chain_scale(frames, "left", [0, 1, 2], 0.6)["value"]
    changed_scale = linked_chain_scale(changed, "left", [0, 1, 2], 0.6)["value"]
    original = coordination_measurement(
        frames, "left", [0, 1, 2], 0.6, original_scale, DeadliftAnalysisConfig()
    )
    transformed = coordination_measurement(
        changed, "left", [0, 1, 2], 0.6, changed_scale, DeadliftAnalysisConfig()
    )

    assert changed_scale == pytest.approx(original_scale * 2.3)
    assert transformed[
        "framewise_hip_minus_shoulder_vertical_progress"
    ] == pytest.approx(
        original["framewise_hip_minus_shoulder_vertical_progress"], abs=1e-12
    )
    assert transformed["classification"] == original["classification"]
