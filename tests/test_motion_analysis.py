from powerlifting_coach.motion_analysis import (
    _movement_angle_summary,
    analyse_squat_landmarks,
    calculate_frame_metrics,
)
from powerlifting_coach.app import _analysis_summary, _calibration_diagnostics, _phase_angle_table
from powerlifting_coach.llm_feedback import build_feedback_payload


def _frame(hip_y: float, visibility: float = 0.9, presence: float = 0.9):
    return {
        "left_shoulder": {
            "x": 0.5,
            "y": hip_y - 0.3,
            "visibility": visibility,
            "presence": presence,
        },
        "left_hip": {
            "x": 0.5,
            "y": hip_y,
            "visibility": visibility,
            "presence": presence,
        },
        "left_knee": {
            "x": 0.5,
            "y": 0.6,
            "visibility": visibility,
            "presence": presence,
        },
        "left_ankle": {
            "x": 0.5,
            "y": 1.0,
            "visibility": visibility,
            "presence": presence,
        },
        "left_heel": {
            "x": 0.45,
            "y": 1.0,
            "visibility": visibility,
            "presence": presence,
        },
        "left_foot_index": {
            "x": 0.6,
            "y": 1.0,
            "visibility": visibility,
            "presence": presence,
        },
    }


def test_frame_validity_uses_hip_for_phase_tracking_and_abstains_per_metric():
    assert calculate_frame_metrics(_frame(0.2))["valid"] is True

    low_presence = _frame(0.2)
    low_presence["left_hip"]["presence"] = 0.59
    assert calculate_frame_metrics(low_presence)["valid"] is False

    missing_foot = _frame(0.2)
    del missing_foot["left_foot_index"]
    metrics = calculate_frame_metrics(missing_foot)
    assert metrics["valid"] is True
    assert metrics["left_knee_angle"] is not None
    assert metrics["knee_x_relative_to_toe"] is None

    missing_ankle = _frame(0.2)
    del missing_ankle["left_ankle"]
    metrics = calculate_frame_metrics(missing_ankle)
    assert metrics["valid"] is True
    assert metrics["left_knee_angle"] is None


def test_analysis_uses_setup_phase_bottom_window_and_clean_rep_summary():
    hip_positions = [
        0.2,
        0.2,
        0.22,
        0.24,
        0.3,
        0.45,
        0.65,
        0.8,
        0.65,
        0.45,
        0.3,
        0.22,
        0.2,
    ]
    result = analyse_squat_landmarks([_frame(hip_y) for hip_y in hip_positions], fps=30)

    assert result["quality"]["video_valid"] is True
    assert result["quality"]["tracking_valid_frame_ratio"] == 1.0
    assert result["reps_detected"] == 1
    assert result["frames"][0]["phase"] == "setup"
    assert result["bottom_frame"] in {7, 8}
    assert result["rep"]["phase_timings"]
    assert result["rep"]["phase_angles"]
    assert result["rep"]["findings"]

    calibration = result["kinematic_self_calibration"]
    summary, _, _, _ = _calibration_diagnostics(result)
    assert calibration["diagnostic_only"] is True
    assert calibration["usage"]["body_scale"] == {
        "diagnostic_only": True,
        "consumers": [],
        "classification_input": False,
        "coaching_input": False,
    }
    assert calibration["usage"]["segment_deviations"]["repairs_or_filters_landmarks"] is False
    assert calibration["usage"]["reacquisition"]["diagnostic_only"] is True
    assert summary["diagnostic_only"] is True
    assert summary["usage"] == calibration["usage"]


def test_setup_only_angles_are_not_reported_as_movement_minima():
    phases = ["setup", "setup", "descent", "bottom", "ascent", "final_standing"]
    metrics = [
        {"left_knee_angle": 176.1, "left_hip_angle": 175.2},
        {"left_knee_angle": 177.0, "left_hip_angle": 176.0},
        *[{"left_knee_angle": None, "left_hip_angle": None} for _ in range(3)],
        {"left_knee_angle": 178.0, "left_hip_angle": 177.0},
    ]
    summary = _movement_angle_summary(phases, metrics)
    assert summary["movement_interval"] == {
        "start_frame": 2, "end_frame": 4, "frame_count": 3,
        "phases": ["descent", "bottom", "ascent"],
    }
    assert summary["minimum_knee_angle"]["value_degrees"] is None
    assert summary["minimum_hip_angle"]["evidence_status"] == "unavailable"
    assert "No usable angle observations" in summary["minimum_knee_angle"]["unavailable_or_partial_reason"]


def test_complete_movement_angle_evidence_reports_the_movement_minimum():
    phases = ["setup", "descent", "bottom", "ascent", "final_standing"]
    metrics = [
        {"left_knee_angle": 170.0, "left_hip_angle": 165.0},
        {"left_knee_angle": 130.0, "left_hip_angle": 120.0},
        {"left_knee_angle": 88.4, "left_hip_angle": 75.2},
        {"left_knee_angle": 110.0, "left_hip_angle": 100.0},
        {"left_knee_angle": 175.0, "left_hip_angle": 170.0},
    ]
    summary = _movement_angle_summary(phases, metrics)
    assert summary["minimum_knee_angle"]["value_degrees"] == 88.4
    assert summary["minimum_hip_angle"]["value_degrees"] == 75.2
    assert summary["minimum_knee_angle"]["evidence_status"] == "complete"
    assert summary["minimum_knee_angle"]["coverage_ratio"] == 1.0


def test_partial_movement_angle_evidence_is_qualified_in_all_consumers():
    phases = ["setup", "descent", "bottom", "ascent", "final_standing"]
    metrics = [
        {"left_knee_angle": 175.0, "left_hip_angle": 170.0},
        {"left_knee_angle": 120.0, "left_hip_angle": None},
        {"left_knee_angle": None, "left_hip_angle": 82.0},
        {"left_knee_angle": 105.0, "left_hip_angle": 95.0},
        {"left_knee_angle": 176.0, "left_hip_angle": 172.0},
    ]
    movement = _movement_angle_summary(phases, metrics)
    analysis = {
        "lift": "squat", "analysis_status": "warning",
        "quality": {"analysis_valid": True, "confidence": "medium"},
        "reps_detected": 1,
        "rep": {"phase_angles": {}, "findings": [], "warnings": []},
        "warnings": [],
        "measurements": {
            "minimum_knee_angle": movement["minimum_knee_angle"]["value_degrees"],
            "minimum_hip_angle": movement["minimum_hip_angle"]["value_degrees"],
            "movement_angle_summary": movement,
        },
    }
    assert movement["minimum_knee_angle"]["evidence_status"] == "partial"
    assert movement["minimum_knee_angle"]["coverage_ratio"] == 0.667
    assert "available observations" in movement["minimum_knee_angle"]["description"]
    assert _analysis_summary(analysis)["measurements"] == analysis["measurements"]
    assert "available observations only" in _phase_angle_table(analysis)["data"][-1][1]
    assert build_feedback_payload(analysis)["descriptive_movement_angles"] == movement
