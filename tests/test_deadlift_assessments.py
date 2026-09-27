
from powerlifting_coach.deadlift_analysis import (
    DeadliftAnalysisConfig,
    _detect_phases,
    analyse_deadlift_landmarks,
    assess_early_hip_shoulder_coordination,
    assess_finish_posture,
    deadlift_diagnostics_view,
)


def test_early_coordination_positive_requires_persistent_frames():
    assert assess_early_hip_shoulder_coordination([0.0, 0.11, 0.12, 0.13]) == "EARLY_HIP_RISE_DETECTED"
    assert assess_early_hip_shoulder_coordination([0.11, 0.04, 0.12, 0.03, 0.13]) == "UNCERTAIN"


def test_early_coordination_negative_uncertain_and_insufficient():
    assert assess_early_hip_shoulder_coordination([0.0, 0.02, 0.05]) == "EARLY_HIP_RISE_NOT_DETECTED"
    assert assess_early_hip_shoulder_coordination([0.0, 0.07, 0.09]) == "UNCERTAIN"
    assert assess_early_hip_shoulder_coordination([0.0, 0.12]) == "INSUFFICIENT_EVIDENCE"


def _finish(knee: float, hip: float, torso: float) -> dict:
    return {
        "projected_knee_angle_degrees": knee,
        "projected_trunk_thigh_angle_degrees": hip,
        "torso_inclination_from_vertical_degrees": torso,
    }


def test_finish_posture_complete_incomplete_and_uncertain():
    assert assess_finish_posture(_finish(172, 167, 5), evidence_frame_count=3) == "COMPLETE_FINISH"
    assert assess_finish_posture(_finish(150, 170, 5), evidence_frame_count=3) == "INCOMPLETE_FINISH"
    assert assess_finish_posture(_finish(165, 160, 12), evidence_frame_count=3) == "UNCERTAIN"


def test_finish_posture_insufficient_and_requires_multiple_frames():
    assert assess_finish_posture(None, evidence_frame_count=3) == "INSUFFICIENT_EVIDENCE"
    assert assess_finish_posture(_finish(172, 167, 5), evidence_frame_count=2) == "INSUFFICIENT_EVIDENCE"


def _phase_config() -> DeadliftAnalysisConfig:
    return DeadliftAnalysisConfig(smoothing_window=1)


def test_deadlift_phases_ignore_long_preamble_and_use_local_setup():
    wrist = [0.65] * 8 + [0.68, 0.72, 0.76, 0.80] + [0.80] * 6
    wrist += [0.79, 0.77, 0.74, 0.71, 0.68, 0.65] + [0.65] * 5
    knee = [0.70] * len(wrist)

    phases, diagnostics = _detect_phases(wrist, knee, _phase_config())

    assert diagnostics["failure_reason"] is None
    assert phases["setup"] == {"start_frame": 14, "end_frame": 17}
    assert phases["lift_off"]["start_frame"] == 18
    assert phases["knee_passing_ascent"]["start_frame"] == 21


def test_deadlift_stable_bottom_precedes_sustained_ascent_and_ordered_ranges():
    wrist = [0.82] * 6 + [0.80, 0.77, 0.74, 0.70, 0.66, 0.62, 0.58]
    wrist += [0.58] * 4 + [0.61, 0.65, 0.70, 0.75, 0.80, 0.82] + [0.82] * 4
    knee = [0.72] * len(wrist)

    phases, diagnostics = _detect_phases(wrist, knee, _phase_config())

    lift_off = phases["lift_off"]["start_frame"]
    knee_passing = phases["knee_passing_ascent"]["start_frame"]
    top = phases["top_position"]["start_frame"]
    descent = phases["controlled_descent"]["start_frame"]
    returned = phases["return_to_bottom"]["start_frame"]
    assert phases["setup"]["end_frame"] < lift_off < knee_passing < top < descent < returned
    assert knee_passing - lift_off >= 2
    assert phases["below_knee_ascent"]["start_frame"] <= phases["below_knee_ascent"]["end_frame"]
    assert diagnostics["return_evidence"]["repetition_complete"] is True


def test_deadlift_video_ending_at_top_preserves_top_but_is_not_complete():
    wrist = [0.82] * 6 + [0.80, 0.77, 0.74, 0.70, 0.66, 0.62, 0.58] + [0.58] * 5
    phases, diagnostics = _detect_phases(wrist, [0.72] * len(wrist), _phase_config())

    assert phases["top_position"] == {"start_frame": 12, "end_frame": 17}
    assert "controlled_descent" not in phases
    assert "return_to_bottom" not in phases
    assert diagnostics["return_evidence"]["repetition_complete"] is False


def test_deadlift_top_is_confirmed_before_descent_and_ranges_do_not_overlap():
    wrist = [0.82] * 6 + [0.80, 0.77, 0.74, 0.70, 0.66, 0.62, 0.58]
    wrist += [0.58] * 4 + [0.61, 0.65, 0.70, 0.75, 0.80, 0.82] + [0.82] * 4
    phases, diagnostics = _detect_phases(wrist, [0.72] * len(wrist), _phase_config())

    ordered = [
        phases[name]
        for name in (
            "below_knee_ascent",
            "knee_passing_ascent",
            "top_position",
            "controlled_descent",
            "return_to_bottom",
        )
    ]
    assert all(left["end_frame"] < right["start_frame"] for left, right in zip(ordered, ordered[1:]))
    assert diagnostics["top_position_evidence"]["confirmed_frame"] < phases["controlled_descent"]["start_frame"]


def test_returned_bottom_frames_never_supply_finish_posture():
    wrist = [0.82] * 6 + [0.80, 0.77, 0.74, 0.70, 0.66, 0.62, 0.58]
    wrist += [0.58] * 4 + [0.61, 0.65, 0.70, 0.75, 0.80, 0.82] + [0.82] * 4

    def landmark(x, y):
        return {"x": x, "y": y, "visibility": 1.0, "presence": 1.0}

    frames = []
    for frame_index, wrist_y in enumerate(wrist):
        top = 12 <= frame_index <= 16
        frame = {}
        for side in ("left", "right"):
            points = {
                "shoulder": (0.50, 0.25 if top else 0.42),
                "elbow": (0.50, 0.42),
                "wrist": (0.50, wrist_y),
                "hip": (0.50, 0.50),
                "knee": (0.50 if top else 0.56, 0.68),
                "ankle": (0.50, 0.85),
                "heel": (0.48, 0.87),
                "foot_index": (0.55, 0.87),
            }
            frame.update({f"{side}_{name}": landmark(*point) for name, point in points.items()})
        frames.append(frame)

    result = analyse_deadlift_landmarks(
        frames,
        processing_metadata={"frame_width": 1000, "frame_height": 1000},
        config=_phase_config(),
    )
    finish = next(
        item for item in result["metric_results"]
        if item["metric_id"] == "deadlift_finish_geometry_2d"
    )

    assert finish["source_frames"] == [12, 13, 14, 15, 16]
    assert max(finish["source_frames"]) < result["phase_frames"]["controlled_descent"]["start_frame"]
    assert not set(finish["source_frames"]) & set(
        range(result["phase_frames"]["return_to_bottom"]["start_frame"], len(frames))
    )

    calibration = result["calibration"]
    body_scale = calibration["usage"]["body_scale"]
    assert result["kinematic_self_calibration"] == calibration
    assert calibration["diagnostic_only"] is False
    assert body_scale["classification_input"] is True
    assert body_scale["coaching_input"] is True
    assert body_scale["consumers"] == [
        "deadlift_early_coordination_2d measurement",
        "deadlift_early_coordination_2d classification",
        "deadlift_early_coordination_2d coaching",
    ]
    assert calibration["usage"]["segment_deviations"]["diagnostic_only"] is True
    assert calibration["usage"]["reacquisition"]["repairs_or_filters_landmarks"] is False
    assert deadlift_diagnostics_view(result)["calibration_summary"]["usage"] == calibration["usage"]


def test_deadlift_rejects_late_ascent_plateau_and_uses_genuine_top_for_finish():
    wrist = [0.82] * 6 + [0.80, 0.76, 0.72, 0.68, 0.64]
    early_plateau = list(range(len(wrist), len(wrist) + 5))
    wrist += [0.60] * 5 + [0.57, 0.54, 0.51, 0.48] + [0.48] * 5
    genuine_top = list(range(20, 25))
    wrist += [0.485, 0.49, 0.50, 0.515, 0.53, 0.56, 0.60, 0.65, 0.71]
    wrist += [0.77, 0.81, 0.82] + [0.82] * 4

    def landmark(x, y):
        return {"x": x, "y": y, "visibility": 1.0, "presence": 1.0}

    frames = []
    for frame_index, wrist_y in enumerate(wrist):
        is_true_top = frame_index in genuine_top
        frame = {}
        for side in ("left", "right"):
            points = {
                "shoulder": (0.50, 0.25 if is_true_top else 0.42),
                "elbow": (0.50, 0.42),
                "wrist": (0.50, wrist_y),
                "hip": (0.50, 0.50),
                "knee": (0.50 if is_true_top else 0.56, 0.68),
                "ankle": (0.50, 0.85),
                "heel": (0.48, 0.87),
                "foot_index": (0.55, 0.87),
            }
            frame.update(
                {f"{side}_{name}": landmark(*point) for name, point in points.items()}
            )
        frames.append(frame)

    result = analyse_deadlift_landmarks(
        frames,
        processing_metadata={"frame_width": 1000, "frame_height": 1000},
        config=_phase_config(),
    )
    phases = result["phase_frames"]
    evidence = result["diagnostics"]["phase_detection"]["top_position_evidence"]
    finish = next(
        item
        for item in result["metric_results"]
        if item["metric_id"] == "deadlift_finish_geometry_2d"
    )

    assert not set(early_plateau) & set(
        range(phases["top_position"]["start_frame"], phases["top_position"]["end_frame"] + 1)
    )
    assert phases["top_position"] == {"start_frame": 20, "end_frame": 24}
    assert evidence["wrist_proxy_values"] == [0.48] * 5
    assert finish["source_frames"] == genuine_top
    assert finish["classification"] == "COMPLETE_FINISH"
    assert phases["controlled_descent"]["start_frame"] == 25
    assert phases["controlled_descent"]["end_frame"] >= 34
    assert phases["return_to_bottom"]["start_frame"] > 34


def test_deadlift_rejects_pull_without_real_below_knee_interval():
    wrist = [0.82] * 6 + [0.70, 0.66, 0.64] + [0.64] * 5
    knee = [0.70] * len(wrist)

    phases, diagnostics = _detect_phases(wrist, knee, _phase_config())

    assert phases == {}
    assert diagnostics["failure_reason"] == "lift_off_not_observed"
