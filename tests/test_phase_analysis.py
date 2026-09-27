from powerlifting_coach.motion_analysis import (
    SquatAnalysisConfig,
    analyse_squat_landmarks,
)


def _frame(
    hip_y: float,
    shoulder_y: float | None = None,
    hip_x: float = 0.5,
    shoulder_x: float = 0.5,
):
    shoulder_y = hip_y - 0.3 if shoulder_y is None else shoulder_y
    return {
        "left_shoulder": {
            "x": shoulder_x,
            "y": shoulder_y,
            "visibility": 0.95,
            "presence": 0.95,
        },
        "left_hip": {"x": hip_x, "y": hip_y, "visibility": 0.95, "presence": 0.95},
        "left_knee": {"x": 0.55, "y": 0.62, "visibility": 0.95, "presence": 0.95},
        "left_ankle": {"x": 0.5, "y": 1.0, "visibility": 0.95, "presence": 0.95},
        "left_heel": {"x": 0.45, "y": 1.0, "visibility": 0.95, "presence": 0.95},
        "left_foot_index": {"x": 0.65, "y": 1.0, "visibility": 0.95, "presence": 0.95},
    }


def _rep(bottom_pause_frames=3, ascent=None, fps=10):
    ascent = ascent or [0.68, 0.58, 0.48, 0.38, 0.28, 0.20, 0.20, 0.20, 0.20, 0.20]
    hips = (
        [0.20] * 6
        + [0.28, 0.38, 0.50, 0.62, 0.72]
        + [0.74] * bottom_pause_frames
        + ascent
    )
    return analyse_squat_landmarks([_frame(y) for y in hips], fps=fps)


def test_state_transitions_from_setup_through_final_standing():
    result = _rep(bottom_pause_frames=3)
    phases = [row["phase"] for row in result["frames"]]

    ordered = [
        phase
        for phase in ["setup", "descent", "bottom", "ascent", "final_standing"]
        if phase in phases
    ]
    assert ordered == ["setup", "descent", "bottom", "ascent", "final_standing"]
    assert result["reps_detected"] == 1


def test_prolonged_slow_ascent_is_not_reclassified_as_bottom():
    slow_ascent = [
        0.735,
        0.73,
        0.725,
        0.72,
        0.715,
        0.70,
        0.66,
        0.58,
        0.46,
        0.32,
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
    ]
    result = _rep(bottom_pause_frames=3, ascent=slow_ascent)
    phases = [row["phase"] for row in result["frames"]]
    first_ascent = phases.index("ascent")

    assert "bottom" not in phases[first_ascent:]
    assert all(phase != "bottom" for phase in phases[first_ascent:])


def test_bottom_pause_requires_duration_strictly_greater_than_one_second():
    short_pause = _rep(bottom_pause_frames=10, fps=10)
    long_pause = _rep(bottom_pause_frames=13, fps=10)

    assert short_pause["rep"]["derived_metrics"]["bottom_pause_duration_seconds"] <= 1.0
    assert short_pause["rep"]["derived_metrics"]["bottom_pause_detected"] is False
    assert long_pause["rep"]["derived_metrics"]["bottom_pause_duration_seconds"] > 1.0
    assert long_pause["rep"]["derived_metrics"]["bottom_pause_detected"] is True


def test_depth_uses_bottom_phase_medians():
    result = _rep(bottom_pause_frames=5)
    depth = next(f for f in result["rep"]["findings"] if f["name"] == "depth")

    assert (
        depth["numeric_evidence"]["bottom_hip_y_median"]
        > depth["numeric_evidence"]["bottom_knee_y_median"]
    )
    assert depth["status"] == "likely_sufficient_depth"


def test_depth_medians_use_only_reported_co_observed_bottom_frames():
    hips = (
        [0.20] * 6
        + [0.28, 0.38, 0.50, 0.62, 0.72]
        + [0.74] * 5
        + [0.68, 0.58, 0.48, 0.38, 0.28, 0.20, 0.20, 0.20, 0.20, 0.20]
    )
    frames = [_frame(y) for y in hips]
    for frame in frames:
        for joint in ("shoulder", "hip", "knee", "ankle", "foot_index"):
            frame[f"right_{joint}"] = {
                **frame[f"left_{joint}"],
                "visibility": 0.99,
                "presence": 0.99,
            }
    for frame_index in (12, 13, 14):
        frames[frame_index]["left_knee"]["visibility"] = 0.0
    frames[12]["right_hip"]["visibility"] = 0.0
    frames[13]["right_knee"]["visibility"] = 0.0
    frames[14]["right_hip"]["y"] = 0.70
    frames[14]["right_knee"]["y"] = 0.76
    frames[15]["right_hip"]["y"] = 0.80
    frames[15]["right_knee"]["y"] = 0.76

    result = analyse_squat_landmarks(frames, fps=10)
    depth = next(
        metric
        for metric in result["metric_results"]
        if metric["metric_id"] == "squat_depth_2d_proxy"
    )

    assert depth["source_frames"] == [14, 15]
    assert depth["measurement"]["bottom_hip_y_median"] == 0.75
    assert depth["measurement"]["bottom_knee_y_median"] == 0.76
    assert depth["measurement"]["hip_minus_knee_y"] == -0.01
    assert depth["classification"] == "BORDERLINE"


def test_setup_to_bottom_torso_change_remains_descriptive_not_a_fault():
    hips = (
        [0.20] * 6
        + [0.28, 0.38, 0.50, 0.62, 0.72]
        + [0.74] * 5
        + [0.68, 0.58, 0.48, 0.38, 0.28, 0.20, 0.20, 0.20, 0.20, 0.20]
    )
    frames = [
        _frame(y, hip_x=0.5, shoulder_x=0.53 if y < 0.3 else 0.6205)
        for y in hips
    ]

    result = analyse_squat_landmarks(frames, fps=10)

    angles = result["rep"]["phase_angles"]
    assert angles["setup"]["torso_lean_degrees"]["median"] == 5.7
    assert angles["bottom"]["torso_lean_degrees"]["median"] == 21.9
    assert result["rep"]["derived_metrics"]["bottom_torso_lean_change_degrees"] == 16.2
    assert all(f["name"] != "torso_lean_change" for f in result["rep"]["findings"])
    assert all(f["name"] != "torso_lean_change" for f in result["detected_faults"])


def test_phase_summary_output_shape():
    result = _rep(bottom_pause_frames=3)
    rep = result["rep"]

    assert set(rep["phase_timings"]) == {
        "setup",
        "descent",
        "bottom",
        "ascent",
        "final_standing",
    }
    assert set(rep["phase_angles"]["bottom"]) == {
        "knee_angle",
        "hip_angle",
        "torso_lean_degrees",
    }
    assert set(rep["phase_angles"]["bottom"]["knee_angle"]) == {"median", "min", "max"}


def test_hip_first_ascent_finding_from_synthetic_hip_and_shoulder_sequences():
    hip_values = (
        [0.20] * 6
        + [0.35, 0.55, 0.74]
        + [0.74] * 3
        + [0.62, 0.50, 0.38, 0.26, 0.20, 0.20, 0.20, 0.20, 0.20]
    )
    shoulder_values = [y - 0.3 for y in hip_values]
    shoulder_values[12:16] = [0.43, 0.425, 0.415, 0.38]
    result = analyse_squat_landmarks(
        [_frame(h, s) for h, s in zip(hip_values, shoulder_values)],
        fps=10,
        config=SquatAnalysisConfig(
            hip_first_progress_threshold=0.1, hip_first_min_duration_seconds=0.1
        ),
    )

    assert any(f["name"] == "hip_first_ascent" for f in result["rep"]["findings"])


def test_early_ascent_coordination_rejects_smoothed_shoulder_dropout():
    hip_values = (
        [0.20] * 6
        + [0.35, 0.55, 0.74]
        + [0.74] * 3
        + [0.62, 0.50, 0.38, 0.26, 0.20, 0.20, 0.20, 0.20, 0.20]
    )
    shoulder_values = [y - 0.3 for y in hip_values]
    shoulder_values[12:16] = [0.43, 0.425, 0.415, 0.38]
    frames = [_frame(h, s) for h, s in zip(hip_values, shoulder_values)]
    for frame_index in (12, 13):
        frames[frame_index]["left_shoulder"]["visibility"] = 0.0

    result = analyse_squat_landmarks(
        frames,
        fps=10,
        config=SquatAnalysisConfig(
            hip_first_progress_threshold=0.1, hip_first_min_duration_seconds=0.1
        ),
    )
    coordination = next(
        metric
        for metric in result["metric_results"]
        if metric["metric_id"] == "squat_early_ascent_coordination"
    )

    assert coordination["evidence_status"] == "INSUFFICIENT"
    assert coordination["classification"] is None
    assert coordination["measurement"] is None


def _interpolate(points: list[tuple[int, float]], total: int = 570) -> list[float]:
    values = [points[0][1]] * total
    for (start_f, start_y), (end_f, end_y) in zip(points, points[1:]):
        span = max(1, end_f - start_f)
        for frame in range(start_f, min(end_f + 1, total)):
            t = (frame - start_f) / span
            values[frame] = start_y + t * (end_y - start_y)
    if points[-1][0] < total:
        for frame in range(points[-1][0], total):
            values[frame] = points[-1][1]
    return values


def _regression_fixture_frames():
    hips = _interpolate(
        [
            (0, 0.604),
            (45, 0.604),
            (104, 0.636),
            (188, 0.767),
            (205, 0.765),
            (330, 0.604),
            (569, 0.604),
        ]
    )
    return [_frame(y) for y in hips]


def test_regression_fixture_descent_bottom_and_monotonic_ascent_phase_order():
    result = analyse_squat_landmarks(_regression_fixture_frames(), fps=30)
    phases = [row["phase"] for row in result["frames"]]
    descent_start = phases.index("descent")
    bottom_indices = [i for i, phase in enumerate(phases) if phase == "bottom"]
    ascent_start = phases.index("ascent")

    assert result["reps_detected"] == 1
    assert 84 <= descent_start <= 124
    assert bottom_indices
    assert 176 <= result["bottom_frame"] <= 200
    assert ascent_start > bottom_indices[-1]
    assert "bottom" not in phases[ascent_start:]


def test_no_rep_input_does_not_emit_depth_finding_or_bottom_fallback():
    result = analyse_squat_landmarks([_frame(0.604) for _ in range(90)], fps=30)

    assert result["reps_detected"] == 0
    assert result["depth_assessment"] == "not_assessed"
    assert result["bottom_frame"] is None
    assert result["bottom_timestamp_seconds"] is None
    assert result["rep"] == {}


def test_setup_baseline_is_frozen_in_segmentation_debug():
    result = analyse_squat_landmarks(_regression_fixture_frames(), fps=30)
    debug = result["segmentation_debug"]

    assert debug["baseline_start_frame"] < debug["first_descent_candidate_frame"]
    assert debug["baseline_end_frame"] < debug["first_descent_candidate_frame"]
    assert (
        debug["baseline_hip_y"] == result["rep"]["segmentation_debug"]["baseline_hip_y"]
    )
    assert abs(debug["baseline_hip_y"] - 0.604) < 0.003


def test_velocity_uses_fps_delta_seconds_not_implicit_frame_units():
    result = analyse_squat_landmarks(
        [_frame(y) for y in [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27]],
        fps=30,
    )

    assert (
        result["segmentation_debug"]["descent_velocity_threshold_per_second"] == 0.015
    )
    assert result["segmentation_debug"]["first_descent_candidate_frame"] is not None
