from powerlifting_coach.motion_analysis import analyse_squat_landmarks


def _frame(hip_y: float, visibility: float = 0.95) -> dict:
    return {
        f"left_{name}": {
            "x": x,
            "y": y,
            "visibility": visibility,
            "presence": visibility,
        }
        for name, x, y in (
            ("shoulder", 0.5, hip_y - 0.3),
            ("hip", 0.5, hip_y),
            ("knee", 0.55, 0.62),
            ("ankle", 0.5, 1.0),
            ("foot_index", 0.65, 1.0),
        )
    }


def _complete_rep() -> list[dict]:
    hips = (
        [0.20] * 6
        + [0.28, 0.38, 0.50, 0.62, 0.72]
        + [0.74] * 3
        + [0.68, 0.58, 0.48, 0.38, 0.28, 0.20]
        + [0.20] * 4
    )
    return [_frame(y) for y in hips]


def test_neither_side_with_usable_phase_signal_is_rejected():
    result = analyse_squat_landmarks([_frame(0.2, 0.5) for _ in range(20)], fps=10)

    assert result["analysis_status"] == "invalid"
    assert result["selected_side"] is None
    assert result["visibility"]["left_score"] == 0.5
    assert result["visibility"]["right_score"] == 0.0
    assert "no_usable_phase_trajectory" in result["recording_reliability"]["rejection_reasons"]


def test_short_hip_gaps_are_smoothed_instead_of_globally_rejected():
    frames = _complete_rep()
    for frame in frames[::2]:
        frame["left_hip"]["visibility"] = 0.1
    result = analyse_squat_landmarks(frames, fps=10)

    assert result["analysis_status"] in {"valid", "warning"}
    assert result["reps_detected"] == 1
    assert result["recording_reliability"]["selected_side_usable_frame_ratio"] < 0.75
    assert result["recording_reliability"]["rejection_reasons"] == []


def test_incomplete_movement_is_rejected_and_contains_no_faults():
    result = analyse_squat_landmarks([_frame(y) for y in [0.2] * 6 + [0.3, 0.5, 0.7]], fps=10)

    assert result["analysis_status"] == "invalid"
    assert "incomplete_repetition" in result["recording_reliability"]["rejection_reasons"]
    assert result["detected_faults"] == []


def test_missing_metric_landmark_abstains_without_invalidating_phase_tracking():
    frames = _complete_rep()
    for frame in frames:
        del frame["left_knee"]
    result = analyse_squat_landmarks(frames, fps=10)
    depth = next(item for item in result["metric_results"] if item["metric_id"] == "squat_depth_2d_proxy")

    assert result["reps_detected"] == 1
    assert depth["evidence_status"] == "INSUFFICIENT"
    assert depth["classification"] is None


def test_front_view_is_guidance_instead_of_a_global_rejection():
    result = analyse_squat_landmarks(
        _complete_rep(), fps=10, processing_metadata={"camera_view": "FRONT"}
    )

    assert result["analysis_status"] in {"valid", "warning"}
    assert result["recording_reliability"]["rejection_reasons"] == []
    assert "declared_front_view_may_limit_side_view_metrics" in result["recording_reliability"]["warnings"]


def test_low_aggregate_scores_do_not_block_a_usable_hip_trajectory():
    frames = _complete_rep()
    for frame in frames:
        for name in ("left_shoulder", "left_knee", "left_ankle"):
            frame[name]["visibility"] = 0.1
            frame[name]["presence"] = 0.1

    result = analyse_squat_landmarks(frames, fps=10)

    assert result["visibility"]["left_score"] < 0.6
    assert result["selected_side"] == "left"
    assert result["reps_detected"] == 1


def test_metric_can_use_other_side_than_phase_tracking():
    frames = _complete_rep()
    for frame in frames:
        frame["right_hip"] = dict(frame["left_hip"])
        frame["right_knee"] = dict(frame["left_knee"])
    for frame in frames[6:20]:
        frame["left_knee"]["visibility"] = 0.1
        frame["left_knee"]["presence"] = 0.1

    result = analyse_squat_landmarks(frames, fps=10)
    depth = next(item for item in result["metric_results"] if item["metric_id"] == "squat_depth_2d_proxy")

    assert result["selected_side"] == "left"
    assert depth["evidence_status"] == "SUFFICIENT"
    assert depth["landmarks_used"] == ["right_hip", "right_knee"]


def test_missing_foot_landmarks_do_not_prevent_phase_detection():
    frames = _complete_rep()
    for frame in frames:
        frame.pop("left_foot_index")

    result = analyse_squat_landmarks(frames, fps=10)

    assert result["reps_detected"] == 1
