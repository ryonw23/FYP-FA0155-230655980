
import json

from powerlifting_coach.coaching_rule_library import build_actionable_coaching_brief
from powerlifting_coach.bench_analysis import (
    BenchAnalysisConfig,
    analyse_bench_press_landmarks,
    bench_pause_diagnostics_export,
    bench_diagnostics_view,
)
from powerlifting_coach.llm_feedback import (
    _merge_llm_prose_with_brief,
    build_feedback_payload,
    build_template_feedback,
)
from powerlifting_coach.motion_analysis import save_analysis_json


def analysis(depth: str, pause: str, extension: str) -> dict:
    classifications = {
        "observed_side_elbow_depth_2d_proxy": depth,
        "motionless_bottom_2d_proxy": pause,
        "terminal_elbow_extension_2d_proxy": extension,
    }
    return {
        "lift": "bench_press",
        "analysis_status": "valid",
        "metric_results": [
            {
                "metric_id": metric_id,
                "classification": classification,
                "evidence_status": (
                    "INSUFFICIENT"
                    if classification == "INSUFFICIENT_EVIDENCE"
                    else "SUFFICIENT"
                ),
                "measurement": {},
                "reliability": {"confidence": "high"},
                "warnings": [],
            }
            for metric_id, classification in classifications.items()
        ],
        "warnings": [],
    }


def landmark_repetition(
    right_visible: bool = True, bottom_frames: int = 8, finish_frames: int = 8
) -> list[dict]:
    wrist_y = (
        [0.2] * 5
        + [0.3, 0.4, 0.5, 0.6]
        + [0.7] * bottom_frames
        + [0.6, 0.5, 0.4, 0.3, 0.2]
        + [0.2] * finish_frames
    )
    frames = []
    for y in wrist_y:
        frame = {}
        for side in ("left", "right"):
            visibility = 0.99 if side == "left" or right_visible else 0.1
            elbow_y = 0.52 if y > 0.65 else 0.4
            for joint, joint_y in (("shoulder", 0.5), ("elbow", elbow_y), ("wrist", y)):
                frame[f"{side}_{joint}"] = {
                    "x": 0.5,
                    "y": joint_y,
                    "visibility": visibility,
                    "presence": visibility,
                }
        frames.append(frame)
    return frames


def bench_repetition_with_wrist_path(
    wrist_y: list[float], *, terminal_elbow_x: float = 0.5
) -> list[dict]:
    frames = []
    for index, y in enumerate(wrist_y):
        frame = {}
        for side in ("left", "right"):
            elbow_y = 0.52 if y >= 0.65 else 0.4
            elbow_x = terminal_elbow_x if index >= len(wrist_y) - 8 else 0.5
            for joint, x, joint_y in (
                ("shoulder", 0.5, 0.5),
                ("elbow", elbow_x, elbow_y),
                ("wrist", 0.5, y),
            ):
                frame[f"{side}_{joint}"] = {
                    "x": x,
                    "y": joint_y,
                    "visibility": 0.99,
                    "presence": 0.99,
                }
        frames.append(frame)
    return frames


def terminal_metric(result: dict) -> dict:
    return competition_metrics(result)["terminal_elbow_extension_2d_proxy"]


def test_early_wrist_return_is_rejected_when_ascent_meaningfully_continues():
    path = [0.2] * 6 + [0.3, 0.4, 0.5, 0.6, 0.7] + [0.6, 0.5, 0.35]
    path += [0.24] * 8 + [0.18, 0.16] + [0.15] * 8
    result = analyse_bench_press_landmarks(
        bench_repetition_with_wrist_path(path), fps=30.0
    )
    diagnostic = result["diagnostics"]["phase_detection"]
    assert diagnostic["terminal_evidence_window"][0] > 21
    assert any(
        candidate["continued_ascent_after_window"]
        for candidate in diagnostic["terminal_confirmation"]["tested_candidates"]
    )


def test_genuine_stable_top_supplies_the_only_terminal_evidence_window():
    result = analyse_bench_press_landmarks(landmark_repetition(), fps=30.0)
    diagnostic = result["diagnostics"]["phase_detection"]
    terminal = terminal_metric(result)
    assert diagnostic["terminal_position_confirmed"] is True
    assert result["phase_frames"]["complete"]["start_frame"] == diagnostic[
        "terminal_evidence_window"
    ][0]
    assert terminal["classification"] == "COMPLETE"
    assert terminal["source_frames"] == diagnostic["terminal_evidence_window"]
    assert terminal["landmark_origin"]


def test_bench_calibration_usage_metadata_and_summary_are_consistent():
    result = analyse_bench_press_landmarks(landmark_repetition(), fps=30.0)
    calibration = result["calibration"]
    body_scale = calibration["usage"]["body_scale"]

    assert result["kinematic_self_calibration"] == calibration
    assert calibration["diagnostic_only"] is False
    assert body_scale["diagnostic_only"] is False
    assert body_scale["classification_input"] is False
    assert body_scale["coaching_input"] is True
    assert body_scale["consumers"] == [
        "bench_wrist_trajectory_2d descriptive measurement",
        "bench_wrist_trajectory_2d descriptive coaching",
    ]
    assert calibration["usage"]["segment_deviations"]["diagnostic_only"] is True
    assert calibration["usage"]["reacquisition"]["repairs_or_filters_landmarks"] is False
    assert bench_diagnostics_view(result)["calibration_summary"]["usage"] == calibration["usage"]


def test_stable_top_with_bent_elbows_remains_detectable_as_incomplete():
    frames = landmark_repetition()
    for frame in frames[-8:]:
        frame["left_elbow"]["x"] = 0.65
        frame["right_elbow"]["x"] = 0.65
    result = analyse_bench_press_landmarks(frames, fps=30.0)
    assert result["diagnostics"]["phase_detection"]["terminal_position_confirmed"]
    assert terminal_metric(result)["classification"] == "INCOMPLETE"


def test_tracking_loss_cannot_turn_smoothed_derived_top_into_terminal_evidence():
    frames = landmark_repetition()
    for frame in frames[-8:]:
        for side in ("left", "right"):
            frame[f"{side}_wrist"]["visibility"] = 0.1
            frame[f"{side}_wrist"]["presence"] = 0.1
    result = analyse_bench_press_landmarks(frames, fps=30.0)
    terminal = terminal_metric(result)
    assert terminal["classification"] == "INSUFFICIENT_EVIDENCE"
    assert "fully observed" in terminal["warnings"][0]
    assert any(
        candidate["signal_provenance"] == "derived_only_or_unavailable"
        for candidate in result["diagnostics"]["phase_detection"][
            "terminal_confirmation"
        ]["tested_candidates"]
    )


def test_unconfirmed_top_has_no_completion_or_finish_comparison():
    frames = landmark_repetition(finish_frames=3)
    result = analyse_bench_press_landmarks(frames, fps=30.0)
    metrics = {item["metric_id"]: item for item in result["metric_results"]}
    phases = result["phase_frames"]
    diagnostic = result["diagnostics"]["phase_detection"]

    assert diagnostic["terminal_position_confirmed"] is False
    assert "complete" not in phases
    assert phases["ascent"]["end_frame"] == len(frames) - 1
    assert diagnostic["observed_ascent_end_frame"] == phases["ascent"]["end_frame"]
    assert metrics["bench_wrist_trajectory_2d"]["measurement"] is None
    assert metrics["bench_finish_geometry_similarity"]["measurement"] is None
    assert metrics["observed_side_elbow_depth_2d_proxy"]["classification"] == (
        "DEPTH_SUFFICIENT"
    )
    assert (
        metrics["terminal_elbow_extension_2d_proxy"]["classification"]
        == "INSUFFICIENT_EVIDENCE"
    )


def test_analyser_separates_supported_assessments_from_experimental_pause():
    result = analyse_bench_press_landmarks(landmark_repetition(), fps=30.0)
    classifications = {
        metric["metric_id"]: metric["classification"]
        for metric in result["metric_results"]
    }
    assert classifications["observed_side_elbow_depth_2d_proxy"] == "DEPTH_SUFFICIENT"
    assert classifications["terminal_elbow_extension_2d_proxy"] == "COMPLETE"
    pause = result["experimental_diagnostics"]["motionless_bottom_2d_proxy"]
    assert pause["scope"] == "experimental_diagnostic_only"
    assert pause["coaching_eligible"] is False
    assert pause["result"]["classification"] == "MOTIONLESS_BOTTOM_OBSERVED"


def test_analyser_withholds_bilateral_depth_conclusion_for_one_visible_side():
    result = analyse_bench_press_landmarks(
        landmark_repetition(right_visible=False), fps=30.0
    )
    depth = next(
        metric
        for metric in result["metric_results"]
        if metric["metric_id"] == "observed_side_elbow_depth_2d_proxy"
    )
    assert depth["classification"] == "INSUFFICIENT_EVIDENCE"
    assert depth["measurement"]["reliable_sides"] == ["left"]


def test_single_raw_bottom_frame_cannot_support_depth_or_coaching():
    frames = landmark_repetition()
    for index in range(9, 17):
        if index == 12:
            continue
        for side in ("left", "right"):
            frames[index][f"{side}_elbow"]["visibility"] = 0.1
            frames[index][f"{side}_elbow"]["presence"] = 0.1

    result = analyse_bench_press_landmarks(frames, fps=30.0)
    depth = competition_metrics(result)["observed_side_elbow_depth_2d_proxy"]
    support = depth["measurement"]["per_side_support"]

    assert depth["classification"] == "INSUFFICIENT_EVIDENCE"
    assert support["left"]["raw_observed_frames"] == [12]
    assert support["left"]["raw_sample_count"] == 1
    assert support["left"]["elapsed_evidence_seconds"] == 0.0
    feedback = build_template_feedback(result)
    assert feedback["priority_cues"] == []
    assert not any(
        "depth" in str(item).lower() for item in feedback["positive_observations"]
    )


def test_persistent_depth_results_have_temporal_support():
    sufficient = analyse_bench_press_landmarks(landmark_repetition(), fps=30.0)
    insufficient_frames = landmark_repetition()
    for frame in insufficient_frames[9:17]:
        for side in ("left", "right"):
            frame[f"{side}_elbow"]["y"] = 0.42

    sufficient_depth = competition_metrics(sufficient)[
        "observed_side_elbow_depth_2d_proxy"
    ]
    insufficient_depth = competition_metrics(
        analyse_bench_press_landmarks(insufficient_frames, fps=30.0)
    )["observed_side_elbow_depth_2d_proxy"]

    assert sufficient_depth["classification"] == "DEPTH_SUFFICIENT"
    assert insufficient_depth["classification"] == "DEPTH_INSUFFICIENT"
    assert all(
        side["support_sufficient"]
        for side in insufficient_depth["measurement"]["per_side_support"].values()
    )


def test_conflicting_supported_depth_measurements_are_uncertain():
    frames = landmark_repetition()
    for index in range(9, 17):
        elbow_y = 0.52 if index % 2 else 0.42
        for side in ("left", "right"):
            frames[index][f"{side}_elbow"]["y"] = elbow_y

    depth = competition_metrics(analyse_bench_press_landmarks(frames, fps=30.0))[
        "observed_side_elbow_depth_2d_proxy"
    ]

    assert depth["classification"] == "UNCERTAIN"
    assert all(
        side["contradictory_evidence"]
        for side in depth["measurement"]["per_side_support"].values()
    )


def test_missing_bottom_pairs_are_explicitly_insufficient():
    frames = landmark_repetition()
    for frame in frames[9:17]:
        for side in ("left", "right"):
            frame[f"{side}_shoulder"]["visibility"] = 0.1
            frame[f"{side}_shoulder"]["presence"] = 0.1

    depth = competition_metrics(analyse_bench_press_landmarks(frames, fps=30.0))[
        "observed_side_elbow_depth_2d_proxy"
    ]

    assert depth["classification"] == "INSUFFICIENT_EVIDENCE"
    assert depth["measurement"]["evidence_frames"] == {"left": [], "right": []}


def test_depth_temporal_support_is_equivalent_at_different_fps():
    classifications = []
    elapsed_spans = []
    for fps, bottom_frames in ((15.0, 4), (30.0, 8), (60.0, 16)):
        result = analyse_bench_press_landmarks(
            landmark_repetition(bottom_frames=bottom_frames), fps=fps
        )
        depth = competition_metrics(result)["observed_side_elbow_depth_2d_proxy"]
        classifications.append(depth["classification"])
        elapsed_spans.append(
            depth["measurement"]["per_side_support"]["left"]["elapsed_evidence_seconds"]
        )

    assert classifications == ["DEPTH_SUFFICIENT"] * 3
    assert all(
        span >= BenchAnalysisConfig().minimum_elbow_depth_evidence_seconds
        for span in elapsed_spans
    )


def test_positive_checks_are_recognised_and_receive_maintenance_cue():
    feedback = build_template_feedback(
        analysis("DEPTH_SUFFICIENT", "MOTIONLESS_BOTTOM_OBSERVED", "COMPLETE")
    )
    assert len(feedback["positive_observations"]) == 2
    assert "Repeat the same depth" in feedback["overall_assessment"]
    assert feedback["priority_cues"] == []


def test_corrective_checks_use_exact_deterministic_cues_and_priority_order():
    feedback = build_template_feedback(
        analysis("DEPTH_INSUFFICIENT", "NO_MOTIONLESS_BOTTOM", "INCOMPLETE")
    )
    assert [cue["next_rep_cue"] for cue in feedback["priority_cues"]] == [
        "Lower the bar until your observed elbow reaches at least shoulder level.",
        "Finish with your elbows fully extended and hold the top position.",
    ]


def test_mixed_result_reports_success_and_only_the_detected_correction():
    feedback = build_template_feedback(
        analysis("DEPTH_SUFFICIENT", "MOTIONLESS_BOTTOM_OBSERVED", "INCOMPLETE")
    )
    assert len(feedback["positive_observations"]) == 1
    assert feedback["priority_cues"][0]["next_rep_cue"] == (
        "Finish with your elbows fully extended and hold the top position."
    )


def test_uncertain_check_is_a_limitation_not_a_fault():
    feedback = build_template_feedback(
        analysis("UNCERTAIN", "MOTIONLESS_BOTTOM_OBSERVED", "COMPLETE")
    )
    assert feedback["priority_cues"] == []
    assert any("uncertain" in item for item in feedback["limitations"])


def test_one_sided_depth_evidence_is_not_coached_as_a_fault():
    feedback = build_template_feedback(
        analysis("INSUFFICIENT_EVIDENCE", "MOTIONLESS_BOTTOM_OBSERVED", "COMPLETE")
    )
    assert feedback["priority_cues"] == []
    assert any(
        "not supported by sufficient evidence" in item
        for item in feedback["limitations"]
    )


def test_local_model_prose_cannot_replace_deterministic_bench_outcome_or_cue():
    source = analysis("DEPTH_INSUFFICIENT", "MOTIONLESS_BOTTOM_OBSERVED", "COMPLETE")
    brief = build_actionable_coaching_brief(source)
    merged = _merge_llm_prose_with_brief(
        "Everything was perfect.",
        {
            "BP-DEPTH": {
                "what_was_observed": "No depth issue occurred.",
                "why_focus_on_this": "Model prose may explain this focus without changing its outcome.",
            }
        },
        brief,
    )
    assert merged["overall_assessment"] == brief["overall_summary"]
    assert (
        merged["priority_cues"][0]["what_was_observed"]
        == brief["priority_themes"][0]["what_was_observed"]
    )
    assert (
        merged["priority_cues"][0]["next_rep_cue"]
        == "Lower the bar until your observed elbow reaches at least shoulder level."
    )


def competition_metrics(result: dict) -> dict[str, dict]:
    metrics = {metric["metric_id"]: metric for metric in result["metric_results"]}
    pause = (
        result.get("experimental_diagnostics", {})
        .get("motionless_bottom_2d_proxy", {})
        .get("result")
    )
    if pause:
        metrics[pause["metric_id"]] = pause
    return metrics


def test_touch_and_go_has_bilateral_motion_evidence_but_no_motionless_bottom():
    result = analyse_bench_press_landmarks(
        landmark_repetition(bottom_frames=1), fps=30.0
    )
    pause = competition_metrics(result)["motionless_bottom_2d_proxy"]
    assert pause["classification"] == "NO_MOTIONLESS_BOTTOM"
    assert pause["measurement"]["bilateral_channels_reliable"] == {
        "wrist": True,
        "elbow": True,
    }
    assert pause["measurement"]["longest_stable_interval"]["duration_seconds"] < 0.2
    assert (
        "Ordered descent was followed by ascent"
        in pause["measurement"]["classification_reason"]
    )
    pause_only_failure = analysis(
        "DEPTH_SUFFICIENT", "NO_MOTIONLESS_BOTTOM", "COMPLETE"
    )
    feedback = build_template_feedback(pause_only_failure)
    assert feedback["priority_cues"] == []
    assert len(feedback["positive_observations"]) == 2
    assert all(
        "motionless" not in item.lower() for item in feedback["positive_observations"]
    )
    assert all("pause" not in item.lower() for item in feedback["limitations"])
    payload = build_feedback_payload(pause_only_failure)
    assert "motionless_bottom_2d_proxy" not in json.dumps(payload)

    uncertain_pause = build_template_feedback(
        analysis("DEPTH_SUFFICIENT", "INSUFFICIENT_EVIDENCE", "COMPLETE")
    )
    assert uncertain_pause["priority_cues"] == []
    assert len(uncertain_pause["positive_observations"]) == 2
    assert all(
        "motionless" not in item.lower() for item in uncertain_pause["limitations"]
    )
    assert all("pause" not in item.lower() for item in uncertain_pause["limitations"])


def test_touch_and_go_export_retains_canonical_competition_assessments(tmp_path):
    frames = landmark_repetition(bottom_frames=1)
    for frame in frames[8:11]:
        frame["left_elbow"]["y"] = 0.52
        frame["right_elbow"]["y"] = 0.52
    for frame in frames[-8:]:
        frame["left_elbow"]["x"] = 0.55
        frame["right_elbow"]["x"] = 0.55

    result = analyse_bench_press_landmarks(frames, fps=30.0)
    output_path = tmp_path / "bench-analysis.json"
    save_analysis_json(result, output_path)
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert exported["analysis_status"] == result["analysis_status"]
    assert exported["analysis_valid"] is True
    assert {fault["fault_id"] for fault in exported["detected_faults"]} == {
        "incomplete_elbow_extension",
    }
    assessments = exported["competition_rule_assessments"]
    assert (
        assessments["bilateral_elbow_depth"]["classification"]
        == "INSUFFICIENT_EVIDENCE"
    )
    assert "bilateral_motionless_bottom" not in assessments
    assert assessments["bilateral_terminal_extension"]["classification"] == "INCOMPLETE"
    for assessment in assessments.values():
        assert assessment["evidence_status"]
        assert assessment["threshold"] is not None
        assert assessment["source_frames"]
        assert assessment["landmarks_used"]
        assert assessment["landmark_state"]
        assert assessment["landmark_origin"]
        assert assessment["measurement"]["classification_reason"]
    experimental = exported["experimental_diagnostics"]["motionless_bottom_2d_proxy"]
    assert experimental["scope"] == "experimental_diagnostic_only"
    assert experimental["coaching_eligible"] is False
    assert experimental["result"]["classification"] == "NO_MOTIONLESS_BOTTOM"


def test_genuine_pause_persists_bilateral_signals_and_fps_aware_duration():
    result = analyse_bench_press_landmarks(landmark_repetition(), fps=30.0)
    pause = competition_metrics(result)["motionless_bottom_2d_proxy"]
    measurement = pause["measurement"]
    assert pause["classification"] == "MOTIONLESS_BOTTOM_OBSERVED"
    assert measurement["longest_stable_interval"]["duration_frames"] == 7
    assert measurement["longest_stable_interval"]["duration_seconds"] == 7 / 30.0
    assert measurement["phase_duration_seconds"] == (
        measurement["phase_duration_frames"] / 30.0
    )
    assert set(measurement["per_side_coverage"]) == {"left", "right"}
    assert all(
        measurement["longest_stable_interval"]["per_side_measurements"][side][
            "elbow_angle_change_degrees"
        ]
        == 0.0
        for side in ("left", "right")
    )


def test_pause_diagnostic_traces_processed_bilateral_windows_and_phase_boundaries():
    result = analyse_bench_press_landmarks(landmark_repetition(), fps=30.0)
    pause = competition_metrics(result)["motionless_bottom_2d_proxy"]
    measurement = pause["measurement"]
    diagnostic = measurement["pause_diagnostics"]

    assert (
        measurement["candidate_frame_range"]["start_frame"]
        == result["phase_frames"]["bottom"]["start_frame"]
    )
    assert (
        measurement["candidate_frame_range"]["elapsed_seconds"]
        == (
            measurement["candidate_frame_range"]["end_frame"]
            - measurement["candidate_frame_range"]["start_frame"]
        )
        / 30.0
    )
    assert diagnostic["phase_selection"] == {
        "selected_side": result["selected_side"],
        "absolute_bottom_frame": result["diagnostics"]["phase_detection"][
            "bottom_frame"
        ],
        "bottom_band_boundary_y": result["diagnostics"]["phase_detection"][
            "bottom_band_boundary_y"
        ],
        "ascent_start_criterion_y": result["diagnostics"]["phase_detection"][
            "ascent_start_criterion_y"
        ],
        "ascent_start_frame": result["phase_frames"]["ascent"]["start_frame"],
    }
    assert set(diagnostic["arm_scales"]) == {"left", "right"}
    row = diagnostic["transition_table"][0]
    assert all(
        f"{side}_{signal}_{kind}" in row
        for side in ("left", "right")
        for signal in ("wrist", "elbow", "elbow_angle")
        for kind in ("raw", "smoothed")
    )
    assert (
        diagnostic["transition_frame_range"][
            "included_smoothing_context_frames_each_side"
        ]
        == 2
    )
    assert (
        diagnostic["tested_window_count"]
        == len(diagnostic["tested_candidate_windows"])
        + diagnostic["tested_windows_omitted"]
    )
    window = diagnostic["tested_candidate_windows"][0]
    assert set(window["by_side"]) == {"left", "right"}
    assert diagnostic["thresholds"]["position_range_body_scale"] > 0


def test_compact_pause_export_is_bounded_and_does_not_change_analysis_outcomes():
    result = analyse_bench_press_landmarks(landmark_repetition(), fps=30.0)
    classifications_before = {
        metric["metric_id"]: metric["classification"]
        for metric in result["metric_results"]
    }

    exported = bench_pause_diagnostics_export(result)

    assert exported["schema"] == "bench_pause_diagnostics_v1"
    assert exported["scope"] == "experimental_diagnostic_only"
    assert exported["coaching_eligible"] is False
    assert "metric_results" not in exported
    assert "detected_faults" not in exported
    assert exported["classification"] == "MOTIONLESS_BOTTOM_OBSERVED"
    assert len(exported["tested_candidate_windows"]) <= exported["window_detail_limit"]
    assert (
        len(exported["transition_table"])
        <= exported["transition_frame_range"]["detail_limit"]
    )
    assert classifications_before == {
        metric["metric_id"]: metric["classification"]
        for metric in result["metric_results"]
    }


def test_reliably_observed_unilateral_failure_is_corrective_for_bilateral_rules():
    frames = landmark_repetition()
    for index, frame in enumerate(frames):
        if 9 <= index <= 16:
            frame["right_elbow"]["y"] = 0.45
        if index >= 21:
            frame["right_elbow"]["x"] = 0.65
    metrics = competition_metrics(analyse_bench_press_landmarks(frames, fps=30.0))
    depth = metrics["observed_side_elbow_depth_2d_proxy"]
    extension = metrics["terminal_elbow_extension_2d_proxy"]
    assert depth["classification"] == "DEPTH_INSUFFICIENT"
    assert depth["measurement"]["side_classifications"]["right"] == "DEPTH_INSUFFICIENT"
    assert extension["classification"] == "INCOMPLETE"
    assert extension["measurement"]["side_classifications"]["right"] == "INCOMPLETE"


def test_unilateral_missing_evidence_with_no_failure_is_insufficient():
    metrics = competition_metrics(
        analyse_bench_press_landmarks(
            landmark_repetition(right_visible=False), fps=30.0
        )
    )
    assert (
        metrics["observed_side_elbow_depth_2d_proxy"]["classification"]
        == "INSUFFICIENT_EVIDENCE"
    )
    assert (
        metrics["motionless_bottom_2d_proxy"]["classification"]
        == "INSUFFICIENT_EVIDENCE"
    )
    assert (
        metrics["terminal_elbow_extension_2d_proxy"]["classification"]
        == "INSUFFICIENT_EVIDENCE"
    )


def test_bilateral_missing_evidence_withholds_all_competition_proxies():
    frames = landmark_repetition()
    for frame in frames:
        for landmark in frame.values():
            landmark["visibility"] = 0.1
            landmark["presence"] = 0.1
    metrics = competition_metrics(analyse_bench_press_landmarks(frames, fps=30.0))
    assert all(
        metrics[metric_id]["classification"] == "INSUFFICIENT_EVIDENCE"
        for metric_id in (
            "observed_side_elbow_depth_2d_proxy",
            "motionless_bottom_2d_proxy",
            "terminal_elbow_extension_2d_proxy",
        )
    )
