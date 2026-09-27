
from __future__ import annotations

import csv
import html
import json
import math
import sys
import threading
import uuid
from pathlib import Path

import gradio as gr

from .bench_analysis import (
    BENCH_LANDMARK_CONFIGURATION,
    BENCH_OVERLAY_CONNECTIONS,
    BenchAnalysisConfig,
    analyse_bench_press_landmarks,
    bench_diagnostics_view,
    bench_pause_diagnostics_export,
)
from .coaching_rule_library import COACHING_RULES, build_actionable_coaching_brief
from .coaching_sources import SOURCE_TYPE_LABELS
from .deadlift_analysis import (
    DEADLIFT_LANDMARK_CONFIGURATION,
    DEADLIFT_OVERLAY_CONNECTIONS,
    DeadliftAnalysisConfig,
    analyse_deadlift_landmarks,
    deadlift_diagnostics_view,
)
from .motion_analysis import (
    SQUAT_LANDMARKS,
    SQUAT_OVERLAY_CONNECTIONS,
    SquatAnalysisConfig,
    analyse_squat_landmarks,
    save_analysis_json,
)
from .pose_extraction import extract_pose, render_pose_overlay
from .llm_feedback import (
    LocalModelConfig,
    generate_local_feedback,
    test_local_model,
)
from .model_assets import POSE_MODEL_PATH, valid_pose_model

DEFAULT_MODEL_PATH = POSE_MODEL_PATH
HEAVY_MODEL_PATH = Path("models/pose_landmarker_heavy.task")
POSE_MODEL_PATHS = {
    "Full": DEFAULT_MODEL_PATH,
    "Heavy": HEAVY_MODEL_PATH,
}
DEFAULT_OUTPUT_PATH = Path("output/annotated_squat.mp4")
DEFAULT_ANALYSIS_PATH = Path("output/squat_analysis.json")
DEFAULT_LANDMARKS_PATH = Path("output/landmark_timeseries.json")
DEFAULT_BENCH_OUTPUT_PATH = Path("output/annotated_bench_press.mp4")
DEFAULT_BENCH_ANALYSIS_PATH = Path("output/bench_press_analysis.json")
DEFAULT_BENCH_LANDMARKS_PATH = Path("output/bench_press_landmark_timeseries.json")
DEFAULT_BENCH_PAUSE_DIAGNOSTICS_PATH = Path("output/bench_pause_diagnostics.json")
DEFAULT_DEADLIFT_OUTPUT_PATH = Path("output/annotated_deadlift.mp4")
DEFAULT_DEADLIFT_ANALYSIS_PATH = Path("output/deadlift_analysis.json")
DEFAULT_DEADLIFT_LANDMARKS_PATH = Path("output/deadlift_landmark_timeseries.json")

PHASE_TABLE_HEADERS = [
    "Phase", "Start frame", "End frame", "Start (s)", "End (s)", "Duration (s)"
]
PHASE_ANGLE_HEADERS = ["Phase / position", "Angle evidence"]
SQUAT_PHASE_ANGLE_HEADERS = [
    "Phase", "Knee angle (°)", "Hip angle (°)", "Torso inclination from vertical (°)"
]
BENCH_PHASE_ANGLE_HEADERS = [
    "Phase / position", "Measurement", "Observed side", "Value (°)",
    "Source field",
]
DEADLIFT_PHASE_ANGLE_HEADERS = [
    "Phase boundary", "Projected knee angle (°)",
    "Projected trunk–thigh angle (°)", "Torso inclination from vertical (°)",
]

LIGHT_THEME = gr.themes.Soft().set(
    body_background_fill="#fbfbfa",
    body_background_fill_dark="#fbfbfa",
    body_text_color="#20242a",
    body_text_color_dark="#20242a",
    body_text_color_subdued="#5b6470",
    body_text_color_subdued_dark="#5b6470",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#fbfbfa",
    background_fill_secondary_dark="#fbfbfa",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#e5e7eb",
    block_border_color_dark="#e5e7eb",
    block_label_background_fill="#f8fafc",
    block_label_background_fill_dark="#f8fafc",
    block_label_text_color="#374151",
    block_label_text_color_dark="#374151",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#d1d5db",
    input_border_color_dark="#d1d5db",
    input_text_size="1rem",
    panel_background_fill="#ffffff",
    panel_background_fill_dark="#ffffff",
    panel_border_color="#e5e7eb",
    panel_border_color_dark="#e5e7eb",
    table_even_background_fill="#ffffff",
    table_even_background_fill_dark="#ffffff",
    table_odd_background_fill="#f9fafb",
    table_odd_background_fill_dark="#f9fafb",
    table_text_color="#20242a",
    table_text_color_dark="#20242a",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#ffffff",
    button_secondary_background_fill_hover="#f3f4f6",
    button_secondary_background_fill_hover_dark="#f3f4f6",
    button_secondary_border_color="#d1d5db",
    button_secondary_border_color_dark="#d1d5db",
    button_secondary_text_color="#20242a",
    button_secondary_text_color_dark="#20242a",
)

CUSTOM_CSS = """
:root,
body,
.gradio-container {
  color-scheme: light !important;
}
.gradio-container,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.feedback-root {
  font-family:
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    Helvetica,
    Arial,
    sans-serif !important;
}
.gradio-container {
  width: 100% !important;
  max-width: 1180px !important;
  margin: 0 auto !important;
  background: #fbfbfa !important;
  color: #20242a !important;
}
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container table {
  background: #ffffff !important;
  color: #20242a !important;
  border-color: #e5e7eb !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: #ffffff !important;
  color: #20242a !important;
  border-color: #d1d5db !important;
}
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .prose,
.gradio-container .markdown {
  color: #20242a !important;
}
#upload-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 0 16px;
  border-bottom: 1px solid #e5e7eb;
}
#video-upload-button button {
  border-radius: 5px !important;
  min-height: 34px !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}
#selected-video-name input {
  color: #4b5563 !important;
  font-size: 0.92rem !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
#analyse-button button {
  border-radius: 5px !important;
  min-height: 34px !important;
  box-shadow: none !important;
}
#analyse-button button:focus-visible,
.gradio-container button:focus-visible,
.gradio-container input:focus-visible {
  outline: 3px solid #2563eb !important;
  outline-offset: 2px;
}
.upload-helper {
  color: #6b7280;
  font-size: 0.9rem;
  margin: 0 0 18px;
}
#coaching-feedback-panel {
  margin-top: 18px;
  margin-bottom: 22px;
  padding: 22px 0;
  border-top: 1px solid #d9dde3;
  border-bottom: 1px solid #d9dde3;
  border-radius: 0;
  background: #ffffff;
  box-shadow: none !important;
}
.feedback-root {
  color: #20242a;
  line-height: 1.55;
  padding: 0 22px;
}
.feedback-root h2 {
  font-size: 1.35rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  margin: 0 0 18px;
}
.feedback-root h3 {
  font-size: 1rem;
  font-weight: 650;
  margin: 0 0 6px;
}
.feedback-root p {
  margin: 0;
}
.feedback-status {
  font-size: 0.9rem;
  color: #5b6470;
  margin: 0 0 8px;
}
.feedback-row,
.assessment-section,
.positive-section,
.limitations-section {
  border-top: 1px solid #e5e7eb;
  padding: 14px 0;
}
.feedback-row span,
.evidence-details summary {
  display: block;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: #5b6470;
  margin-bottom: 5px;
}
.cue-section {
  border-top: 1px solid #d9dde3;
  padding: 18px 0 4px;
}
.cue-number {
  color: #5b6470;
  font-size: 0.9rem;
  font-weight: 700;
}
.next-cue-section {
  border-top: 1px solid #d9dde3;
  border-bottom: 1px solid #d9dde3;
  padding: 14px 0;
  margin: 16px 0;
}
.next-cue-section span {
  display: block;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: #5b6470;
  margin-bottom: 5px;
}
.next-cue-section p {
  font-size: 1.05rem;
  font-weight: 600;
  color: #1f2933;
}
.evidence-details {
  border-top: 1px solid #e5e7eb;
  padding: 14px 0;
}
.evidence-details ul,
.positive-section ul,
.limitations-section ul {
  margin: 8px 0 0 18px;
  padding: 0;
}
@media (max-width: 700px) {
  #upload-controls { flex-direction: column; align-items: stretch; }
  .gradio-container { padding: 12px !important; }
}
"""


def allow_large_csv_fields() -> None:
    """Raise Python's CSV field limit so Gradio flagging can log large media rows."""

    max_field_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_field_size)
            return
        except OverflowError:
            max_field_size //= 10


def _format_angle_cell(stats: dict) -> str:
    if not stats or stats.get("median") is None:
        return "—"
    return f"{stats['median']} [{stats['min']}–{stats['max']}]"


def _analysis_summary(analysis: dict) -> dict:
    rep = analysis.get("rep", {})
    return {
        "lift": analysis.get("lift", "squat"),
        "analysis_status": analysis.get("analysis_status", "invalid"),
        "selected_side": analysis.get("selected_side"),
        "visibility": analysis.get("visibility", {}),
        "processing_information": analysis.get("processing_information", {}),
        "warnings": analysis.get("warnings", []),
        "reps_detected": analysis.get("reps_detected", 0),
        "depth_assessment": rep.get(
            "depth_assessment", analysis.get("depth_assessment")
        ),
        "bottom_timestamp_seconds": analysis.get("bottom_timestamp_seconds"),
        "confidence": analysis.get("quality", {}).get(
            "confidence", analysis.get("confidence", "unknown")
        ),
        "quality": analysis.get("quality", {}),
        "derived_metrics": rep.get("derived_metrics", {}),
        "measurements": analysis.get("measurements", {}),
        "metric_results": analysis.get("metric_results", []),
    }


def _metric_result(analysis: dict, metric_id: str) -> dict:
    return next(
        (
            result
            for result in analysis.get("metric_results", [])
            if result.get("metric_id") == metric_id
        ),
        {},
    )


def _longest_missing_run(frames: list[dict], landmark: str) -> int:
    longest = current = 0
    for frame in frames:
        if landmark in frame.get("available_landmarks", []):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _squat_diagnostics(
    analysis: dict, camera_view: str
) -> tuple[dict, dict, dict, dict, str]:
    processing = analysis.get("processing_information", {})
    frames = analysis.get("frames", [])
    pose_counts = processing.get("pose_counts_by_frame", [])
    total = int(processing.get("total_processed_frames", len(frames)) or 0)
    frames_with_pose = sum(int(count) > 0 for count in pose_counts)
    selected_side = analysis.get("selected_side")
    selected_landmarks = {
        base: f"{selected_side}_{base}" if selected_side else None
        for base in ("shoulder", "hip", "knee", "ankle")
    }
    recording = {
        "manual_camera_view": camera_view,
        "camera_view_note": "Manual evaluation metadata only; it is not inferred or used by analysis.",
        "total_decoded_frames": total,
        "frames_with_pose": frames_with_pose if pose_counts else None,
        "pose_coverage_percentage": (
            round(frames_with_pose / total * 100, 1) if pose_counts and total else None
        ),
        "multi_pose_frame_count": sum(int(count) > 1 for count in pose_counts),
        "multi_pose_count_note": (
            "The normal extractor requests one pose, so this retained count cannot reveal "
            "additional people beyond that configured limit."
        ),
        "selected_side": selected_side,
        "left_side_score": analysis.get("visibility", {}).get("left_score"),
        "right_side_score": analysis.get("visibility", {}).get("right_score"),
        "selected_side_landmarks": {
            base: {
                "available_frames": (
                    sum(
                        name in frame.get("available_landmarks", []) for frame in frames
                    )
                    if name
                    else 0
                ),
                "availability_percentage": (
                    round(
                        sum(
                            name in frame.get("available_landmarks", [])
                            for frame in frames
                        )
                        / len(frames)
                        * 100,
                        1,
                    )
                    if name and frames
                    else None
                ),
                "longest_consecutive_dropout_frames": (
                    _longest_missing_run(frames, name) if name else None
                ),
            }
            for base, name in selected_landmarks.items()
        },
        "landmark_scores": analysis.get("visibility", {}).get("landmark_scores", {}),
    }

    phase_frames = analysis.get("phase_frames", {})
    segmentation = analysis.get("segmentation_debug", {})
    phase = {
        "phase_detection_succeeded": bool(analysis.get("reps_detected")),
        "phase_failure_reason": segmentation.get("failure_reason"),
        "descent_start_frame": phase_frames.get("descent", {}).get("start_frame"),
        "detected_bottom_frame": analysis.get("bottom_frame"),
        "ascent_start_frame": phase_frames.get("ascent", {}).get("start_frame"),
        "final_standing_start_frame": phase_frames.get("final_standing", {}).get(
            "start_frame"
        ),
        "completion_frame": phase_frames.get("final_standing", {}).get("end_frame"),
        "hip_travel": segmentation.get("maximum_depth_from_baseline"),
        "state_machine_debug": segmentation,
    }

    depth = _metric_result(analysis, "squat_depth_2d_proxy")
    depth_measurement = depth.get("measurement") or {}
    depth_diagnostics = {
        "notice": "2D depth proxy only; the numeric boundary is legacy prototype calibration, not an official judging decision.",
        "evidence_status": depth.get("evidence_status"),
        "classification": depth.get("classification"),
        "bottom_source_frames": depth.get("source_frames", []),
        "bottom_hip_median": depth_measurement.get("bottom_hip_y_median"),
        "bottom_knee_median": depth_measurement.get("bottom_knee_y_median"),
        "hip_minus_knee_measurement": depth_measurement.get("hip_minus_knee_y"),
        "current_depth_margin": (depth.get("threshold") or {}).get("depth_margin"),
        "distance_from_boundary": depth.get("distance_from_boundary"),
        "landmark_evidence": depth.get("landmark_quality", {}),
        "landmark_state": depth.get("landmark_state", {}),
        "observational_reliability": depth.get("reliability", {}),
        "warnings": depth.get("warnings", []),
        "sensitivity_result": depth.get("sensitivity_result"),
        "provenance": {
            "landmarks_used": depth.get("landmarks_used", []),
            "landmark_origin": depth.get("landmark_origin", {}),
            "normalisation": depth.get("normalisation"),
            "decision_rule": depth.get("decision_rule"),
            "threshold": depth.get("threshold"),
            "reliability": depth.get("reliability", {}),
        },
    }

    coordination = _metric_result(analysis, "squat_early_ascent_coordination")
    coordination_measurement = coordination.get("measurement") or {}
    coordination_diagnostics = {
        "evidence_status": coordination.get("evidence_status"),
        "classification": coordination.get("classification"),
        "ascent_frame_range": coordination_measurement.get("ascent_frame_range"),
        "early_ascent_frame_range": coordination_measurement.get(
            "early_ascent_frame_range"
        ),
        "maximum_hip_minus_shoulder_progress_difference": coordination_measurement.get(
            "maximum_progress_difference"
        ),
        "qualifying_duration_seconds": coordination_measurement.get(
            "qualifying_duration_seconds"
        ),
        "current_threshold": coordination.get("threshold"),
        "source_frames": coordination.get("source_frames", []),
        "landmark_evidence": coordination.get("landmark_quality", {}),
        "landmark_state": coordination.get("landmark_state", {}),
        "observational_reliability": coordination.get("reliability", {}),
        "warnings": coordination.get("warnings", []),
        "sensitivity_result": coordination.get("sensitivity_result"),
        "provenance": {
            "landmarks_used": coordination.get("landmarks_used", []),
            "landmark_origin": coordination.get("landmark_origin", {}),
            "normalisation": coordination.get("normalisation"),
            "decision_rule": coordination.get("decision_rule"),
            "reliability": coordination.get("reliability", {}),
        },
    }
    return (
        recording,
        phase,
        depth_diagnostics,
        coordination_diagnostics,
        _sensitivity_tables(depth, coordination),
    )


def _sensitivity_tables(depth: dict, coordination: dict) -> str:
    def table(title: str, headers: list[str], rows: list[list[object]]) -> str:
        header = "".join(f"<th>{value}</th>" for value in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
            for row in rows
        )
        return f"<h4>{title}</h4><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    depth_sweep = (
        (depth.get("sensitivity_result") or {}).get("parameter_sweeps") or {}
    ).get("depth_margin", {})
    coordination_sweeps = (coordination.get("sensitivity_result") or {}).get(
        "parameter_sweeps"
    ) or {}
    if not depth_sweep and not coordination_sweeps:
        return "<p>Sensitivity analysis was not performed because metric evidence was insufficient.</p>"
    parts = [
        "<p><strong>Experimental sensitivity sweep values only.</strong> Production thresholds and classifications remain authoritative; these values are not calibration data.</p>"
    ]
    if depth_sweep:
        parts.append(
            table(
                "Depth margin sensitivity",
                [
                    "Depth margin",
                    "Measurement",
                    "Hypothetical classification",
                    "Distance from boundary",
                    "Production value",
                ],
                [
                    [
                        row.get("value"),
                        row.get("measurement"),
                        row.get("classification"),
                        row.get("distance_from_relevant_boundary"),
                        "Yes" if row.get("is_production_value") else "No",
                    ]
                    for row in depth_sweep.get("outcomes", [])
                ],
            )
        )
    for key, title in (
        ("progress_difference", "Progress-difference sensitivity"),
        ("minimum_duration_seconds", "Minimum-duration sensitivity"),
    ):
        sweep = coordination_sweeps.get(key, {})
        if sweep:
            parts.append(
                table(
                    title,
                    [
                        "Value",
                        "Threshold",
                        "Maximum observed difference",
                        "Qualifying duration",
                        "Duration requirement",
                        "Hypothetical classification",
                        "Production value",
                    ],
                    [
                        [
                            row.get("value"),
                            row.get("threshold"),
                            row.get("maximum_observed_progress_difference"),
                            row.get("qualifying_duration_seconds"),
                            row.get("duration_requirement_seconds"),
                            row.get("classification"),
                            "Yes" if row.get("is_production_value") else "No",
                        ]
                        for row in sweep.get("outcomes", [])
                    ],
                )
            )
    return "".join(parts)


def _calibration_diagnostics(
    analysis: dict,
) -> tuple[dict, list[list[object]], dict, list[list[object]]]:
    calibration = analysis.get("kinematic_self_calibration", {})
    profile = calibration.get("profile", {})

    def selected_statistics(profiles: dict, fields: tuple[str, ...]) -> dict:
        return {
            name: {field: values.get(field) for field in fields}
            for name, values in profiles.items()
        }

    summary = {
        "notice": calibration.get("interpretation"),
        "diagnostic_only": calibration.get("diagnostic_only", True),
        "usage": calibration.get("usage", {}),
        "frames_analysed": profile.get("source_frame_count", 0),
        "frames_used_for_calibration": profile.get("calibration_frame_count", 0),
        "calibration_frames_used": profile.get("calibration_frames_used", []),
        "frames_excluded_due_to_existing_quality_requirements": profile.get(
            "frames_excluded_due_to_existing_quality_requirements", []
        ),
        "selected_side": profile.get("selected_side"),
        "body_scale": {
            "method": profile.get("body_scale_method"),
            "value": profile.get("body_scale"),
        },
        "warnings": profile.get("warnings", []),
        "provenance": profile.get("provenance", {}),
        "segment_profiles": selected_statistics(
            profile.get("segment_profiles", {}),
            ("sample_count", "median", "mad", "p05", "p95"),
        ),
        "motion_profiles": selected_statistics(
            profile.get("motion_profiles", {}),
            ("sample_count", "median", "mad", "p95", "maximum"),
        ),
        "segment_change_profiles": selected_statistics(
            profile.get("segment_change_profiles", {}),
            ("sample_count", "median", "mad", "p05", "p95", "maximum"),
        ),
    }
    rows = []
    for observation in calibration.get("observations", []):
        components = []
        if observation.get("motion"):
            components.append(("motion", observation["motion"]))
        components.extend(
            (f"segment:{name}", evidence)
            for name, evidence in observation.get("segments", {}).items()
        )
        components.extend(
            (f"segment_change:{name}", evidence)
            for name, evidence in observation.get("segment_changes", {}).items()
        )
        for signal, evidence in components:
            deviation = evidence.get("robust_deviation")
            if isinstance(deviation, (int, float)) and math.isfinite(deviation):
                rows.append(
                    [
                        observation.get("frame"),
                        observation.get("landmark"),
                        signal,
                        evidence.get("value"),
                        evidence.get("profile_median"),
                        evidence.get("profile_mad"),
                        deviation,
                    ]
                )
    rows.sort(key=lambda row: float(row[-1]), reverse=True)
    events = calibration.get("reacquisition_events", [])
    reacquisition_rows = []
    for event in events:
        evidence = event.get("evidence", {})
        motion = evidence.get("reacquisition_motion", {})
        segments = evidence.get("segment_consistency", {})
        segment_deviations = {
            name: values.get("robust_deviation") for name, values in segments.items()
        }
        post = evidence.get("post_reacquisition", {})
        sortable_deviations = [
            motion.get("robust_deviation"),
            *segment_deviations.values(),
        ]
        finite_deviations = [
            float(value)
            for value in sortable_deviations
            if isinstance(value, (int, float)) and math.isfinite(value)
        ]
        reacquisition_rows.append(
            [
                event.get("landmark"),
                event.get("last_valid_frame"),
                event.get("reacquired_frame"),
                event.get("gap_frame_count"),
                motion.get("robust_deviation"),
                json.dumps(segment_deviations, sort_keys=True),
                post.get("consecutive_frames_available", 0),
                max(finite_deviations, default=-1.0),
            ]
        )
    reacquisition_rows.sort(
        key=lambda row: (float(row[-1]), int(row[3] or 0)), reverse=True
    )
    for row in reacquisition_rows:
        row.pop()  # internal descriptive sort key, not an anomaly score
    reacquisition_json = {
        "notice": (
            "Reacquisition events are descriptive subject-relative evidence, not "
            "confirmed tracking errors and not inputs to biomechanical decisions."
        ),
        "definition": calibration.get("reacquisition_definition"),
        "post_reacquisition_window": calibration.get("post_reacquisition_window"),
        "events": events,
    }
    return summary, rows[:30], reacquisition_json, reacquisition_rows


def _diagnostic_line_plot(
    series: list[tuple[str, list[tuple[int, float]], str]],
    markers: list[tuple[str, int | None, str]],
    title: str,
    invert_y: bool = False,
) -> str:
    points = [(x, y) for _, values, _ in series for x, y in values if math.isfinite(y)]
    if not points:
        return (
            f"<p><strong>{html.escape(title)}</strong>: no available signal data.</p>"
        )
    width, height, left, top, right, bottom = 900, 330, 55, 35, 20, 50
    min_x, max_x = min(x for x, _ in points), max(x for x, _ in points)
    min_y, max_y = min(y for _, y in points), max(y for _, y in points)
    x_span, y_span = max(max_x - min_x, 1), max(max_y - min_y, 1e-9)

    def x_scale(x: float) -> float:
        return left + (x - min_x) / x_span * (width - left - right)

    def y_scale(y: float) -> float:
        direction = (y - min_y) / y_span if invert_y else (max_y - y) / y_span
        return top + direction * (height - top - bottom)

    parts = [
        f'<div><strong>{html.escape(title)}</strong><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}" style="width:100%;background:#fff;border:1px solid #e5e7eb">',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af"/>',
    ]
    for label, frame, color in markers:
        if frame is None or frame < min_x or frame > max_x:
            continue
        x = x_scale(frame)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="{color}" stroke-dasharray="5 4"/><text x="{x + 3:.1f}" y="{top + 12}" fill="{color}" font-size="11">{html.escape(label)} {frame}</text>'
        )
    for label, values, color in series:
        segments, current = [], []
        value_map = dict(values)
        for frame in range(min_x, max_x + 1):
            if frame in value_map and math.isfinite(value_map[frame]):
                current.append(f"{x_scale(frame):.1f},{y_scale(value_map[frame]):.1f}")
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        parts.extend(
            f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>'
            for segment in segments
        )
    legend = " · ".join(
        f'<span style="color:{color}">━ {html.escape(label)}</span>'
        for label, _, color in series
    )
    parts.append(
        f'<text x="{left}" y="{height - 18}" font-size="12">Frame {min_x}–{max_x}</text></svg><div>{legend}</div></div>'
    )
    return "".join(parts)


def _squat_diagnostic_plots(analysis: dict) -> tuple[str, str]:
    frames = analysis.get("frames", [])
    phase_frames = analysis.get("phase_frames", {})
    markers = [
        ("descent", phase_frames.get("descent", {}).get("start_frame"), "#7c3aed"),
        ("bottom", analysis.get("bottom_frame"), "#dc2626"),
        ("ascent", phase_frames.get("ascent", {}).get("start_frame"), "#059669"),
        (
            "complete",
            phase_frames.get("final_standing", {}).get("end_frame"),
            "#2563eb",
        ),
    ]
    trajectory = _diagnostic_line_plot(
        [
            (
                "raw hip y",
                [
                    (i, float(f["hip_y"]))
                    for i, f in enumerate(frames)
                    if isinstance(f.get("hip_y"), (int, float))
                ],
                "#9ca3af",
            ),
            (
                "smoothed hip y",
                [
                    (i, float(f["smoothed_hip_y"]))
                    for i, f in enumerate(frames)
                    if isinstance(f.get("smoothed_hip_y"), (int, float))
                ],
                "#dc2626",
            ),
            (
                "raw shoulder y",
                [
                    (i, float(f["shoulder_y"]))
                    for i, f in enumerate(frames)
                    if isinstance(f.get("shoulder_y"), (int, float))
                ],
                "#2563eb",
            ),
        ],
        markers,
        "Squat trajectories (image y increases downward)",
        invert_y=True,
    )
    metric_plot_parts = []
    for metric_id, title in (
        ("squat_depth_2d_proxy", "Depth evaluation-window raw coordinates"),
        (
            "squat_early_ascent_coordination",
            "Early-ascent evaluation-window raw coordinates",
        ),
    ):
        reliability = _metric_result(analysis, metric_id).get("reliability", {})
        raw = reliability.get("raw_trajectories", {})
        colours = ("#dc2626", "#2563eb", "#f59e0b", "#059669")
        coordinate_series = []
        colour_index = 0
        for landmark, values in raw.items():
            for axis in ("x", "y"):
                coordinate_series.append(
                    (
                        f"{landmark} {axis}",
                        [
                            (int(frame), float(point[axis]))
                            for frame, point in values.items()
                            if isinstance(point.get(axis), (int, float))
                        ],
                        colours[colour_index % len(colours)],
                    )
                )
                colour_index += 1
        window = reliability.get("evaluation_window", {})
        metric_plot_parts.append(
            _diagnostic_line_plot(
                coordinate_series,
                [],
                f"{title} (frames {window.get('start_frame')}–{window.get('end_frame')})",
                invert_y=True,
            )
        )
    trajectory += "".join(metric_plot_parts)
    measurement = (
        _metric_result(analysis, "squat_early_ascent_coordination").get("measurement")
        or {}
    )
    progress = _diagnostic_line_plot(
        [
            (
                "normalised hip progress",
                [
                    (int(k), float(v))
                    for k, v in measurement.get("hip_progress_by_frame", {}).items()
                ],
                "#dc2626",
            ),
            (
                "normalised shoulder progress",
                [
                    (int(k), float(v))
                    for k, v in measurement.get(
                        "shoulder_progress_by_frame", {}
                    ).items()
                ],
                "#2563eb",
            ),
        ],
        [],
        "Early-ascent normalised progress",
    )
    return trajectory, progress


def _table_value(headers: list[str], rows: list[list[object]]) -> dict:
    return {"headers": headers, "data": rows}


def _phase_timing_table(analysis: dict) -> dict:
    timings = analysis.get("rep", {}).get("phase_timings", {})
    rows = [
        [
            phase,
            data.get("start_frame"),
            data.get("end_frame"),
            data.get("start_timestamp_seconds"),
            data.get("end_timestamp_seconds"),
            data.get("duration_seconds"),
        ]
        for phase, data in timings.items()
    ]
    if not rows:
        fps = float(
            analysis.get("processing_information", {}).get("fps", 30.0) or 30.0
        )
        rows = [
            [
                phase,
                bounds.get("start_frame"),
                bounds.get("end_frame"),
                round(bounds["start_frame"] / fps, 3),
                round(bounds["end_frame"] / fps, 3),
                round((bounds["end_frame"] - bounds["start_frame"] + 1) / fps, 3),
            ]
            for phase, bounds in analysis.get("phase_frames", {}).items()
            if bounds.get("start_frame") is not None
            and bounds.get("end_frame") is not None
        ]
    return _table_value(PHASE_TABLE_HEADERS, rows)


def _phase_angle_table(analysis: dict) -> dict:
    lift = analysis.get("lift")
    if lift == "bench_press":
        return _bench_phase_angle_table(analysis)
    if lift == "deadlift":
        return _deadlift_phase_angle_table(analysis)
    if lift != "squat":
        return _table_value(PHASE_ANGLE_HEADERS, [])

    angles = analysis.get("rep", {}).get("phase_angles", {})
    rows = [
        [
            phase,
            _format_angle_cell(data.get("knee_angle", {})),
            _format_angle_cell(data.get("hip_angle", {})),
            _format_angle_cell(data.get("torso_lean_degrees", {})),
        ]
        for phase, data in angles.items()
    ]
    movement = analysis.get("measurements", {}).get("movement_angle_summary", {})

    def movement_cell(name: str) -> str:
        evidence = movement.get(name, {})
        value = evidence.get("value_degrees")
        if value is None:
            return "— (unavailable)"
        coverage = f"{evidence.get('observed_frames', 0)}/{evidence.get('interval_frames', 0)} frames"
        qualifier = "complete" if evidence.get("evidence_status") == "complete" else "available observations only"
        return f"{value}° min ({coverage}; {qualifier})"

    if movement:
        interval = movement.get("movement_interval", {})
        rows.append([
            f"movement frames {interval.get('start_frame')}–{interval.get('end_frame')}",
            movement_cell("minimum_knee_angle"), movement_cell("minimum_hip_angle"), "—",
        ])
    if not rows:
        rows = [["Unavailable", "No squat phase-angle evidence was calculated.", "—", "—"]]
    return _table_value(SQUAT_PHASE_ANGLE_HEADERS, rows)


def _measurement_result(analysis: dict, metric_id: str) -> dict:
    return next((item for item in analysis.get("metric_results", [])
                 if item.get("metric_id") == metric_id), {})


def _unavailable_angle_row(message: str, columns: int = 4) -> list[str]:
    return ["Unavailable", message, *["—"] * (columns - 2)]


def _bench_phase_angle_table(analysis: dict) -> dict:
    rows: list[list[object]] = []
    selected_side = analysis.get("selected_side") or "unavailable"
    bottom = _measurement_result(analysis, "bench_elbow_flexion_2d")
    measurements = analysis.get("measurements") or {}
    bottom_measurement = (
        bottom.get("measurement")
        or measurements.get("bench_elbow_flexion_2d")
        or {}
    )
    if bottom_measurement:
        bottom_fields = (
            ("Minimum across bottom window", "minimum_elbow_angle_degrees"),
            ("Median across bottom window", "bottom_median_elbow_angle_degrees"),
        )
        rows.extend(
            ["Bottom window", label, selected_side, bottom_measurement[field],
             "Bottom elbow flexion"]
            for label, field in bottom_fields
            if bottom_measurement.get(field) is not None
        )
    terminal = _measurement_result(analysis, "terminal_elbow_extension_2d_proxy")
    terminal_measurement = (
        terminal.get("measurement")
        or measurements.get("terminal_elbow_extension_2d_proxy")
        or {}
    )
    by_side = terminal_measurement.get("terminal_median_elbow_angle_degrees_by_side") or {}
    for side in ("left", "right"):
        if by_side.get(side) is not None:
            rows.append([
                "Returned top", "Median elbow-extension proxy", side,
                by_side[side],
                "Terminal elbow extension",
            ])
    if not rows:
        warnings = bottom.get("warnings") or terminal.get("warnings") or analysis.get("warnings") or []
        rows = [_unavailable_angle_row(
            str(warnings[0] if warnings else "No bench elbow-angle evidence was calculated."),
            len(BENCH_PHASE_ANGLE_HEADERS),
        )]
    return _table_value(BENCH_PHASE_ANGLE_HEADERS, rows)


def _deadlift_phase_angle_table(analysis: dict) -> dict:
    extension = _measurement_result(analysis, "deadlift_phase_extension_2d")
    measurements = analysis.get("measurements") or {}
    measurement = (
        extension.get("measurement")
        or measurements.get("deadlift_phase_extension_2d")
        or {}
    )
    geometry = measurement.get("boundary_geometry") or {}
    labels = {"lift_off": "lift off", "knee_passing": "knee passing", "top_position": "top position"}
    rows = [[labels[key], values.get("projected_knee_angle_degrees", "—"),
             values.get("projected_trunk_thigh_angle_degrees", "—"),
             values.get("torso_inclination_from_vertical_degrees", "—")]
            for key in labels if isinstance((values := geometry.get(key)), dict) and values]
    if not rows:
        warnings = extension.get("warnings") or analysis.get("warnings") or []
        rows = [_unavailable_angle_row(str(warnings[0] if warnings else "No deadlift boundary-angle evidence was calculated."))]
    return _table_value(DEADLIFT_PHASE_ANGLE_HEADERS, rows)


def _findings_markdown(analysis: dict) -> str:
    rep = analysis.get("rep", {})
    lines = ["### Ranked deterministic findings"]
    for finding in rep.get("findings", []):
        lines.append(
            f"* **{finding.get('name')}** ({finding.get('status')}, priority {finding.get('priority')}): {finding.get('evidence')}"
        )
    if not rep.get("findings"):
        metric_results = analysis.get("metric_results", [])
        classified = [result for result in metric_results if result.get("classification")]
        if classified:
            for result in classified:
                lines.append(f"* **{result.get('metric_id')}**: {result.get('classification')} "
                             f"(evidence: {result.get('evidence_status', 'unknown')}).")
        elif metric_results:
            lines.append("* No deterministic classification was produced; the available metrics are descriptive only.")
        else:
            lines.append("* No deterministic findings available.")
    warnings = rep.get("warnings", analysis.get("warnings", []))
    brief = build_actionable_coaching_brief(analysis)
    lines.append("\n<details><summary>Sources and rule traceability</summary>\n")
    lines.append("\n```json")
    lines.append(
        json.dumps(
            {
                "triggered_rule_ids": brief.get("triggered_rule_ids", []),
                "source_metadata": brief.get("source_metadata", []),
                "caveats": [
                    COACHING_RULES[r]["caveats"]
                    for r in brief.get("triggered_rule_ids", [])
                    if r in COACHING_RULES
                ],
                "deterministic_evidence": rep.get("findings", []),
            },
            indent=2,
        )
    )
    lines.append("```\n</details>")
    lines.append("\n### Warnings and confidence")
    lines.append(
        f"* Confidence: **{analysis.get('quality', {}).get('confidence', analysis.get('confidence', 'unknown'))}**"
    )
    (
        lines.extend(f"* {warning}" for warning in warnings)
        if warnings
        else lines.append("* No warnings.")
    )
    return "\n".join(lines)


def _prototype_metric_panels(analysis: dict, diagnostics: dict) -> tuple[dict, dict]:
    if analysis.get("lift") == "deadlift":
        evidence = diagnostics.get("metric_evidence", {})
        return (_evidence_for_display(evidence.get("deadlift_phase_extension_2d", {})),
                _evidence_for_display(evidence.get("deadlift_finish_geometry_2d", {})))
    return (_evidence_for_display(diagnostics.get("elbow_angle_evidence", {})),
            _evidence_for_display(diagnostics.get("wrist_trajectory_evidence", {})))


def _html_text(value: object, fallback: str = "") -> str:
    if value is None:
        value = fallback
    return html.escape(str(value))


def _html_list(items: list[object], empty_text: str) -> str:
    values = items or [empty_text]
    return "<ul>" + "".join(f"<li>{_html_text(item)}</li>" for item in values) + "</ul>"


_METRIC_DISPLAY_NAMES = {
    "observed_side_elbow_depth_2d_proxy": "Bench elbow depth",
    "terminal_elbow_extension_2d_proxy": "Terminal elbow extension",
    "bench_elbow_flexion_2d": "Bench elbow depth",
    "bench_wrist_trajectory_2d": "Wrist trajectory",
    "bench_finish_geometry_similarity": "Start/finish similarity",
}


def _reader_facing_limitation(value: object) -> str:
    text = str(value).strip()
    for identifier, display_name in _METRIC_DISPLAY_NAMES.items():
        text = text.replace(identifier.replace("_", " "), display_name)
        text = text.replace(identifier, display_name)
    reader_wording = {
        "Do not infer weakness.": "This video does not establish weakness.",
        "Do not infer tiredness.": "This video does not establish tiredness.",
        "Do not infer fatigue from one repetition.": (
            "One repetition does not establish fatigue."
        ),
        "Do not infer fatigue, weakness, or excessive effort from video alone.": (
            "Video alone does not establish fatigue, weakness, or excessive effort."
        ),
    }
    return reader_wording.get(text, text)


def _feedback_source_label(source: str, fallback_reason: str | None) -> str:
    if source == "local_model":
        return "Feedback source: local language model"
    if fallback_reason in {
        "local_model_schema_validation_failed",
        "local_model_response_quality_rejected",
        "local_model_unsafe_after_revision",
        "local_model_invalid_json",
    }:
        return (
            "Deterministic coaching feedback. Generated feedback was rejected "
            "by validation."
        )
    unavailable = {
        "local_model_unavailable": "the local language model could not be loaded",
        "local_model_timeout": "local generation timed out",
        "local_model_missing": "the local language model is unavailable",
        "local_model_error": "local generation failed",
    }
    if fallback_reason in unavailable:
        return (
            f"Generated feedback was unavailable because {unavailable[fallback_reason]}. "
            "Deterministic coaching is shown."
        )
    return "Deterministic coaching feedback."


def render_coaching_feedback_html(
    feedback: dict, source: str = "fallback", fallback_reason: str | None = None
) -> str:
    lines = ['<section class="feedback-root">']
    source_label = _feedback_source_label(source, fallback_reason)
    lines.append(f'<p class="feedback-status">{source_label}</p>')
    lines.append("<h2>Coaching feedback</h2>")

    overall = feedback.get("overall_assessment") or "No coaching feedback available."
    lines.extend(
        [
            '<section class="assessment-section">',
            "<h3>Overall assessment</h3>",
            f"<p>{_html_text(overall)}</p>",
            "</section>",
        ]
    )

    cues = feedback.get("priority_cues", [])
    if cues:
        for cue in cues:
            rank = cue.get("rank", "")
            title = cue.get("title", "Priority cue")
            lines.extend(
                [
                    '<section class="cue-section">',
                    f'<p class="cue-number">{_html_text(rank)}</p>',
                    f"<h3>{_html_text(title)}</h3>",
                    '<div class="feedback-row">',
                    "<span>What happened</span>",
                    f"<p>{_html_text(cue.get('what_was_observed', ''))}</p>",
                    "</div>",
                    '<div class="feedback-row">',
                    "<span>Why focus on this</span>",
                    f"<p>{_html_text(cue.get('why_focus_on_this', ''))}</p>",
                    "</div>",
                    '<div class="next-cue-section">',
                    "<span>Next rep cue</span>",
                    f"<p>{_html_text(cue.get('next_rep_cue', ''))}</p>",
                    "</div>",
                ]
            )
            if cue.get("practice_task"):
                lines.extend(
                    [
                        '<div class="feedback-row">',
                        "<span>Technique practice</span>",
                        f"<p>{_html_text(cue.get('practice_task'))}</p>",
                        "</div>",
                    ]
                )
            if cue.get("success_check"):
                lines.extend(
                    [
                        '<div class="feedback-row">',
                        "<span>What a better rep should look like</span>",
                        f"<p>{_html_text(cue.get('success_check'))}</p>",
                        "</div>",
                    ]
                )

            evidence = cue.get("evidence_and_limits", {})
            detected_pattern = evidence.get("detected_pattern") or cue.get(
                "what_was_observed"
            )
            interpretation = evidence.get("interpretation") or cue.get(
                "why_focus_on_this"
            )
            refs = [
                ref
                for ref in (evidence.get("references") or [])
                if isinstance(ref, dict)
                and any(
                    ref.get(key)
                    for key in ("short_reference", "page_or_section", "source_type")
                )
            ]
            limits = [
                _reader_facing_limitation(item)
                for item in (evidence.get("important_limits") or [])
                if str(item).strip()
            ]
            if detected_pattern or interpretation or refs or limits:
                lines.extend(
                    [
                        '<details class="evidence-details">',
                        "<summary>Evidence and limits</summary>",
                    ]
                )
                if detected_pattern:
                    lines.extend(
                        [
                            '<div class="feedback-row">',
                            "<span>Detected pattern</span>",
                            f"<p>{_html_text(detected_pattern)}</p>",
                            "</div>",
                        ]
                    )
                if interpretation:
                    lines.extend(
                        [
                            '<div class="feedback-row">',
                            "<span>Interpretation</span>",
                            f"<p>{_html_text(interpretation)}</p>",
                            "</div>",
                        ]
                    )
                if refs:
                    lines.append(
                        '<div class="feedback-row"><span>References</span><ul>'
                    )
                    for ref in refs:
                        label = SOURCE_TYPE_LABELS.get(
                            ref.get("source_type"), ref.get("source_type")
                        )
                        lines.append(
                            "<li>"
                            f"{_html_text(ref.get('short_reference', ''))}, "
                            f"{_html_text(ref.get('page_or_section', ''))} "
                            f"({_html_text(label)})."
                            "</li>"
                        )
                    lines.append("</ul></div>")
                if limits:
                    lines.append(
                        '<div class="feedback-row"><span>Important limits</span>'
                        + _html_list(limits, "None listed.")
                        + "</div>"
                    )
                lines.append("</details>")
            lines.append("</section>")
    elif feedback.get("overall_assessment"):
        lines.append(
            '<section class="cue-section"><p>No supported correction was produced. '
            "The assessment above states what the available checks could establish.</p></section>"
        )

    positives = feedback.get("positive_observations", [])
    if positives:
        lines.extend(
            ['<section class="positive-section">', "<h3>Supported positives</h3>", _html_list(positives, ""), "</section>"]
        )

    limitations = [
        _reader_facing_limitation(item)
        for item in feedback.get("limitations", [])
        if str(item).strip()
    ]
    context = [
        item for item in feedback.get("neutral_context", []) if str(item).strip()
    ]
    if limitations or context:
        lines.extend(['<section class="limitations-section">', "<h3>Specific limitations</h3>"])
        if limitations:
            lines.append(_html_list(limitations, ""))
    if context:
        lines.extend(["<h3>Additional context</h3>", _html_list(context, "")])
    if limitations or context:
        lines.append("</section>")
    lines.append("</section>")
    return "\n".join(lines)


def _feedback_markdown(feedback: dict) -> str:
    return render_coaching_feedback_html(feedback)


def _sources_markdown(analysis: dict) -> str:
    brief = build_actionable_coaching_brief(analysis)
    lines = ["### Sources / rule traceability"]
    rule_ids = brief.get("triggered_rule_ids", [])
    lines.append(
        "**Triggered rule IDs:** " + (", ".join(rule_ids) if rule_ids else "None")
    )
    metadata = brief.get("source_metadata", [])
    if metadata:
        lines.append("\n#### Source metadata")
        for source in metadata:
            label = SOURCE_TYPE_LABELS.get(
                source.get("source_type"), source.get("source_type")
            )
            pending = (
                " Bibliography details pending verification from the original source."
                if source.get("citation_complete") is False
                else ""
            )
            lines.append(
                f"* **{source.get('short_reference')}** — {source.get('page_or_section')} ({label}).{pending}"
            )
    caveats = [
        caveat
        for rule_id in rule_ids
        for caveat in COACHING_RULES.get(rule_id, {}).get("caveats", [])
    ]
    lines.append("\n#### Caveats")
    (
        lines.extend(f"* {caveat}" for caveat in caveats)
        if caveats
        else lines.append("* None listed.")
    )
    return "\n".join(lines)


def _llm_debug_summary(result: dict, config: LocalModelConfig | None = None) -> dict:
    config = config or LocalModelConfig()

    def truncate(value):
        text = str(value)
        return text if len(text) <= 3000 else text[:3000] + "…"

    summary = {
        "Configured model": config.model,
        "Model path": config.model_path,
        "Generation mode": result.get("generation_mode"),
        "Fallback reason code": result.get("fallback_reason_code"),
        "Fallback reason detail": result.get("fallback_reason_detail"),
        "Local model status": result.get("local_model_status"),
    }
    if "local_model_attempt_count" in result or "revision_attempted" in result:
        summary.update(
            {
                "Local-model attempt count": result.get("local_model_attempt_count"),
                "Revision attempted": result.get("revision_attempted"),
                "Initial quality failure codes": result.get(
                    "initial_quality_failures", []
                ),
                "Initial quality failure details": result.get(
                    "initial_quality_failure_details", []
                ),
                "Final quality failure codes": result.get("final_quality_failures", []),
                "Final quality failure details": result.get(
                    "final_quality_failure_details", []
                ),
                "Initial parsed response": truncate(
                    result.get("initial_parsed_response")
                ),
                "Revised parsed response": truncate(
                    result.get("revised_parsed_response")
                ),
            }
        )
    return summary


def run_local_model_health_check() -> dict:
    result = test_local_model(LocalModelConfig())
    return {
        "Configured model": result.get("configured_model"),
        "Model path": result.get("model_path"),
        "Generation mode": "local_model" if result.get("ok") else "fallback",
        "Fallback reason code": None if result.get("ok") else result.get("error"),
        "Generation response": result.get("response_content"),
        "Fallback reason detail": result.get("error_detail"),
    }


def _bench_phase_timing_table(analysis: dict) -> dict:
    fps = float(analysis.get("processing_information", {}).get("fps", 30.0) or 30.0)
    rows = [
        [
            phase,
            bounds.get("start_frame"),
            bounds.get("end_frame"),
            round(bounds["start_frame"] / fps, 3),
            round(bounds["end_frame"] / fps, 3),
            round((bounds["end_frame"] - bounds["start_frame"] + 1) / fps, 3),
        ]
        for phase, bounds in analysis.get("phase_frames", {}).items()
        if bounds.get("start_frame") is not None
        and bounds.get("end_frame") is not None
    ]
    return _table_value(PHASE_TABLE_HEADERS, rows)


def _evidence_for_display(evidence: dict | None) -> dict:
    displayed = dict(evidence or {})
    status = displayed.get("evidence_status")
    if status in {"insufficient", "INSUFFICIENT", "INSUFFICIENT_EVIDENCE"}:
        reasons = displayed.get("warnings") or []
        if not reasons:
            state = displayed.get("landmark_state") or displayed.get(
                "provenance", {}
            ).get("landmark_state")
            reasons = [state] if state else []
        displayed["availability"] = (
            f"Unavailable: {reasons[0]}" if reasons else "Unavailable: insufficient evidence."
        )
    return displayed


def _warnings_for_display(analysis: dict, diagnostics: dict) -> dict:
    metric_limitations = []
    for key, evidence in diagnostics.items():
        if not key.endswith("_evidence") or not isinstance(evidence, dict):
            continue
        for warning in evidence.get("warnings") or []:
            metric_limitations.append({"metric": key, "limitation": warning})
    return {
        "analysis_warnings": analysis.get("warnings", []),
        "metric_specific_limitations": metric_limitations,
    }


def _bench_trajectory_plot(diagnostics: dict) -> str:
    trajectory = diagnostics.get("trajectory", {})
    phases = diagnostics.get("phase_frames") or {}
    detection = trajectory.get("phase_detection") or {}
    if "lift_off" in phases:
        markers = [
            ("lift off", phases["lift_off"].get("start_frame"), "#7c3aed"),
            (
                "knee passing",
                phases.get("knee_passing", {}).get("start_frame"),
                "#dc2626",
            ),
            ("complete", phases.get("complete", {}).get("start_frame"), "#2563eb"),
        ]
    else:
        markers = [
            ("descent", phases.get("descent", {}).get("start_frame"), "#7c3aed"),
            ("bottom", detection.get("bottom_frame"), "#dc2626"),
            ("press", phases.get("ascent", {}).get("start_frame"), "#059669"),
            ("complete", phases.get("complete", {}).get("start_frame"), "#2563eb"),
        ]
    return _diagnostic_line_plot(
        [
            (
                "raw wrist y",
                [
                    (i, float(value))
                    for i, value in enumerate(trajectory.get("raw") or [])
                    if value is not None
                ],
                "#9ca3af",
            ),
            (
                "smoothed wrist y",
                [
                    (i, float(value))
                    for i, value in enumerate(trajectory.get("smoothed") or [])
                    if value is not None
                ],
                "#dc2626",
            ),
        ],
        markers,
        "Wrist proxy trajectory (image y increases downward)",
        invert_y=True,
    )


def lift_ui_text(lift: str) -> tuple[dict, str]:
    if lift == "Bench Press":
        return (
            gr.update(value="Analyse bench press"),
            "Observed tracking preference across 36 processed recordings: place the camera near the lifter's feet, elevated and angled slightly downward, with the whole body visible • this is a visual-tracking observation, not validated biomechanical accuracy or a universal camera rule",
        )
    if lift == "Deadlift":
        return (
            gr.update(value="Analyse deadlift"),
            "Observed tracking preference across 36 processed recordings: film conventional deadlift diagonally in front from either side, with the whole lifter and pull visible • this is a visual-tracking observation, not validated biomechanical accuracy or a universal camera rule",
        )
    return (
        gr.update(value="Analyse squat"),
        "Observed tracking preference across 36 processed recordings: film diagonally behind the lifter from either side, with the full body and repetition visible • this is a visual-tracking observation, not validated biomechanical accuracy or a universal camera rule",
    )


def _file_output(
    path: str | Path | None, artifact_name: str, *, required: bool = False
) -> str | None:
    if path is None or not str(path).strip():
        if required:
            raise gr.Error(f"Required artifact could not be produced: {artifact_name}.")
        return None

    candidate = Path(path)
    if not candidate.is_file():
        if required:
            raise gr.Error(
                f"Required artifact could not be produced: {artifact_name} "
                f"({candidate})."
            )
        return None
    return str(candidate.resolve())


def process_uploaded_video(
    video_path: str,
    camera_view: str = "FULL_SIDE",
    model_path: str = str(DEFAULT_MODEL_PATH),
    lift: str = "Squat",
) -> tuple | str:
    """Run the selected lift through shared extraction and its deterministic analyser."""

    model_path_object = Path(model_path)
    model_name = next(
        (
            name
            for name, configured_path in POSE_MODEL_PATHS.items()
            if model_path_object == configured_path
        ),
        model_path_object.stem,
    )
    if not valid_pose_model(model_path_object):
        if model_name == "Full":
            raise gr.Error(
                f"Required Pose Landmarker Full model is missing or invalid at "
                f"{model_path_object}. Run python -m powerlifting_coach.setup, then restart the app."
            )
        else:
            raise gr.Error(
                f"Optional Pose Landmarker {model_name} model is missing at "
                f"{model_path_object}. Select Full, or install the optional Heavy model manually."
            )

    # Model-specific directories retain both sets of artifacts for side-by-side
    # development review. Selecting a model changes no inference or analysis setting.
    request_id = uuid.uuid4().hex
    model_output_directory = Path("output") / model_name.lower() / request_id
    model_output_directory.mkdir(parents=True, exist_ok=False)

    dispatch = {
        "Squat": {
            "config": SquatAnalysisConfig(),
            "landmarks": SQUAT_LANDMARKS,
            "connections": SQUAT_OVERLAY_CONNECTIONS,
            "output": DEFAULT_OUTPUT_PATH,
            "analysis": DEFAULT_ANALYSIS_PATH,
            "timeseries": DEFAULT_LANDMARKS_PATH,
        },
        "Bench Press": {
            "config": BenchAnalysisConfig(),
            "landmarks": BENCH_LANDMARK_CONFIGURATION.retained_landmarks,
            "connections": BENCH_OVERLAY_CONNECTIONS,
            "output": DEFAULT_BENCH_OUTPUT_PATH,
            "analysis": DEFAULT_BENCH_ANALYSIS_PATH,
            "timeseries": DEFAULT_BENCH_LANDMARKS_PATH,
        },
        "Deadlift": {
            "config": DeadliftAnalysisConfig(),
            "landmarks": DEADLIFT_LANDMARK_CONFIGURATION.retained_landmarks,
            "connections": DEADLIFT_OVERLAY_CONNECTIONS,
            "output": DEFAULT_DEADLIFT_OUTPUT_PATH,
            "analysis": DEFAULT_DEADLIFT_ANALYSIS_PATH,
            "timeseries": DEFAULT_DEADLIFT_LANDMARKS_PATH,
        },
    }
    selected = dispatch.get(lift, dispatch["Squat"])
    config = selected["config"]
    retained_landmarks = selected["landmarks"]
    overlay_connections = selected["connections"]
    output_path = model_output_directory / selected["output"].name
    analysis_path = model_output_directory / selected["analysis"].name
    landmarks_path = model_output_directory / selected["timeseries"].name
    extraction = extract_pose(
        video_path=video_path,
        required_joints=tuple(retained_landmarks),
        visibility_threshold=config.visibility_threshold,
        annotated_video_path=output_path if lift != "Squat" else None,
        log_path=landmarks_path,
        task_model_path=model_path,
        retained_landmarks=retained_landmarks,
        overlay_connections=overlay_connections,
        return_result=True,
    )
    processing_metadata = {
        **extraction.metadata,
        "camera_view": camera_view,
        "pose_model": model_name,
        "pose_model_path": str(model_path_object),
        "request_id": request_id,
        "artifact_directory": str(model_output_directory),
    }
    if lift == "Bench Press":
        analysis = analyse_bench_press_landmarks(
            extraction.frames,
            fps=float(extraction.metadata["fps"]),
            processing_metadata=processing_metadata,
            config=config,
        )
    elif lift == "Deadlift":
        analysis = analyse_deadlift_landmarks(
            extraction.frames,
            fps=float(extraction.metadata["fps"]),
            processing_metadata=processing_metadata,
            config=config,
        )
    else:
        analysis = analyse_squat_landmarks(
            extraction.frames,
            fps=float(extraction.metadata["fps"]),
            pose_counts=extraction.metadata["pose_counts_by_frame"],
            processing_metadata=processing_metadata,
            config=config,
        )
        render_pose_overlay(
            video_path,
            output_path,
            extraction.frames,
            SQUAT_OVERLAY_CONNECTIONS,
            config.visibility_threshold,
        )
    save_analysis_json(analysis, analysis_path)
    bench_pause_path = None
    if lift == "Bench Press":
        bench_pause_diagnostics_path = (
            model_output_directory / DEFAULT_BENCH_PAUSE_DIAGNOSTICS_PATH.name
        )
        bench_pause_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        bench_pause_diagnostics_path.write_text(
            json.dumps(bench_pause_diagnostics_export(analysis), indent=2),
            encoding="utf-8",
        )
        bench_pause_path = _file_output(
            bench_pause_diagnostics_path, "bench-pause diagnostics"
        )

    input_video_output = _file_output(video_path, "uploaded video", required=True)
    annotated_video_output = _file_output(
        output_path, "annotated video", required=True
    )
    analysis_file_output = _file_output(
        analysis_path, "analysis JSON", required=True
    )
    landmarks_file_output = _file_output(
        landmarks_path, "landmark timeseries JSON", required=True
    )
    if lift in {"Bench Press", "Deadlift"}:
        is_deadlift = lift == "Deadlift"
        prototype_diagnostics = (
            deadlift_diagnostics_view(analysis)
            if is_deadlift
            else bench_diagnostics_view(analysis)
        )
        llm_result = generate_local_feedback(analysis)
        primary_evidence, secondary_evidence = _prototype_metric_panels(analysis, prototype_diagnostics)
        empty = {}
        return (
            input_video_output,
            annotated_video_output,
            input_video_output,
            annotated_video_output,
            render_coaching_feedback_html(
                llm_result.get("feedback", {}),
                llm_result.get("source", "fallback"),
                llm_result.get("fallback_reason_code"),
            ),
            f"**Pose model: {model_name}.**",
            _analysis_summary(analysis),
            _bench_phase_timing_table(analysis),
            _phase_angle_table(analysis),
            _findings_markdown(analysis),
            _sources_markdown(analysis),
            _llm_debug_summary(llm_result),
            analysis_file_output,
            empty,
            empty,
            "",
            empty,
            empty,
            "",
            "",
            empty,
            [],
            empty,
            [],
            landmarks_file_output,
            analysis,
            prototype_diagnostics,
            _bench_trajectory_plot(prototype_diagnostics),
            primary_evidence,
            secondary_evidence,
            prototype_diagnostics.get("calibration_summary", {}),
            _warnings_for_display(analysis, prototype_diagnostics),
            bench_pause_path,
            {},
            [],
        )

    llm_result = generate_local_feedback(analysis)
    annotated_video_path = annotated_video_output
    status = f"**Pose model: {model_name}.**"
    (
        recording_diagnostics,
        phase_diagnostics,
        depth_diagnostics,
        coordination_diagnostics,
        sensitivity_tables,
    ) = _squat_diagnostics(analysis, camera_view)
    (
        calibration_diagnostics,
        unusual_observations,
        reacquisition_diagnostics,
        reacquisition_events,
    ) = _calibration_diagnostics(analysis)
    trajectory_plot, coordination_plot = _squat_diagnostic_plots(analysis)
    return (
        input_video_output,
        annotated_video_path,
        input_video_output,
        annotated_video_path,
        render_coaching_feedback_html(
            llm_result.get("feedback", {}),
            llm_result.get("source", "fallback"),
            llm_result.get("fallback_reason_code"),
        ),
        status,
        _analysis_summary(analysis),
        _phase_timing_table(analysis),
        _phase_angle_table(analysis),
        _findings_markdown(analysis),
        _sources_markdown(analysis),
        _llm_debug_summary(llm_result),
        analysis_file_output,
        recording_diagnostics,
        phase_diagnostics,
        trajectory_plot,
        depth_diagnostics,
        coordination_diagnostics,
        sensitivity_tables,
        coordination_plot,
        calibration_diagnostics,
        unusual_observations,
        reacquisition_diagnostics,
        reacquisition_events,
        landmarks_file_output,
        {},
        {},
        "",
        {},
        {},
        {},
        [],
        None,
        analysis,
        _warnings_for_display(analysis, {}),
    )


def store_uploaded_video(video_path: str | None) -> tuple[str | None, str]:
    if not video_path:
        return None, "No video selected"
    return video_path, Path(video_path).name


_SESSION_REQUESTS: dict[str, dict[str, object]] = {}
_SESSION_REQUESTS_LOCK = threading.Lock()


def _session_key(request: gr.Request | None) -> str:
    return request.session_hash if request and request.session_hash else "local-session"


def _mark_selection_changed(request: gr.Request | None) -> str:
    revision = uuid.uuid4().hex
    with _SESSION_REQUESTS_LOCK:
        state = _SESSION_REQUESTS.setdefault(_session_key(request), {})
        state["revision"] = revision
    return revision


def _claim_request(request: gr.Request | None, revision: str) -> str | None:
    with _SESSION_REQUESTS_LOCK:
        state = _SESSION_REQUESTS.setdefault(_session_key(request), {})
        if state.get("active"):
            return None
        token = uuid.uuid4().hex
        state.update(active=token, revision=revision)
        return token


def _request_is_current(request: gr.Request | None, token: str, revision: str) -> bool:
    with _SESSION_REQUESTS_LOCK:
        state = _SESSION_REQUESTS.get(_session_key(request), {})
        return state.get("active") == token and state.get("revision") == revision


def _release_request(request: gr.Request | None, token: str) -> None:
    with _SESSION_REQUESTS_LOCK:
        state = _SESSION_REQUESTS.get(_session_key(request), {})
        if state.get("active") == token:
            state["active"] = None


def build_demo() -> gr.Blocks:
    allow_large_csv_fields()
    with gr.Blocks(title="AI Powerlifting Buddy", fill_width=True) as demo:
        gr.Markdown(
            "# AI Powerlifting Buddy\n"
            "Upload one repetition for a deterministic, evidence-backed technique review."
        )
        uploaded_video_path = gr.State()
        selection_revision = gr.State(uuid.uuid4().hex)

        lift_selector = gr.Dropdown(
            choices=["Squat", "Bench Press", ("Conventional Deadlift", "Deadlift")],
            value="Squat", label="Lift", interactive=True,
        )
        upload_guidance = gr.Markdown(
            "Observed tracking preference across 36 processed recordings: film diagonally behind the lifter from either side, with the full body and repetition visible • this is a visual-tracking observation, not validated biomechanical accuracy or a universal camera rule",
            elem_classes=["upload-helper"],
        )
        with gr.Row(elem_id="upload-controls"):
            video_upload_button = gr.UploadButton(
                "Choose video", file_types=["video"], file_count="single", type="filepath",
                variant="secondary", size="sm", elem_id="video-upload-button",
            )
            selected_video_name = gr.Textbox(
                value="No video selected", show_label=False, interactive=False,
                container=False, elem_id="selected-video-name",
            )
            submit = gr.Button(
                "Analyse squat", variant="primary", elem_id="analyse-button", size="lg",
                interactive=False,
            )

        with gr.Accordion("Advanced settings", open=False):
            camera_view = gr.Dropdown(
                choices=[
                    ("Side view", "FULL_SIDE"),
                    ("Front-side oblique", "FRONT_OBLIQUE"),
                    ("Rear-side oblique", "REAR_OBLIQUE"),
                    ("Front-facing", "FRONT"),
                    ("Behind or above the head (bench)", "HEAD_END_BENCH"),
                    ("Other / unsure", "OTHER"),
                ],
                value="FULL_SIDE", label="Declared camera position",
                info="Side views best support the current 2D checks. Other views may reduce or prevent specific assessments; reliability is decided from the recorded evidence.",
            )
            pose_model = gr.Dropdown(
                choices=list(POSE_MODEL_PATHS), value="Full", label="Pose Landmarker model",
                info="Full is installed by the setup command. Heavy is slower and optional; install it manually to use it.",
            )

        status = gr.Markdown("Select a video to enable analysis.", elem_classes=["feedback-status"])
        with gr.Column(visible=False) as result_panel:
            completed_result = gr.Markdown()
            coaching_feedback_html = gr.HTML(elem_id="coaching-feedback-panel", container=False)
            gr.Markdown("## Video comparison")
            with gr.Row():
                original_video = gr.Video(label="Original upload", height=360)
                annotated_video = gr.Video(label="Tracked movement", height=360)
            with gr.Row():
                debug_file = gr.File(label="Download analysis JSON")
                raw_trajectory_export = gr.DownloadButton("Download landmark export")

            with gr.Accordion("Technical details", open=False):
                gr.Markdown("Diagnostics are evidence from the selected lift only; they do not alter classifications.")
                summary = gr.JSON(label="Structured result summary")
                phase_timings = gr.Dataframe(headers=PHASE_TABLE_HEADERS, label="Phase timings")
                phase_angles = gr.Dataframe(headers=PHASE_ANGLE_HEADERS, label="Phase angles")
                findings = gr.Markdown()
                sources = gr.Markdown()
                debug = gr.JSON(label="Feedback generation details")
                with gr.Group(visible=False) as squat_diagnostics_group:
                    gr.Markdown(
                        "### Selected lift diagnostics\n"
                        "The primary and secondary metric evidence below is specific to the squat."
                    )
                    squat_structured_result = gr.JSON(
                        label="Complete structured result"
                    )
                    phase_diagnostics = gr.JSON(
                        label="Phase, trajectory and metric evidence"
                    )
                    trajectory_plot = gr.HTML()
                    depth_diagnostics = gr.JSON(label="Primary metric evidence")
                    coordination_diagnostics = gr.JSON(label="Secondary metric evidence")
                    calibration_diagnostics = gr.JSON(
                        label="Self-calibration usage and profiles"
                    )
                    recording_diagnostics = gr.JSON(label="Recording / pose summary")
                    squat_warnings = gr.JSON(label="Warnings")
                    with gr.Accordion("Additional squat diagnostics", open=False):
                        gr.Markdown(
                            "Detailed sensitivity and landmark-quality evidence for manual review."
                        )
                        sensitivity_tables = gr.HTML()
                        coordination_plot = gr.HTML()
                        unusual_observations = gr.Dataframe(headers=["Frame", "Landmark", "Signal", "Value", "Median", "MAD", "Robust deviation"], interactive=False)
                        reacquisition_diagnostics = gr.JSON(label="Landmark reacquisition diagnostics")
                        reacquisition_events = gr.Dataframe(headers=["Landmark", "Last valid frame", "Reacquired frame", "Gap frames", "Motion deviation", "Adjacent segment deviations", "Post-reacquisition continuity"], interactive=False)
                with gr.Group(visible=False) as prototype_diagnostics_group:
                    gr.Markdown("### Selected lift diagnostics")
                    with gr.Group(visible=False) as bench_pause_diagnostics_group:
                        gr.Markdown(
                            "Bench pause diagnostics are experimental and excluded from coaching."
                        )
                    bench_structured_result = gr.JSON(label="Complete structured result")
                    bench_diagnostics = gr.JSON(label="Phase, trajectory and metric evidence")
                    bench_trajectory_plot = gr.HTML()
                    bench_elbow_evidence = gr.JSON(label="Primary metric evidence")
                    bench_wrist_evidence = gr.JSON(label="Secondary metric evidence")
                    bench_calibration = gr.JSON(label="Self-calibration usage and profiles")
                    bench_warnings = gr.JSON(label="Warnings")
                    with gr.Group(visible=False) as bench_pause_export_group:
                        bench_pause_export = gr.DownloadButton("Download experimental bench-pause diagnostics")

        result_components = [
            original_video, annotated_video, coaching_feedback_html, summary, phase_timings,
            phase_angles, findings, sources, debug, debug_file, recording_diagnostics,
            phase_diagnostics, trajectory_plot, depth_diagnostics, coordination_diagnostics,
            sensitivity_tables, coordination_plot, calibration_diagnostics,
            unusual_observations, reacquisition_diagnostics, reacquisition_events,
            raw_trajectory_export, bench_structured_result, bench_diagnostics,
            bench_trajectory_plot, bench_elbow_evidence, bench_wrist_evidence,
            bench_calibration, bench_warnings, bench_pause_export,
            squat_structured_result, squat_warnings,
        ]
        analysis_outputs = [
            status,
            result_panel,
            completed_result,
            squat_diagnostics_group,
            prototype_diagnostics_group,
            bench_pause_diagnostics_group,
            bench_pause_export_group,
            *result_components,
        ]
        invalidation_outputs = [
            selection_revision, submit, upload_guidance, result_panel, status,
            *result_components,
        ]

        def invalidate(video_path, lift, request: gr.Request):
            revision = _mark_selection_changed(request)
            label, guidance = lift_ui_text(lift)
            updates = {
                selection_revision: revision,
                submit: gr.update(value=label.get("value"), interactive=bool(video_path)),
                upload_guidance: guidance,
                result_panel: gr.update(visible=False),
                status: "Video selected. Ready to analyse." if video_path else "Select a video to enable analysis.",
            }
            updates.update({component: None for component in result_components})
            updates[phase_timings] = _table_value(PHASE_TABLE_HEADERS, [])
            updates[phase_angles] = _table_value(PHASE_ANGLE_HEADERS, [])
            return updates

        def upload_selected(video_path, lift, request: gr.Request):
            path, name = store_uploaded_video(video_path)
            updates = invalidate(path, lift, request)
            updates.update({uploaded_video_path: path, selected_video_name: name})
            return updates

        def analyse(video_path, view, lift, model, revision, request: gr.Request):
            token = _claim_request(request, revision)
            if token is None:
                yield {status: "Analysis is already running for this session."}
                return
            yield {status: "Analysing the submitted video… This may take a few minutes.", result_panel: gr.update(visible=False)}
            try:
                values = process_uploaded_video(video_path, view, str(POSE_MODEL_PATHS[model]), lift)
                if not _request_is_current(request, token, revision):
                    yield {status: "Selection changed while analysis was running. The obsolete result was not displayed."}
                    return
                displayed_lift = "Conventional Deadlift" if lift == "Deadlift" else lift
                result_values = [values[0], values[1], values[4], *values[6:]]
                updates = {component: value for component, value in zip(result_components, result_values)}
                updates.update({
                    completed_result: f"## Completed assessment\n**{Path(video_path).name} · {displayed_lift}**",
                    result_panel: gr.update(visible=True),
                    squat_diagnostics_group: gr.update(visible=lift == "Squat"),
                    prototype_diagnostics_group: gr.update(visible=lift != "Squat"),
                    bench_pause_diagnostics_group: gr.update(visible=lift == "Bench Press"),
                    bench_pause_export_group: gr.update(visible=lift == "Bench Press"),
                    status: values[5],
                })
                yield updates
            except Exception as exc:
                yield {result_panel: gr.update(visible=False), status: f"**Analysis failed.** {type(exc).__name__}: {exc}"}
            finally:
                _release_request(request, token)

        video_upload_button.upload(upload_selected, [video_upload_button, lift_selector],
                                   [uploaded_video_path, selected_video_name, *invalidation_outputs], queue=False)
        lift_selector.input(invalidate, [uploaded_video_path, lift_selector],
                            invalidation_outputs, queue=False)
        camera_view.input(invalidate, [uploaded_video_path, lift_selector],
                          invalidation_outputs, queue=False)
        pose_model.input(invalidate, [uploaded_video_path, lift_selector],
                         invalidation_outputs, queue=False)
        submit.click(
            analyse,
            [uploaded_video_path, camera_view, lift_selector, pose_model, selection_revision],
            analysis_outputs,
            concurrency_limit=None,
        )
    return demo


def launch_demo(**kwargs):
    return build_demo().launch(theme=LIGHT_THEME, css=CUSTOM_CSS, **kwargs)


if __name__ == "__main__":
    launch_demo()
