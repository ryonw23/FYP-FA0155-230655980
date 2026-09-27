"""Small deterministic bench-press analyser built on the shared lift context."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any

from .lift_analysis import (
    LiftLandmarkConfiguration,
    prepare_lift_analysis_context,
    select_evidence_side,
)
from .metric_contracts import (
    EvidenceStatus,
    LandmarkState,
    MetricEvidence,
    MetricSpecification,
)
from .pose_calibration import attach_calibration_usage
from .motion_analysis import calculate_angle

MAX_PAUSE_WINDOW_DIAGNOSTICS = 200
MAX_PAUSE_TRANSITION_FRAMES = 120

BENCH_LANDMARK_BASES = ("shoulder", "elbow", "wrist")
BENCH_SIDE_SELECTION_BASES = ("shoulder", "elbow", "wrist")
BENCH_LANDMARKS = tuple(
    f"{side}_{joint}" for side in ("left", "right") for joint in BENCH_LANDMARK_BASES
)
BENCH_CALIBRATION_SEGMENTS = {
    "shoulder_elbow": ("shoulder", "elbow"),
    "elbow_wrist": ("elbow", "wrist"),
}
BENCH_OVERLAY_CONNECTIONS = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)
BENCH_LANDMARK_CONFIGURATION = LiftLandmarkConfiguration(
    retained_landmarks=BENCH_LANDMARKS,
    side_selection_bases=BENCH_SIDE_SELECTION_BASES,
    calibration_segments=BENCH_CALIBRATION_SEGMENTS,
)


@dataclass(frozen=True)
class BenchAnalysisConfig:
    visibility_threshold: float = 0.6
    minimum_side_visibility_score: float = 0.6
    minimum_usable_frame_ratio: float = 0.75
    smoothing_window: int = 5
    baseline_fraction: float = 0.2
    minimum_baseline_frames: int = 3
    minimum_motion_frames: int = 2
    minimum_wrist_travel: float = 0.04
    motion_tolerance: float = 0.004
    bottom_band_fraction: float = 0.08
    starting_region_fraction_of_travel: float = 0.15
    elbow_depth_tolerance_body_scale: float = 0.025
    elbow_depth_uncertainty_body_scale: float = 0.01
    # Provisional evidence-engineering requirements, not biomechanics standards.
    # Both conditions must be met by raw observations inside the detected bottom.
    minimum_elbow_depth_samples: int = 3
    minimum_elbow_depth_evidence_seconds: float = 0.1
    minimum_bottom_stable_seconds: float = 0.2
    bottom_position_range_tolerance_body_scale: float = 0.025
    bottom_velocity_tolerance_body_scale_per_second: float = 0.2
    bottom_stability_uncertainty_fraction: float = 0.2
    terminal_window_seconds: float = 0.2
    # Completion is a wrist-trajectory decision, deliberately independent of
    # the elbow-angle thresholds used by the subsequent technique assessment.
    terminal_wrist_range_tolerance: float = 0.012
    terminal_wrist_velocity_tolerance_per_second: float = 0.18
    terminal_continued_ascent_tolerance: float = 0.015
    terminal_complete_angle_degrees: float = 170.0
    terminal_incomplete_angle_degrees: float = 160.0


def bench_metric_specifications(
    config: BenchAnalysisConfig | None = None,
) -> dict[str, MetricSpecification]:
    cfg = config or BenchAnalysisConfig()
    reliability = {
        "minimum_landmark_visibility": cfg.visibility_threshold,
        "missing_landmarks": "not_permitted_in_source_frames",
    }
    no_threshold = {
        "threshold_source_type": "none_descriptive_metric",
        "threshold_source_reference": None,
    }
    return {
        "observed_side_elbow_depth_2d_proxy": MetricSpecification(
            metric_id="observed_side_elbow_depth_2d_proxy",
            lift="bench_press",
            required_landmarks=("shoulder", "elbow"),
            required_phase="stable bottom",
            supported_view="side_or_oblique_with_both_elbows_visible_for_overall_assessment",
            measurement_key="elbow_minus_shoulder_y_by_side_normalised",
            normalisation_method="image-y elbow-centre minus shoulder-centre divided by same-side arm-chain scale",
            decision_rule="each arm needs multiple raw shoulder/elbow observations spanning the configured elapsed time; contradictory supported frames are uncertain; both supported sides must reach the existing centre-landmark proxy boundary, while a supported unilateral failure remains corrective",
            threshold={
                "tolerance_body_scale": cfg.elbow_depth_tolerance_body_scale,
                "uncertainty_body_scale": cfg.elbow_depth_uncertainty_body_scale,
                "minimum_raw_sample_count": cfg.minimum_elbow_depth_samples,
                "minimum_elapsed_evidence_seconds": cfg.minimum_elbow_depth_evidence_seconds,
            },
            threshold_source={
                "threshold_source_type": "engineering_proxy_for_IPF_bilateral_elbow_depth_rule",
                "threshold_source_reference": "MediaPipe joint centres do not represent the joint surfaces named by the rule",
            },
            threshold_version="bench_competition_proxy_v1",
            reliability_requirements=reliability,
            missingness_policy="raw observations only; each side must meet sample-count and elapsed-span requirements; both sides are required to pass, but an adequately supported unilateral failure remains corrective",
            sensitivity_parameters=None,
            evidence_sources=(
                "pose_estimation",
                "stable_bottom_phase",
                "IPF_bilateral_rule_proxy",
            ),
        ),
        "motionless_bottom_2d_proxy": MetricSpecification(
            metric_id="motionless_bottom_2d_proxy",
            lift="bench_press",
            required_landmarks=("wrist", "elbow"),
            required_phase="bottom before press",
            supported_view="side_or_near-side",
            measurement_key="stable_duration_seconds",
            normalisation_method="bilateral smoothed wrist/elbow image-y displacement divided by same-side arm-chain scale, plus bilateral elbow-angle change",
            decision_rule="FPS-aware bilateral multi-signal stable window before concentric motion; does not infer chest contact or a referee command",
            threshold={
                "minimum_stable_seconds": cfg.minimum_bottom_stable_seconds,
                "position_range_body_scale": cfg.bottom_position_range_tolerance_body_scale,
                "velocity_body_scale_per_second": cfg.bottom_velocity_tolerance_body_scale_per_second,
                "elbow_angle_range_degrees": cfg.terminal_complete_angle_degrees
                - cfg.terminal_incomplete_angle_degrees,
                "uncertainty_fraction": cfg.bottom_stability_uncertainty_fraction,
            },
            threshold_source={
                "threshold_source_type": "provisional_engineering_operationalisation",
                "threshold_source_reference": "IPF specifies motionless but no universal duration",
            },
            threshold_version="bench_competition_proxy_v1",
            reliability_requirements=reliability,
            missingness_policy="insufficient unless an FPS-aware pre-press bottom window and body scale are available",
            sensitivity_parameters=None,
            evidence_sources=(
                "pose_estimation",
                "bilateral_smoothed_wrist_and_elbow_signals",
                "phase_state_machine",
            ),
        ),
        "terminal_elbow_extension_2d_proxy": MetricSpecification(
            metric_id="terminal_elbow_extension_2d_proxy",
            lift="bench_press",
            required_landmarks=("shoulder", "elbow", "wrist"),
            required_phase="stable returned top",
            supported_view="side_or_oblique",
            measurement_key="terminal_median_elbow_angle_degrees_by_side",
            normalisation_method=None,
            decision_rule="COMPLETE at >=170 degrees, INCOMPLETE at <160 degrees, otherwise UNCERTAIN",
            threshold={
                "complete_degrees": cfg.terminal_complete_angle_degrees,
                "incomplete_degrees": cfg.terminal_incomplete_angle_degrees,
                "top_confirmation_seconds": cfg.terminal_window_seconds,
                "top_wrist_range_tolerance": cfg.terminal_wrist_range_tolerance,
                "top_wrist_velocity_tolerance_per_second": cfg.terminal_wrist_velocity_tolerance_per_second,
                "continued_ascent_tolerance": cfg.terminal_continued_ascent_tolerance,
            },
            threshold_source={
                "threshold_source_type": "research_informed_engineering_proxy",
                "threshold_source_reference": "not official IPF angle thresholds",
            },
            threshold_version="bench_competition_proxy_v1",
            reliability_requirements=reliability,
            missingness_policy="report only reliably observed terminal elbow evidence",
            sensitivity_parameters=None,
            evidence_sources=("pose_estimation", "stable_returned_top_window"),
        ),
        "bench_elbow_flexion_2d": MetricSpecification(
            metric_id="bench_elbow_flexion_2d",
            lift="bench_press",
            required_landmarks=("shoulder", "elbow", "wrist"),
            required_phase="bottom",
            supported_view="side_or_near-side",
            measurement_key="minimum_elbow_angle_degrees",
            normalisation_method=None,
            decision_rule="descriptive 2D angle; no technique classification",
            threshold=None,
            threshold_source=no_threshold,
            threshold_version="not_applicable",
            reliability_requirements=reliability,
            missingness_policy="insufficient evidence unless shoulder, elbow and wrist coexist in a bottom source frame",
            sensitivity_parameters=None,
            evidence_sources=("pose_estimation", "phase_state_machine"),
        ),
        "bench_wrist_trajectory_2d": MetricSpecification(
            metric_id="bench_wrist_trajectory_2d",
            lift="bench_press",
            required_landmarks=("wrist",),
            required_phase="complete bench repetition",
            supported_view="side_or_near-side",
            measurement_key="wrist_trajectory_displacements_normalised",
            normalisation_method="2D MediaPipe wrist displacement divided by the median shoulder-elbow plus elbow-wrist arm-chain scale; wrist is a hand/bar-end proxy",
            decision_rule="descriptive 2D trajectory; no technique or bar-path classification",
            threshold=None,
            threshold_source=no_threshold,
            threshold_version="not_applicable",
            reliability_requirements=reliability,
            missingness_policy="insufficient evidence unless wrist observations support start, bottom and completion",
            sensitivity_parameters=None,
            evidence_sources=(
                "pose_estimation",
                "median_smoothed_signal",
                "phase_state_machine",
            ),
        ),
        "bench_finish_geometry_similarity": MetricSpecification(
            metric_id="bench_finish_geometry_similarity",
            lift="bench_press",
            required_landmarks=("shoulder", "elbow", "wrist"),
            required_phase="top and complete",
            supported_view="side_or_near-side",
            measurement_key="start_to_finish_geometry_difference",
            normalisation_method="short-window medians; wrist distance divided by median shoulder-elbow plus elbow-wrist arm-chain scale",
            decision_rule="descriptive self-comparison with starting arm geometry; not a lockout classification",
            threshold=None,
            threshold_source=no_threshold,
            threshold_version="not_applicable",
            reliability_requirements=reliability,
            missingness_policy="insufficient evidence unless shoulder, elbow and wrist support both top and complete windows and arm-chain scale is available",
            sensitivity_parameters=None,
            evidence_sources=(
                "pose_estimation",
                "phase_state_machine",
                "median_window",
            ),
        ),
    }


def _smooth(values: list[float | None], window: int) -> list[float | None]:
    half = window // 2
    return [
        (
            float(
                median(
                    v for v in values[max(0, i - half) : i + half + 1] if v is not None
                )
            )
            if any(v is not None for v in values[max(0, i - half) : i + half + 1])
            else None
        )
        for i in range(len(values))
    ]


def _visible(frame: dict[str, dict[str, float]], name: str, threshold: float) -> bool:
    landmark = frame.get(name)
    return bool(
        landmark
        and float(landmark.get("visibility", 0)) >= threshold
        and float(landmark.get("presence", 1)) >= threshold
    )


def _quality(
    frames: list[dict[str, dict[str, float]]], names: list[str], source: list[int]
) -> tuple[dict[str, Any], dict[str, LandmarkState], dict[str, str]]:
    quality: dict[str, Any] = {}
    states: dict[str, LandmarkState] = {}
    origins: dict[str, str] = {}
    for name in names:
        observations = [frames[i][name] for i in source if name in frames[i]]
        quality[name] = {
            "source_frame_coverage": (
                round(len(observations) / len(source), 3) if source else 0.0
            ),
            "median_visibility": (
                round(
                    float(median(float(v.get("visibility", 0)) for v in observations)),
                    3,
                )
                if observations
                else None
            ),
            "median_presence": (
                round(
                    float(median(float(v.get("presence", 1)) for v in observations)), 3
                )
                if observations
                else None
            ),
        }
        states[name] = (
            LandmarkState.OBSERVED
            if len(observations) == len(source) and source
            else LandmarkState.UNAVAILABLE
        )
        origins[name] = "mediapipe_pose_observation" if observations else "unavailable"
    return quality, states, origins


def _insufficient(
    spec: MetricSpecification, side: str | None, warning: str
) -> dict[str, Any]:
    names = [f"{side}_{base}" if side else base for base in spec.required_landmarks]
    return MetricEvidence(
        spec.metric_id,
        1,
        spec.required_phase,
        [],
        names,
        {},
        {name: LandmarkState.UNAVAILABLE for name in names},
        {name: "unavailable" for name in names},
        None,
        spec.normalisation_method,
        spec.decision_rule,
        spec.threshold,
        None,
        {"sufficient": False, "metric_specific": True},
        None,
        EvidenceStatus.INSUFFICIENT,
        None,
        [warning],
    ).to_dict()


def _detect_phases(
    wrist: list[float | None],
    raw_wrist: list[float | None],
    fps: float,
    cfg: BenchAnalysisConfig,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    available = [i for i, value in enumerate(wrist) if value is not None]
    if len(available) < max(2 * cfg.minimum_baseline_frames, 6):
        return {}, {"failure_reason": "insufficient_wrist_trajectory"}
    baseline_count = max(
        cfg.minimum_baseline_frames, round(len(available) * cfg.baseline_fraction)
    )
    baseline_frames = available[:baseline_count]
    baseline = float(
        median(float(wrist[i]) for i in baseline_frames if wrist[i] is not None)
    )
    bottom = max(available, key=lambda i: float(wrist[i] or -math.inf))
    travel = float(wrist[bottom]) - baseline  # image y increases downwards
    if travel < cfg.minimum_wrist_travel or bottom <= baseline_frames[-1]:
        return {}, {
            "failure_reason": "no_coherent_descent",
            "wrist_travel": round(travel, 4),
            "baseline_wrist_y": baseline,
        }
    down = [
        i
        for i in available
        if i > baseline_frames[-1]
        and i <= bottom
        and float(wrist[i]) >= baseline + cfg.motion_tolerance
    ]
    up = [
        i
        for i in available
        if i > bottom and float(wrist[i]) <= float(wrist[bottom]) - cfg.motion_tolerance
    ]
    if len(down) < cfg.minimum_motion_frames or len(up) < cfg.minimum_motion_frames:
        return {}, {
            "failure_reason": "ordered_descent_and_ascent_not_observed",
            "wrist_travel": round(travel, 4),
        }
    descent_start, ascent_start = down[0], up[0]
    return_tolerance = max(
        cfg.motion_tolerance, travel * cfg.starting_region_fraction_of_travel
    )
    returned = [
        i
        for i in available
        if i >= ascent_start and abs(float(wrist[i]) - baseline) <= return_tolerance
    ]
    required_terminal_samples = (
        max(2, math.ceil(cfg.terminal_window_seconds * fps) + 1) if fps > 0 else 0
    )
    tested_candidates: list[dict[str, Any]] = []
    confirmed_window: list[int] = []
    for candidate in returned:
        window = list(range(candidate, candidate + required_terminal_samples))
        observed = (
            bool(window)
            and window[-1] < len(raw_wrist)
            and all(raw_wrist[i] is not None for i in window)
        )
        values = (
            [float(wrist[i]) for i in window if wrist[i] is not None]
            if observed
            else []
        )
        contiguous_derived = len(values) == len(window)
        position_range = max(values) - min(values) if values else None
        max_velocity = (
            max(
                (abs(b - a) * fps for a, b in zip(values, values[1:])),
                default=0.0,
            )
            if values and fps > 0
            else None
        )
        later = [
            float(wrist[i])
            for i in available
            if window and i > window[-1] and wrist[i] is not None
        ]
        continued_ascent = bool(
            values
            and later
            and min(later) < min(values) - cfg.terminal_continued_ascent_tolerance
        )
        stable = bool(
            observed
            and contiguous_derived
            and position_range is not None
            and position_range <= cfg.terminal_wrist_range_tolerance
            and max_velocity is not None
            and max_velocity <= cfg.terminal_wrist_velocity_tolerance_per_second
            and not continued_ascent
        )
        tested_candidates.append(
            {
                "candidate_frame": candidate,
                "window_start_frame": window[0] if window else None,
                "window_end_frame": window[-1] if window else None,
                "elapsed_seconds": (
                    (window[-1] - window[0]) / fps if window and fps > 0 else None
                ),
                "raw_observations_complete": observed,
                "signal_provenance": (
                    "observed_and_smoothed"
                    if observed
                    else "derived_only_or_unavailable"
                ),
                "wrist_position_range": (
                    round(position_range, 6) if position_range is not None else None
                ),
                "maximum_wrist_velocity_per_second": (
                    round(max_velocity, 6) if max_velocity is not None else None
                ),
                "continued_ascent_after_window": continued_ascent,
                "confirmed": stable,
            }
        )
        if stable:
            confirmed_window = window
            break
    complete = confirmed_window[0] if confirmed_window else None
    band = max(cfg.motion_tolerance, travel * cfg.bottom_band_fraction)
    bottom_frames = [
        i
        for i in available
        if descent_start <= i <= ascent_start
        and float(wrist[i]) >= float(wrist[bottom]) - band
    ]
    bottom_start, bottom_end = min(bottom_frames), max(bottom_frames)
    phases = {
        "top": {"start_frame": available[0], "end_frame": descent_start - 1},
        "descent": {"start_frame": descent_start, "end_frame": bottom_start - 1},
        "bottom": {"start_frame": bottom_start, "end_frame": bottom_end},
        "ascent": {
            "start_frame": bottom_end + 1,
            "end_frame": complete - 1 if complete is not None else available[-1],
        },
    }
    if complete is not None:
        phases["complete"] = {
            "start_frame": complete,
            "end_frame": available[-1],
        }
    return phases, {
        "failure_reason": None,
        "baseline_wrist_y": round(baseline, 6),
        "bottom_frame": bottom,
        "bottom_band_boundary_y": round(float(wrist[bottom]) - band, 6),
        "ascent_start_criterion_y": round(
            float(wrist[bottom]) - cfg.motion_tolerance, 6
        ),
        "wrist_travel": round(travel, 6),
        "return_tolerance": round(return_tolerance, 6),
        "terminal_evidence_window": confirmed_window,
        "terminal_position_confirmed": bool(confirmed_window),
        "observed_ascent_end_frame": available[-1],
        "terminal_unavailable_reason": (
            None
            if confirmed_window
            else "No returned-top candidate had a fully observed, FPS-duration wrist window that stayed within the configured motion tolerances without meaningful continued ascent."
        ),
        "terminal_confirmation": {
            "required_elapsed_seconds": cfg.terminal_window_seconds,
            "required_sample_count": required_terminal_samples,
            "wrist_position_range_tolerance": cfg.terminal_wrist_range_tolerance,
            "wrist_velocity_tolerance_per_second": cfg.terminal_wrist_velocity_tolerance_per_second,
            "continued_ascent_tolerance": cfg.terminal_continued_ascent_tolerance,
            "tested_candidates": tested_candidates,
        },
    }


def _arm_scale(
    frames: list[dict[str, dict[str, float]]],
    side: str,
    source: list[int],
    threshold: float,
) -> float | None:
    names = [f"{side}_{joint}" for joint in ("shoulder", "elbow", "wrist")]
    scales = []
    for i in source:
        if all(_visible(frames[i], name, threshold) for name in names):
            points = [frames[i][name] for name in names]
            scales.append(
                math.hypot(
                    points[1]["x"] - points[0]["x"], points[1]["y"] - points[0]["y"]
                )
                + math.hypot(
                    points[2]["x"] - points[1]["x"], points[2]["y"] - points[1]["y"]
                )
            )
    return float(median(scales)) if scales and median(scales) > 0 else None


def _assessment_evidence(
    spec: MetricSpecification,
    frames: list[dict[str, dict[str, float]]],
    names: list[str],
    source: list[int],
    measurement: dict[str, Any],
    classification: str,
) -> dict[str, Any]:
    quality, states, origins = _quality(frames, names, source)
    return MetricEvidence(
        spec.metric_id,
        1,
        spec.required_phase,
        source,
        names,
        quality,
        states,
        origins,
        measurement,
        spec.normalisation_method,
        spec.decision_rule,
        spec.threshold,
        None,
        {
            "sufficient": classification != "INSUFFICIENT_EVIDENCE",
            "metric_specific": True,
            "confidence": "medium",
        },
        None,
        (
            EvidenceStatus.SUFFICIENT
            if classification != "INSUFFICIENT_EVIDENCE"
            else EvidenceStatus.INSUFFICIENT
        ),
        classification,
        [],
    ).to_dict()


def _competition_proxy_assessments(
    context: Any,
    phases: dict[str, dict[str, int]],
    cfg: BenchAnalysisConfig,
    specs: dict[str, MetricSpecification],
    phase_diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    frames = context.frames
    bottom = list(
        range(phases["bottom"]["start_frame"], phases["bottom"]["end_frame"] + 1)
    )
    depth_values: dict[str, float] = {}
    depth_support: dict[str, dict[str, Any]] = {}
    tolerance, margin = (
        cfg.elbow_depth_tolerance_body_scale,
        cfg.elbow_depth_uncertainty_body_scale,
    )
    for candidate in ("left", "right"):
        shoulder, elbow = f"{candidate}_shoulder", f"{candidate}_elbow"
        source = [
            i
            for i in bottom
            if _visible(frames[i], shoulder, cfg.visibility_threshold)
            and _visible(frames[i], elbow, cfg.visibility_threshold)
        ]
        scale = _arm_scale(frames, candidate, bottom, cfg.visibility_threshold)
        values = (
            [(frames[i][elbow]["y"] - frames[i][shoulder]["y"]) / scale for i in source]
            if scale
            else []
        )
        elapsed = (
            (source[-1] - source[0]) / context.fps
            if len(source) >= 2 and context.fps > 0
            else 0.0
        )
        enough_samples = len(values) >= cfg.minimum_elbow_depth_samples
        enough_span = elapsed >= cfg.minimum_elbow_depth_evidence_seconds
        supported = bool(scale and enough_samples and enough_span)
        frame_classes = [
            (
                "DEPTH_SUFFICIENT"
                if value >= -tolerance + margin
                else (
                    "DEPTH_INSUFFICIENT" if value < -tolerance - margin else "UNCERTAIN"
                )
            )
            for value in values
        ]
        contradictory = (
            "DEPTH_SUFFICIENT" in frame_classes
            and "DEPTH_INSUFFICIENT" in frame_classes
        )
        summary = float(median(values)) if values else None
        if not supported:
            side_class = "INSUFFICIENT_EVIDENCE"
            side_reason = "Raw shoulder/elbow evidence did not meet both the provisional sample-count and elapsed-span requirements."
        elif contradictory:
            side_class = "UNCERTAIN"
            side_reason = "Raw observations crossed both depth decision boundaries."
        elif summary is not None:
            side_class = (
                "DEPTH_SUFFICIENT"
                if summary >= -tolerance + margin
                else (
                    "DEPTH_INSUFFICIENT"
                    if summary < -tolerance - margin
                    else "UNCERTAIN"
                )
            )
            side_reason = "The median of the supported raw observations was classified against the existing boundaries."
        depth_support[candidate] = {
            "raw_observed_frames": source,
            "raw_sample_count": len(values),
            "elapsed_evidence_seconds": round(elapsed, 6),
            "minimum_raw_sample_count": cfg.minimum_elbow_depth_samples,
            "minimum_elapsed_evidence_seconds": cfg.minimum_elbow_depth_evidence_seconds,
            "sample_count_requirement_met": enough_samples,
            "elapsed_span_requirement_met": enough_span,
            "support_sufficient": supported,
            "normalisation_scale": round(scale, 6) if scale else None,
            "per_frame_normalised_measurements": [
                {
                    "frame": frame,
                    "value": round(value, 6),
                    "classification": classification,
                }
                for frame, value, classification in zip(source, values, frame_classes)
            ],
            "contradictory_evidence": contradictory,
            "classification": side_class,
            "classification_reason": side_reason,
        }
        if supported and summary is not None:
            depth_values[candidate] = summary
    per_side = {
        side: detail["classification"] for side, detail in depth_support.items()
    }
    if "DEPTH_INSUFFICIENT" in per_side.values():
        depth_class = "DEPTH_INSUFFICIENT"
    elif any(value == "INSUFFICIENT_EVIDENCE" for value in per_side.values()):
        depth_class = "INSUFFICIENT_EVIDENCE"
    elif all(value == "DEPTH_SUFFICIENT" for value in per_side.values()):
        depth_class = "DEPTH_SUFFICIENT"
    else:
        depth_class = "UNCERTAIN"
    observed_depth_sides = [
        side
        for side, details in depth_support.items()
        if details["raw_observed_frames"]
    ]
    depth_names = [
        f"{side}_{joint}"
        for side in observed_depth_sides
        for joint in ("shoulder", "elbow")
    ]
    depth_source = sorted(
        {
            frame
            for details in depth_support.values()
            for frame in details["raw_observed_frames"]
        }
    )
    depth_measurement = {
        "proxy_label": "observed_side_elbow_depth_2d_proxy; not competition judging",
        "elbow_minus_shoulder_y_by_side_normalised": {
            s: round(v, 6) for s, v in depth_values.items()
        },
        "side_classifications": per_side,
        "reliable_sides": list(depth_values),
        "per_side_support": depth_support,
        "evidence_frames": {
            side: details["raw_observed_frames"]
            for side, details in depth_support.items()
        },
        "bilateral_evidence_required": True,
        "classification_reason": (
            "At least one reliably observed elbow remained above the configured shoulder-level proxy boundary."
            if depth_class == "DEPTH_INSUFFICIENT"
            else (
                "Both reliably observed elbows reached the configured shoulder-level proxy boundary."
                if depth_class == "DEPTH_SUFFICIENT"
                else (
                    "Adequately sampled evidence was conflicting or inside the configured uncertainty band."
                    if depth_class == "UNCERTAIN"
                    else "At least one arm lacked the provisional raw sample count or elapsed evidence span, so no bilateral depth conclusion was produced."
                )
            )
        ),
    }
    depth_result = _assessment_evidence(
        specs["observed_side_elbow_depth_2d_proxy"],
        frames,
        depth_names,
        depth_source,
        depth_measurement,
        depth_class,
    )
    if depth_class == "INSUFFICIENT_EVIDENCE":
        depth_result["warnings"] = [
            "Depth evidence did not meet the provisional temporal support requirements on both arms; no depth correction or praise is supported."
        ]

    pause_spec = specs["motionless_bottom_2d_proxy"]
    ascent_start = phases["ascent"]["start_frame"]
    candidates = [i for i in bottom if i < ascent_start]
    required = (
        max(2, math.ceil(cfg.minimum_bottom_stable_seconds * context.fps) + 1)
        if context.fps > 0
        else 0
    )
    angle_tolerance = (
        cfg.terminal_complete_angle_degrees - cfg.terminal_incomplete_angle_degrees
    )
    signals: dict[str, dict[str, list[float | None]]] = {}
    raw_signals: dict[str, dict[str, list[float | None]]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    scales: dict[str, float | None] = {}
    for candidate_side in ("left", "right"):
        scales[candidate_side] = _arm_scale(
            frames, candidate_side, list(range(len(frames))), cfg.visibility_threshold
        )
        side_signals: dict[str, list[float | None]] = {}
        side_raw_signals: dict[str, list[float | None]] = {}
        for joint in ("wrist", "elbow"):
            name = f"{candidate_side}_{joint}"
            raw_signal = [
                (
                    float(frame[name]["y"])
                    if _visible(frame, name, cfg.visibility_threshold)
                    else None
                )
                for frame in frames
            ]
            side_signals[joint] = _smooth(raw_signal, cfg.smoothing_window)
            side_raw_signals[joint] = raw_signal
            observed = sum(raw_signal[i] is not None for i in candidates)
            interpolated = sum(
                raw_signal[i] is None and side_signals[joint][i] is not None
                for i in candidates
            )
            coverage.setdefault(candidate_side, {})[joint] = {
                "observed_frames": observed,
                "interpolated_frames": interpolated,
                "total_frames": len(candidates),
                "coverage_ratio": (
                    round(observed / len(candidates), 3) if candidates else 0.0
                ),
                "evidence_origin": (
                    "observed"
                    if observed == len(candidates) and candidates
                    else (
                        "observed_and_interpolated"
                        if observed and interpolated
                        else "observed" if observed else "unavailable"
                    )
                ),
            }
        raw_angles = []
        names = [f"{candidate_side}_{j}" for j in ("shoulder", "elbow", "wrist")]
        for frame in frames:
            raw_angles.append(
                calculate_angle(frame[names[0]], frame[names[1]], frame[names[2]])
                if all(
                    _visible(frame, name, cfg.visibility_threshold) for name in names
                )
                else None
            )
        side_signals["elbow_angle"] = _smooth(raw_angles, cfg.smoothing_window)
        side_raw_signals["elbow_angle"] = raw_angles
        observed_angles = sum(raw_angles[i] is not None for i in candidates)
        coverage[candidate_side]["elbow_angle"] = {
            "observed_frames": observed_angles,
            "interpolated_frames": sum(
                raw_angles[i] is None and side_signals["elbow_angle"][i] is not None
                for i in candidates
            ),
            "total_frames": len(candidates),
            "coverage_ratio": (
                round(observed_angles / len(candidates), 3) if candidates else 0.0
            ),
            "evidence_origin": (
                "observed"
                if observed_angles == len(candidates) and candidates
                else ("partial_or_interpolated" if observed_angles else "unavailable")
            ),
        }
        signals[candidate_side] = side_signals
        raw_signals[candidate_side] = side_raw_signals

    bilateral_channels = {
        joint: all(
            scales[s] and coverage[s][joint]["observed_frames"] == len(candidates)
            for s in ("left", "right")
        )
        for joint in ("wrist", "elbow")
    }

    thresholds = {
        "position_range_body_scale": cfg.bottom_position_range_tolerance_body_scale,
        "maximum_velocity_body_scale_per_second": cfg.bottom_velocity_tolerance_body_scale_per_second,
        "elbow_angle_change_degrees": angle_tolerance,
    }

    def window_measurements(window: list[int]) -> tuple[bool, dict[str, Any]]:
        details: dict[str, Any] = {}
        stable = True
        for candidate_side in ("left", "right"):
            side_details: dict[str, Any] = {}
            scale = scales[candidate_side]
            for joint in ("wrist", "elbow"):
                values = [signals[candidate_side][joint][i] for i in window]
                if not scale or any(value is None for value in values):
                    side_details[joint] = {
                        "available": False,
                        "failed_checks": ["availability"],
                    }
                    continue
                numeric = [float(value) for value in values if value is not None]
                position_range = (max(numeric) - min(numeric)) / scale
                max_velocity = max(
                    (
                        abs(b - a) * context.fps / scale
                        for a, b in zip(numeric, numeric[1:])
                    ),
                    default=0.0,
                )
                side_details[joint] = {
                    "available": True,
                    "position_range_body_scale": round(position_range, 6),
                    "maximum_velocity_body_scale_per_second": round(max_velocity, 6),
                    "failed_checks": [
                        check
                        for check, failed in (
                            (
                                "position_range",
                                position_range
                                > cfg.bottom_position_range_tolerance_body_scale,
                            ),
                            (
                                "maximum_velocity",
                                max_velocity
                                > cfg.bottom_velocity_tolerance_body_scale_per_second,
                            ),
                        )
                        if failed
                    ],
                }
                stable &= (
                    position_range <= cfg.bottom_position_range_tolerance_body_scale
                )
                stable &= (
                    max_velocity <= cfg.bottom_velocity_tolerance_body_scale_per_second
                )
            angles = [signals[candidate_side]["elbow_angle"][i] for i in window]
            if all(value is not None for value in angles):
                angle_change = max(float(v) for v in angles) - min(
                    float(v) for v in angles
                )
                side_details["elbow_angle_change_degrees"] = round(angle_change, 3)
                side_details["elbow_angle_failed"] = angle_change > angle_tolerance
                stable &= angle_change <= angle_tolerance
            else:
                side_details["elbow_angle_change_degrees"] = None
                side_details["elbow_angle_failed"] = True
            failed_checks: list[dict[str, Any]] = []
            for joint in ("wrist", "elbow"):
                joint_details = side_details[joint]
                if not joint_details.get("available"):
                    failed_checks.append(
                        {
                            "check": f"{joint}_availability",
                            "measured": None,
                            "threshold": "required",
                        }
                    )
                    continue
                for failed_name, measured_key, threshold_key in (
                    (
                        "position_range",
                        "position_range_body_scale",
                        "position_range_body_scale",
                    ),
                    (
                        "maximum_velocity",
                        "maximum_velocity_body_scale_per_second",
                        "maximum_velocity_body_scale_per_second",
                    ),
                ):
                    if failed_name in joint_details["failed_checks"]:
                        failed_checks.append(
                            {
                                "check": f"{joint}_{failed_name}",
                                "measured": joint_details[measured_key],
                                "threshold": thresholds[threshold_key],
                            }
                        )
            if side_details["elbow_angle_failed"]:
                failed_checks.append(
                    {
                        "check": "elbow_angle_change",
                        "measured": side_details["elbow_angle_change_degrees"],
                        "threshold": thresholds["elbow_angle_change_degrees"],
                    }
                )
            side_details["failed_checks"] = failed_checks
            details[candidate_side] = side_details
        return stable, details

    longest: tuple[list[int], dict[str, Any]] = ([], {})
    tested_windows: list[dict[str, Any]] = []
    tested_window_count = 0
    if any(bilateral_channels.values()):
        for start in range(len(candidates)):
            for end in range(start + 1, len(candidates)):
                window = candidates[start : end + 1]
                if any(b != a + 1 for a, b in zip(window, window[1:])):
                    break
                stable, details = window_measurements(window)
                tested_window_count += 1
                if len(tested_windows) < MAX_PAUSE_WINDOW_DIAGNOSTICS:
                    tested_windows.append(
                        {
                            "start_frame": window[0],
                            "end_frame": window[-1],
                            "sample_count": len(window),
                            "passes": stable,
                            "by_side": details,
                        }
                    )
                if stable and len(window) > len(longest[0]):
                    longest = (window, details)
    pause_source = longest[0] or candidates
    stable_span = (longest[0][-1] - longest[0][0]) if longest[0] else 0
    stable_seconds = stable_span / context.fps if context.fps > 0 else 0.0
    half_window = cfg.smoothing_window // 2
    transition_start = max(0, phases["bottom"]["start_frame"] - half_window)
    transition_end = min(len(frames) - 1, phases["ascent"]["start_frame"] + half_window)
    transition_indices = list(range(transition_start, transition_end + 1))
    shown_indices = transition_indices[:MAX_PAUSE_TRANSITION_FRAMES]
    transition_table = [
        {
            "frame": i,
            **{
                f"{candidate_side}_{signal}_{kind}": (
                    round(float(series[i]), 6) if series[i] is not None else None
                )
                for candidate_side in ("left", "right")
                for signal in ("wrist", "elbow", "elbow_angle")
                for kind, series in (
                    ("raw", raw_signals[candidate_side][signal]),
                    ("smoothed", signals[candidate_side][signal]),
                )
            },
        }
        for i in shown_indices
    ]
    if not any(bilateral_channels.values()):
        pause_class = "INSUFFICIENT_EVIDENCE"
        reason = "Neither bilateral wrist nor bilateral elbow observations reliably establish bottom stability."
    elif longest[0] and len(longest[0]) >= required:
        pause_class = "MOTIONLESS_BOTTOM_OBSERVED"
        reason = "Both sides' available motion signals remained within every configured tolerance for the required FPS-aware duration."
    else:
        pause_class = "NO_MOTIONLESS_BOTTOM"
        reason = "Ordered descent was followed by ascent without a bilateral stable interval meeting the configured duration."
    pause_measurement = {
        "required_stable_frames": required,
        "required_stable_seconds": cfg.minimum_bottom_stable_seconds,
        "candidate_frame_range": {
            "start_frame": candidates[0] if candidates else None,
            "end_frame": candidates[-1] if candidates else None,
            "sample_count": len(candidates),
            "elapsed_seconds": (
                (candidates[-1] - candidates[0]) / context.fps
                if candidates and context.fps > 0
                else 0.0
            ),
        },
        "available_pre_press_bottom_frames": len(candidates),
        "phase_duration_frames": max(
            0, phases["bottom"]["end_frame"] - phases["bottom"]["start_frame"]
        ),
        "phase_duration_seconds": (
            (phases["bottom"]["end_frame"] - phases["bottom"]["start_frame"])
            / context.fps
            if context.fps > 0
            else None
        ),
        "per_side_coverage": coverage,
        "bilateral_channels_reliable": bilateral_channels,
        "longest_stable_interval": {
            "start_frame": longest[0][0] if longest[0] else None,
            "end_frame": longest[0][-1] if longest[0] else None,
            "duration_frames": stable_span,
            "sample_count": len(longest[0]),
            "duration_seconds": stable_seconds,
            "per_side_measurements": longest[1],
        },
        "pause_diagnostics": {
            "thresholds": thresholds,
            "tested_candidate_windows": tested_windows,
            "tested_window_count": tested_window_count,
            "tested_windows_omitted": max(0, tested_window_count - len(tested_windows)),
            "window_detail_limit": MAX_PAUSE_WINDOW_DIAGNOSTICS,
            "transition_table": transition_table,
            "transition_frame_range": {
                "start_frame": transition_start,
                "end_frame": transition_end,
                "included_smoothing_context_frames_each_side": half_window,
                "frames_omitted": max(0, len(transition_indices) - len(shown_indices)),
                "detail_limit": MAX_PAUSE_TRANSITION_FRAMES,
            },
            "arm_scales": scales,
            "phase_selection": {
                "selected_side": context.selected_side,
                "absolute_bottom_frame": phase_diagnostics.get("bottom_frame"),
                "bottom_band_boundary_y": phase_diagnostics.get(
                    "bottom_band_boundary_y"
                ),
                "ascent_start_criterion_y": phase_diagnostics.get(
                    "ascent_start_criterion_y"
                ),
                "ascent_start_frame": phases["ascent"]["start_frame"],
            },
        },
        "classification_reason": reason,
        "evidence_origin": {
            s: {joint: coverage[s][joint]["evidence_origin"] for joint in coverage[s]}
            for s in ("left", "right")
        },
        "operationalisation": "provisional engineering duration; no chest contact or referee Press command is inferred",
    }
    pause_result = _assessment_evidence(
        pause_spec,
        frames,
        [
            f"{s}_{joint}"
            for s in ("left", "right")
            for joint in ("shoulder", "elbow", "wrist")
        ],
        pause_source,
        pause_measurement,
        pause_class,
    )

    terminal_spec = specs["terminal_elbow_extension_2d_proxy"]
    complete = list(phase_diagnostics.get("terminal_evidence_window") or [])
    terminal_angles: dict[str, float] = {}
    terminal_sources: dict[str, list[int]] = {}
    for candidate in ("left", "right"):
        names = [f"{candidate}_{j}" for j in ("shoulder", "elbow", "wrist")]
        source = [
            i
            for i in complete
            if all(_visible(frames[i], n, cfg.visibility_threshold) for n in names)
        ]
        if source:
            terminal_angles[candidate] = float(
                median(
                    calculate_angle(
                        frames[i][names[0]], frames[i][names[1]], frames[i][names[2]]
                    )
                    for i in source
                )
            )
            terminal_sources[candidate] = source
    terminal_per_side = {
        s: (
            "COMPLETE"
            if value >= cfg.terminal_complete_angle_degrees
            else (
                "INCOMPLETE"
                if value < cfg.terminal_incomplete_angle_degrees
                else "UNCERTAIN"
            )
        )
        for s, value in terminal_angles.items()
    }
    if "INCOMPLETE" in terminal_per_side.values():
        terminal_class = "INCOMPLETE"
    elif len(terminal_per_side) < 2:
        terminal_class = "INSUFFICIENT_EVIDENCE"
    elif all(value == "COMPLETE" for value in terminal_per_side.values()):
        terminal_class = "COMPLETE"
    else:
        terminal_class = "UNCERTAIN"
    terminal_names = [
        f"{s}_{j}" for s in terminal_sources for j in ("shoulder", "elbow", "wrist")
    ]
    terminal_source = sorted(
        set(i for values in terminal_sources.values() for i in values)
    )
    terminal_measurement = {
        "proxy_label": "terminal elbow-extension 2D proxy; not competition judging",
        "terminal_median_elbow_angle_degrees_by_side": {
            s: round(v, 2) for s, v in terminal_angles.items()
        },
        "side_classifications": terminal_per_side,
        "reliable_sides": list(terminal_angles),
        "bilateral_evidence_required": True,
        "classification_reason": (
            "At least one reliably observed terminal elbow angle was below the configured incomplete-extension boundary."
            if terminal_class == "INCOMPLETE"
            else (
                "Both reliably observed terminal elbow angles reached the configured complete-extension boundary."
                if terminal_class == "COMPLETE"
                else (
                    "Both elbows were observed, but at least one terminal angle was between the configured extension boundaries."
                    if terminal_class == "UNCERTAIN"
                    else (
                        phase_diagnostics.get("terminal_unavailable_reason")
                        or "Both terminal elbows were not reliably observed, so no bilateral extension conclusion was produced."
                    )
                )
            )
        ),
        "research_note": "170°/160° boundaries are research-informed engineering proxies, not official IPF angle thresholds",
    }
    terminal_result = _assessment_evidence(
        terminal_spec,
        frames,
        terminal_names,
        terminal_source,
        terminal_measurement,
        terminal_class,
    )
    if not complete:
        terminal_result["warnings"] = [
            phase_diagnostics.get("terminal_unavailable_reason")
            or "A supported terminal wrist window was unavailable."
        ]
    return [depth_result, pause_result, terminal_result]


def analyse_bench_press_landmarks(
    frames: list[dict[str, dict[str, float]]],
    fps: float = 30.0,
    processing_metadata: dict[str, Any] | None = None,
    config: BenchAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Analyse one bench repetition with provisional depth and extension proxies."""
    cfg = config or BenchAnalysisConfig()
    specs = bench_metric_specifications(cfg)
    context = prepare_lift_analysis_context(
        lift="bench_press",
        frames=frames,
        fps=fps,
        visibility_threshold=cfg.visibility_threshold,
        landmarks=BENCH_LANDMARK_CONFIGURATION,
        processing_metadata=processing_metadata,
        minimum_side_visibility_score=cfg.minimum_side_visibility_score,
        minimum_usable_frame_ratio=cfg.minimum_usable_frame_ratio,
        phase_tracking_bases=("wrist",),
    )
    side = context.selected_side
    phase_candidates = []
    for candidate_side in ("left", "right"):
        name = f"{candidate_side}_wrist"
        candidate_raw = [
            (
                float(frame[name]["y"])
                if _visible(frame, name, cfg.visibility_threshold)
                else None
            )
            for frame in context.frames
        ]
        candidate_smooth = _smooth(candidate_raw, cfg.smoothing_window)
        candidate_phases, candidate_diagnostics = _detect_phases(
            candidate_smooth, candidate_raw, fps, cfg
        )
        phase_candidates.append(
            (
                bool(candidate_phases),
                sum(value is not None for value in candidate_raw),
                candidate_side,
                candidate_raw,
                candidate_smooth,
                candidate_phases,
                candidate_diagnostics,
            )
        )
    phase_candidates.sort(
        key=lambda item: (item[0], item[1], item[2] == "left"), reverse=True
    )
    best_phase = phase_candidates[0]
    if best_phase[1]:
        side = best_phase[2]
        context.visibility["selected_side"] = side
        context.recording_reliability["selected_side"] = side
        context.recording_reliability["selected_side_usable_frames"] = best_phase[1]
        context.recording_reliability["selected_side_usable_frame_ratio"] = (
            round(best_phase[1] / len(context.frames), 3) if context.frames else 0.0
        )
        context.recording_reliability["rejection_reasons"] = []
    wrist_name = f"{side}_wrist" if side else ""
    raw = [
        (
            float(frame[wrist_name]["y"])
            if wrist_name and _visible(frame, wrist_name, cfg.visibility_threshold)
            else None
        )
        for frame in context.frames
    ]
    raw_x = [
        (
            float(frame[wrist_name]["x"])
            if wrist_name and _visible(frame, wrist_name, cfg.visibility_threshold)
            else None
        )
        for frame in context.frames
    ]
    smooth = best_phase[4] if side else _smooth(raw, cfg.smoothing_window)
    smooth_x = _smooth(raw_x, cfg.smoothing_window)
    phases, phase_diagnostics = (
        (best_phase[5], best_phase[6])
        if side
        else _detect_phases(smooth, raw, fps, cfg)
    )
    recording_reliability = context.recording_reliability
    if recording_reliability["rejection_reasons"]:
        phases = {}
        phase_diagnostics = {
            **phase_diagnostics,
            "failure_reason": recording_reliability["rejection_reasons"][0],
        }
    warnings: list[str] = []
    results: list[dict[str, Any]] = []
    if not phases:
        warning = "A coherent TOP–DESCENT–BOTTOM–ASCENT–COMPLETE wrist trajectory was not identified."
        warnings.append(warning)
        results = [_insufficient(spec, side, warning) for spec in specs.values()]
        competition_proxy_ids = {
            "observed_side_elbow_depth_2d_proxy",
            "motionless_bottom_2d_proxy",
            "terminal_elbow_extension_2d_proxy",
        }
        for result in results:
            if result["metric_id"] in competition_proxy_ids:
                result["classification"] = "INSUFFICIENT_EVIDENCE"
        if not recording_reliability["rejection_reasons"]:
            recording_reliability = {
                **recording_reliability,
                "rejection_reasons": ["incomplete_repetition"],
            }
    else:
        bottom_source = list(
            range(phases["bottom"]["start_frame"], phases["bottom"]["end_frame"] + 1)
        )
        results.extend(
            _competition_proxy_assessments(
                context, phases, cfg, specs, phase_diagnostics
            )
        )
        elbow_selection = select_evidence_side(
            context.frames,
            ("shoulder", "elbow", "wrist"),
            cfg.visibility_threshold,
            bottom_source,
        )
        elbow_side = elbow_selection["selected_side"] or side
        elbow_names = [
            f"{elbow_side}_{name}" for name in ("shoulder", "elbow", "wrist")
        ]
        angles = [
            (
                i,
                calculate_angle(
                    context.frames[i][elbow_names[0]],
                    context.frames[i][elbow_names[1]],
                    context.frames[i][elbow_names[2]],
                ),
            )
            for i in bottom_source
            if all(
                _visible(context.frames[i], name, cfg.visibility_threshold)
                for name in elbow_names
            )
        ]
        spec = specs["bench_elbow_flexion_2d"]
        if angles:
            source = [i for i, _ in angles]
            quality, states, origins = _quality(context.frames, elbow_names, source)
            minimum_frame, minimum_angle = min(angles, key=lambda item: item[1])
            results.append(
                MetricEvidence(
                    spec.metric_id,
                    1,
                    "bottom",
                    source,
                    elbow_names,
                    quality,
                    states,
                    origins,
                    {
                        "minimum_elbow_angle_degrees": round(minimum_angle, 2),
                        "minimum_angle_frame": minimum_frame,
                        "bottom_median_elbow_angle_degrees": round(
                            float(median(value for _, value in angles)), 2
                        ),
                        "description": "2D selected-side elbow-angle measurement",
                    },
                    None,
                    spec.decision_rule,
                    None,
                    None,
                    {"sufficient": True, "metric_specific": True},
                    None,
                    EvidenceStatus.SUFFICIENT,
                    None,
                    [],
                ).to_dict()
            )
        else:
            results.append(
                _insufficient(
                    spec,
                    elbow_side,
                    "Bottom phase lacks co-observed shoulder, elbow and wrist landmarks.",
                )
            )
        spec = specs["bench_wrist_trajectory_2d"]
        complete_phase = phases.get("complete")
        trajectory_end = (
            complete_phase["start_frame"]
            if complete_phase
            else phases["ascent"]["end_frame"]
        )
        source = [
            i
            for i in range(phases["top"]["start_frame"], trajectory_end + 1)
            if raw[i] is not None
        ]
        required_frames = (
            [
                phases["top"]["start_frame"],
                phase_diagnostics["bottom_frame"],
                complete_phase["start_frame"],
            ]
            if complete_phase
            else []
        )
        body_scale = context.calibration_profile.get("profile", {}).get("body_scale")
        if required_frames and body_scale and all(
            raw[i] is not None and raw_x[i] is not None for i in required_frames
        ):
            quality, states, origins = _quality(context.frames, [wrist_name], source)
            start_y, bottom_y, end_y = (float(smooth[i]) for i in required_frames)
            start_x, bottom_x, end_x = (float(smooth_x[i]) for i in required_frames)
            trajectory_points = [
                (float(smooth_x[i]), float(smooth[i]))
                for i in source
                if smooth_x[i] is not None and smooth[i] is not None
            ]
            path_length = sum(
                math.hypot(x2 - x1, y2 - y1)
                for (x1, y1), (x2, y2) in zip(trajectory_points, trajectory_points[1:])
            )
            results.append(
                MetricEvidence(
                    spec.metric_id,
                    1,
                    "complete bench repetition",
                    source,
                    [wrist_name],
                    quality,
                    states,
                    origins,
                    {
                        "proxy_label": "selected wrist trajectory / hand-and-bar-end proxy; not direct barbell tracking",
                        "start_wrist_y": round(start_y, 6),
                        "bottom_wrist_y": round(bottom_y, 6),
                        "end_wrist_y": round(end_y, 6),
                        "bottom_frame": required_frames[1],
                        "vertical_descent_displacement_normalised": round(
                            (bottom_y - start_y) / body_scale, 6
                        ),
                        "vertical_ascent_displacement_normalised": round(
                            (bottom_y - end_y) / body_scale, 6
                        ),
                        "end_to_start_vertical_difference_normalised": round(
                            (end_y - start_y) / body_scale, 6
                        ),
                        "horizontal_top_to_bottom_displacement_normalised": round(
                            (bottom_x - start_x) / body_scale, 6
                        ),
                        "horizontal_bottom_to_completion_displacement_normalised": round(
                            (end_x - bottom_x) / body_scale, 6
                        ),
                        "horizontal_top_to_completion_displacement_normalised": round(
                            (end_x - start_x) / body_scale, 6
                        ),
                        "end_2d_distance_from_start_normalised": round(
                            math.hypot(end_x - start_x, end_y - start_y) / body_scale, 6
                        ),
                        "total_2d_wrist_path_length_normalised": round(
                            path_length / body_scale, 6
                        ),
                        "body_scale": round(float(body_scale), 6),
                        # BENCH-01 field aliases retained for result consumers.
                        "descent_displacement": round(bottom_y - start_y, 6),
                        "ascent_displacement": round(bottom_y - end_y, 6),
                        "end_distance_from_start": round(abs(end_y - start_y), 6),
                    },
                    spec.normalisation_method,
                    spec.decision_rule,
                    spec.threshold,
                    None,
                    {"sufficient": True, "metric_specific": True},
                    None,
                    EvidenceStatus.SUFFICIENT,
                    None,
                    [],
                ).to_dict()
            )
        else:
            if not complete_phase:
                trajectory_warning = (
                    "A confirmed returned-top window is unavailable; the observed "
                    "ascent remains in the phase and signal diagnostics, but no "
                    "completion displacement was calculated."
                )
            elif not body_scale:
                trajectory_warning = (
                    "Arm-chain scale is unavailable for normalised wrist trajectory."
                )
            else:
                trajectory_warning = (
                    "Wrist evidence is missing at a required phase boundary."
                )
            results.append(
                _insufficient(
                    spec,
                    side,
                    trajectory_warning,
                )
            )

        spec = specs["bench_finish_geometry_similarity"]
        comparison_window = max(cfg.minimum_baseline_frames, cfg.smoothing_window)
        top_end = phases["top"]["end_frame"]
        top_source = list(
            range(
                max(phases["top"]["start_frame"], top_end - comparison_window + 1),
                top_end + 1,
            )
        )
        complete_source = (
            list(
                range(
                    complete_phase["start_frame"],
                    min(
                        complete_phase["end_frame"],
                        complete_phase["start_frame"] + comparison_window - 1,
                    )
                    + 1,
                )
            )
            if complete_phase
            else []
        )
        finish_selection = select_evidence_side(
            context.frames,
            ("shoulder", "elbow", "wrist"),
            cfg.visibility_threshold,
            top_source + complete_source,
        )
        finish_side = finish_selection["selected_side"] or side
        finish_wrist_name = f"{finish_side}_wrist"
        elbow_names = [
            f"{finish_side}_{name}" for name in ("shoulder", "elbow", "wrist")
        ]

        def region_geometry(
            source_frames: list[int],
        ) -> tuple[float, float, float] | None:
            observations = [
                (
                    calculate_angle(
                        context.frames[i][elbow_names[0]],
                        context.frames[i][elbow_names[1]],
                        context.frames[i][elbow_names[2]],
                    ),
                    float(context.frames[i][finish_wrist_name]["x"]),
                    float(context.frames[i][finish_wrist_name]["y"]),
                )
                for i in source_frames
                if all(
                    _visible(context.frames[i], name, cfg.visibility_threshold)
                    for name in elbow_names
                )
            ]
            if not observations:
                return None
            return tuple(float(median(values)) for values in zip(*observations))

        start_geometry = region_geometry(top_source)
        finish_geometry = region_geometry(complete_source)
        if body_scale and start_geometry and finish_geometry:
            source = top_source + complete_source
            quality, states, origins = _quality(context.frames, elbow_names, source)
            angle_difference = abs(finish_geometry[0] - start_geometry[0])
            wrist_difference = (
                math.hypot(
                    finish_geometry[1] - start_geometry[1],
                    finish_geometry[2] - start_geometry[2],
                )
                / body_scale
            )
            results.append(
                MetricEvidence(
                    spec.metric_id,
                    1,
                    "top and complete",
                    source,
                    elbow_names,
                    quality,
                    states,
                    origins,
                    {
                        "start_median_elbow_angle_degrees": round(start_geometry[0], 2),
                        "complete_median_elbow_angle_degrees": round(
                            finish_geometry[0], 2
                        ),
                        "elbow_angle_difference_degrees": round(angle_difference, 2),
                        "wrist_position_difference_normalised": round(
                            wrist_difference, 6
                        ),
                        "window_summary": "robust medians across detected TOP and COMPLETE regions",
                        "interpretation_scope": "self-reference only; not competition lockout judging",
                    },
                    spec.normalisation_method,
                    spec.decision_rule,
                    None,
                    None,
                    {"sufficient": True, "metric_specific": True},
                    None,
                    EvidenceStatus.SUFFICIENT,
                    None,
                    [],
                ).to_dict()
            )
        else:
            results.append(
                _insufficient(
                    spec,
                    finish_side,
                    (
                        "A confirmed returned-top window is unavailable; finish geometry was not compared."
                        if not complete_phase
                        else "Top and completion arm geometry or arm-chain scale is unavailable."
                    ),
                )
            )
    pause_result = next(
        result
        for result in results
        if result.get("metric_id") == "motionless_bottom_2d_proxy"
    )
    supported_results = [
        result
        for result in results
        if result.get("metric_id") != "motionless_bottom_2d_proxy"
    ]
    sufficient = sum(
        result["evidence_status"] == EvidenceStatus.SUFFICIENT.value
        for result in supported_results
    )
    status = (
        "invalid"
        if not phases
        else ("warning" if sufficient < len(supported_results) else "valid")
    )
    measurements = {
        result["metric_id"]: result["measurement"] for result in supported_results
    }
    competition_assessment_keys = {
        "observed_side_elbow_depth_2d_proxy": "bilateral_elbow_depth",
        "terminal_elbow_extension_2d_proxy": "bilateral_terminal_extension",
    }
    competition_rule_assessments = {
        competition_assessment_keys[result["metric_id"]]: result
        for result in results
        if result.get("metric_id") in competition_assessment_keys
    }
    fault_ids = {
        "DEPTH_INSUFFICIENT": "insufficient_elbow_depth",
        "INCOMPLETE": "incomplete_elbow_extension",
    }
    detected_faults = [
        {
            "fault_id": fault_ids[result["classification"]],
            "metric_id": result["metric_id"],
            "classification": result["classification"],
        }
        for result in competition_rule_assessments.values()
        if result.get("classification") in fault_ids
    ]
    calibration = attach_calibration_usage(
        context.calibration_profile,
        body_scale_consumers=[
            "bench_wrist_trajectory_2d descriptive measurement",
            "bench_wrist_trajectory_2d descriptive coaching",
        ],
        body_scale_classification_input=False,
        body_scale_coaching_input=True,
    )
    return {
        "lift": "bench_press",
        "analysis_status": status,
        "analysis_valid": status in {"valid", "warning"},
        "selected_side": side,
        "visibility": context.visibility,
        "recording_reliability": recording_reliability,
        "phase_frames": phases,
        "measurements": measurements,
        "metric_specifications": {
            key: spec.to_dict()
            for key, spec in specs.items()
            if key != "motionless_bottom_2d_proxy"
        },
        "metric_results": supported_results,
        "competition_rule_assessments": competition_rule_assessments,
        "experimental_diagnostics": {
            "motionless_bottom_2d_proxy": {
                "scope": "experimental_diagnostic_only",
                "coaching_eligible": False,
                "metric_specification": specs["motionless_bottom_2d_proxy"].to_dict(),
                "result": pause_result,
            }
        },
        "detected_faults": detected_faults,
        "calibration": calibration,
        "kinematic_self_calibration": calibration,
        "warnings": warnings,
        "limitations": [
            "These are 2D engineering proxies for selected competition-rule requirements, not competition judging outcomes.",
            "The terminal-extension angle boundaries are research-informed engineering proxies, not official IPF angle thresholds.",
        ],
        "processing_information": {
            "total_frames": len(context.frames),
            "usable_frames": recording_reliability["selected_side_usable_frames"],
            "usable_frame_ratio": recording_reliability[
                "selected_side_usable_frame_ratio"
            ],
            "visibility_threshold": cfg.visibility_threshold,
            "consumed_landmarks": list(BENCH_LANDMARKS),
            **context.extraction_information,
        },
        "diagnostics": {
            "supported_view": "side_or_near-side",
            "phase_signal": "selected-side wrist normalised image y",
            "raw_wrist_y": raw,
            "smoothed_wrist_y": smooth,
            "raw_wrist_x": raw_x,
            "smoothed_wrist_x": smooth_x,
            "phase_detection": phase_diagnostics,
        },
    }


def bench_diagnostics_view(analysis: dict[str, Any]) -> dict[str, Any]:
    """Return the compact development-only Bench diagnostics view model.

    This projection contains no calculations and cannot change evidence.  A UI
    can render it directly while the complete structured result remains the
    authoritative downloadable diagnostic artefact.
    """

    metric_results = {
        result.get("metric_id"): {
            "evidence_status": result.get("evidence_status"),
            "reliability": result.get("reliability"),
            "source_frames": result.get("source_frames"),
            "landmark_quality": result.get("landmark_quality"),
            "measurement": result.get("measurement"),
            "provenance": {
                "landmark_state": result.get("landmark_state"),
                "landmark_origin": result.get("landmark_origin"),
            },
            "warnings": result.get("warnings", []),
        }
        for result in analysis.get("metric_results", [])
    }
    calibration = analysis.get("calibration", {})
    return {
        "selected_side": analysis.get("selected_side"),
        "visibility": analysis.get("visibility", {}),
        "phase_frames": analysis.get("phase_frames", {}),
        "trajectory": {
            "signal": analysis.get("diagnostics", {}).get("phase_signal"),
            "raw": analysis.get("diagnostics", {}).get("raw_wrist_y", []),
            "smoothed": analysis.get("diagnostics", {}).get("smoothed_wrist_y", []),
            "phase_detection": analysis.get("diagnostics", {}).get(
                "phase_detection", {}
            ),
        },
        "elbow_angle_evidence": metric_results.get("bench_elbow_flexion_2d", {}),
        "wrist_trajectory_evidence": metric_results.get(
            "bench_wrist_trajectory_2d", {}
        ),
        "finish_geometry_evidence": metric_results.get(
            "bench_finish_geometry_similarity", {}
        ),
        "calibration_summary": {
            "diagnostic_only": calibration.get("diagnostic_only"),
            "usage": calibration.get("usage", {}),
            "status": calibration.get("status"),
            "profile": calibration.get("profile"),
        },
        "warnings": analysis.get("warnings", []),
    }


def bench_pause_diagnostics_export(analysis: dict[str, Any]) -> dict[str, Any]:
    """Project the bounded experimental pause audit without coaching findings."""

    pause = (
        analysis.get("experimental_diagnostics", {})
        .get("motionless_bottom_2d_proxy", {})
        .get("result", {})
    )
    measurement = pause.get("measurement") or {}
    return {
        "schema": "bench_pause_diagnostics_v1",
        "scope": "experimental_diagnostic_only",
        "coaching_eligible": False,
        "lift": analysis.get("lift"),
        "analysis_status": analysis.get("analysis_status"),
        "classification": pause.get("classification"),
        "candidate_frame_range": measurement.get("candidate_frame_range"),
        "required_stable_frames": measurement.get("required_stable_frames"),
        "required_stable_seconds": measurement.get("required_stable_seconds"),
        "longest_stable_interval": measurement.get("longest_stable_interval"),
        **(measurement.get("pause_diagnostics") or {}),
    }
