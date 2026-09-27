"""Motion analysis utilities for side-view squat landmark time series."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Literal

from .metric_contracts import (
    DepthClassification,
    EarlyAscentClassification,
    EvidenceStatus,
    LandmarkState,
    MetricEvidence,
    MetricSpecification,
    SensitivityEvaluation,
)
from .pose_calibration import attach_calibration_usage
from .lift_analysis import (
    LiftLandmarkConfiguration,
    prepare_lift_analysis_context,
    select_evidence_side,
)

Landmark = dict[str, float]
LandmarkFrame = dict[str, Landmark]
Phase = Literal["setup", "descent", "bottom", "ascent", "final_standing", "unknown"]

# Heel is omitted because no implemented squat measurement uses it.
SQUAT_LANDMARK_BASES = ("shoulder", "hip", "knee", "ankle", "foot_index")
SQUAT_SIDE_SELECTION_BASES = ("shoulder", "hip", "knee", "ankle")
SQUAT_LANDMARKS = tuple(
    f"{side}_{joint}" for side in ("left", "right") for joint in SQUAT_LANDMARK_BASES
)
SQUAT_OVERLAY_CONNECTION_BASES = (
    ("shoulder", "hip"),
    ("hip", "knee"),
    ("knee", "ankle"),
    ("ankle", "foot_index"),
)
SQUAT_OVERLAY_CONNECTIONS = tuple(
    (f"{side}_{start}", f"{side}_{end}")
    for side in ("left", "right")
    for start, end in SQUAT_OVERLAY_CONNECTION_BASES
) + (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
)

SQUAT_CALIBRATION_SEGMENTS = {
    "shoulder_hip": ("shoulder", "hip"),
    "hip_knee": ("hip", "knee"),
    "knee_ankle": ("knee", "ankle"),
}
SQUAT_LANDMARK_CONFIGURATION = LiftLandmarkConfiguration(
    retained_landmarks=SQUAT_LANDMARKS,
    side_selection_bases=SQUAT_SIDE_SELECTION_BASES,
    calibration_segments=SQUAT_CALIBRATION_SEGMENTS,
)
PHASES: tuple[Phase, ...] = ("setup", "descent", "bottom", "ascent", "final_standing")
DEPTH_MARGIN_SENSITIVITY_VALUES = (0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045)
PROGRESS_DIFFERENCE_SENSITIVITY_VALUES = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24)
MINIMUM_DURATION_SENSITIVITY_VALUES = (0.10, 0.15, 0.20, 0.25, 0.30)


@dataclass(frozen=True)
class SquatAnalysisConfig:
    visibility_threshold: float = 0.6
    minimum_side_visibility_score: float = 0.6
    warning_usable_frame_ratio: float = 0.9
    minimum_usable_frame_ratio: float = 0.75
    smoothing_window: int = 5
    setup_baseline_start_seconds: float = 0.5
    setup_baseline_end_seconds: float = 1.0
    setup_baseline_seconds: float = 0.5
    final_standing_window_seconds: float = 0.5
    stable_velocity_threshold_per_second: float = 0.06
    descent_velocity_threshold_per_second: float = 0.015
    ascent_velocity_threshold_per_second: float = 0.015
    motion_velocity_threshold_per_second: float = 0.08
    sustained_motion_seconds: float = 0.1
    minimum_descent_depth: float = 0.03
    bottom_near_max_hip_y_ratio: float = 0.08
    standing_height_tolerance_ratio: float = 0.15
    minimum_rep_hip_travel: float = 0.08
    depth_margin: float = 0.03
    bottom_pause_threshold_seconds: float = 1.0
    sticking_low_speed_ratio: float = 0.35
    sticking_min_duration_seconds: float = 0.3
    prolonged_ascent_ratio: float = 1.5
    limited_hip_min_knee_flexion_change: float = 35.0
    limited_hip_max_hip_to_knee_ratio: float = 0.55
    hip_first_ascent_fraction: float = 0.35
    hip_first_progress_threshold: float = 0.18
    hip_first_min_duration_seconds: float = 0.2


PROTOTYPE_THRESHOLD_SOURCE = {
    "threshold_source_type": "prototype_calibration",
    "threshold_source_reference": None,
}


def squat_metric_specifications(
    config: SquatAnalysisConfig | None = None,
) -> dict[str, MetricSpecification]:
    """Return the two currently migrated squat metric specifications."""

    cfg = config or SquatAnalysisConfig()
    common_reliability = {
        "minimum_landmark_visibility": cfg.visibility_threshold,
        "missing_landmarks": "not_permitted_in_source_frames",
    }

    return {
        "squat_depth_2d_proxy": MetricSpecification(
            metric_id="squat_depth_2d_proxy",
            lift="squat",
            required_landmarks=("hip", "knee"),
            required_phase="bottom",
            supported_view="side_or_near-side",
            measurement_key="bottom_hip_y_minus_knee_y",
            normalisation_method="MediaPipe normalised image y coordinates",
            decision_rule="DEPTH_ACHIEVED above +margin; BORDERLINE within +/-margin; otherwise DEPTH_NOT_ACHIEVED",
            threshold={"depth_margin": cfg.depth_margin},
            threshold_source=dict(PROTOTYPE_THRESHOLD_SOURCE),
            threshold_version="legacy-unversioned",
            reliability_requirements=common_reliability,
            missingness_policy="insufficient evidence unless hip and knee coexist in a bottom source frame",
            sensitivity_parameters={
                "depth_margin": list(DEPTH_MARGIN_SENSITIVITY_VALUES)
            },
            evidence_sources=("pose_estimation", "phase_state_machine"),
        ),
        "squat_early_ascent_coordination": MetricSpecification(
            metric_id="squat_early_ascent_coordination",
            lift="squat",
            required_landmarks=("hip", "shoulder"),
            required_phase="ascent",
            supported_view="side_or_near-side",
            measurement_key="maximum_normalised_hip_minus_shoulder_progress",
            normalisation_method="progress relative to each landmark's full ascent displacement",
            decision_rule="HIP_FIRST_PATTERN_DETECTED when progress difference meets threshold for minimum duration; otherwise HIP_FIRST_PATTERN_NOT_DETECTED",
            threshold={
                "progress_difference": cfg.hip_first_progress_threshold,
                "minimum_duration_seconds": cfg.hip_first_min_duration_seconds,
                "early_ascent_fraction": cfg.hip_first_ascent_fraction,
            },
            threshold_source=dict(PROTOTYPE_THRESHOLD_SOURCE),
            threshold_version="legacy-unversioned",
            reliability_requirements=common_reliability,
            missingness_policy="insufficient evidence unless hip and shoulder support normalised early-ascent progress",
            sensitivity_parameters={
                "progress_difference": list(PROGRESS_DIFFERENCE_SENSITIVITY_VALUES),
                "minimum_duration_seconds": list(MINIMUM_DURATION_SENSITIVITY_VALUES),
            },
            evidence_sources=(
                "pose_estimation",
                "median_smoothed_signals",
                "phase_state_machine",
            ),
        ),
    }


def classify_squat_depth(measurement: float, margin: float) -> DepthClassification:
    """Apply the production depth rule with an explicitly supplied margin."""

    if measurement > margin:
        return DepthClassification.DEPTH_ACHIEVED
    if measurement >= -margin:
        return DepthClassification.BORDERLINE
    return DepthClassification.DEPTH_NOT_ACHIEVED


def classify_early_ascent(
    progress_differences: list[float | None],
    fps: float,
    progress_difference_threshold: float,
    minimum_duration_seconds: float,
) -> dict[str, Any]:
    """Apply the production sustained-progress rule to an existing signal."""

    mask = [
        value is not None and value >= progress_difference_threshold
        for value in progress_differences
    ]
    observed = [value for value in progress_differences if value is not None]
    qualifying_runs = [
        run
        for run in _runs(mask)
        if (run[1] - run[0] + 1) / fps >= minimum_duration_seconds
    ]
    duration = (
        (qualifying_runs[0][1] - qualifying_runs[0][0] + 1) / fps
        if qualifying_runs
        else 0.0
    )
    detected = bool(qualifying_runs)
    return {
        "classification": (
            EarlyAscentClassification.HIP_FIRST_PATTERN_DETECTED
            if detected
            else EarlyAscentClassification.HIP_FIRST_PATTERN_NOT_DETECTED
        ),
        "detected": detected,
        "qualifying_duration_seconds": round(duration, 3),
        "maximum_progress_difference": (round(max(observed), 3) if observed else 0.0),
    }


def _summarise_sweep(
    parameter_name: str,
    production_value: float,
    values: tuple[float, ...],
    outcomes: list[dict[str, Any]],
    production_classification: str,
) -> dict[str, Any]:
    """Build a generic descriptive (not confidence-like) sensitivity summary."""

    classifications = [str(outcome["classification"]) for outcome in outcomes]
    same = sum(value == production_classification for value in classifications)
    crossings = [
        {
            "between": [values[index - 1], values[index]],
            "from": classifications[index - 1],
            "to": classifications[index],
        }
        for index in range(1, len(values))
        if classifications[index] != classifications[index - 1]
    ]
    lower_changes = [
        value
        for value, classification in zip(values, classifications)
        if value < production_value and classification != production_classification
    ]
    upper_changes = [
        value
        for value, classification in zip(values, classifications)
        if value > production_value and classification != production_classification
    ]
    stable = same == len(outcomes)
    return SensitivityEvaluation(
        parameter_name=parameter_name,
        production_value=production_value,
        evaluated_values=list(values),
        outcomes=outcomes,
        production_classification=production_classification,
        alternative_values_evaluated=len(values) - 1,
        same_classification_count=same,
        different_classification_count=len(outcomes) - same,
        classification_changes=len(crossings),
        stable_across_evaluated_range=stable,
        boundary_crossings=crossings,
        first_lower_value_where_classification_changes=(
            max(lower_changes) if lower_changes else None
        ),
        first_upper_value_where_classification_changes=(
            min(upper_changes) if upper_changes else None
        ),
        interpretation=(
            "classification stable across evaluated parameter range"
            if stable
            else "classification sensitive to threshold choice"
        ),
    ).to_dict()


def _depth_sensitivity(measurement: float, cfg: SquatAnalysisConfig) -> dict[str, Any]:
    production = classify_squat_depth(measurement, cfg.depth_margin).value
    outcomes = []
    for margin in DEPTH_MARGIN_SENSITIVITY_VALUES:
        classification = classify_squat_depth(measurement, margin).value
        outcomes.append(
            {
                "value": margin,
                "measurement": round(measurement, 6),
                "classification": classification,
                "distance_from_relevant_boundary": round(
                    min(abs(measurement - margin), abs(measurement + margin)), 6
                ),
                "is_production_value": margin == cfg.depth_margin,
            }
        )
    return {
        "performed": True,
        "method": "one_factor_at_a_time",
        "production_parameters": {"depth_margin": cfg.depth_margin},
        "parameter_sweeps": {
            "depth_margin": _summarise_sweep(
                "depth_margin",
                cfg.depth_margin,
                DEPTH_MARGIN_SENSITIVITY_VALUES,
                outcomes,
                production,
            )
        },
        "production_classification": production,
        "classification_stability": {
            "depth_margin": outcomes == []
            or all(item["classification"] == production for item in outcomes)
        },
        "limitations": [
            "Experimental sweep values are not calibration data and do not select or update a production threshold.",
            "One-factor-at-a-time analysis does not evaluate parameter interactions.",
        ],
    }


def _early_ascent_sensitivity(
    progress_differences: list[float | None], fps: float, cfg: SquatAnalysisConfig
) -> dict[str, Any]:
    production_result = classify_early_ascent(
        progress_differences,
        fps,
        cfg.hip_first_progress_threshold,
        cfg.hip_first_min_duration_seconds,
    )
    production = production_result["classification"].value

    def outcome(threshold: float, duration: float, value: float) -> dict[str, Any]:
        result = classify_early_ascent(progress_differences, fps, threshold, duration)
        return {
            "value": value,
            "threshold": threshold,
            "maximum_observed_progress_difference": result[
                "maximum_progress_difference"
            ],
            "qualifying_duration_seconds": result["qualifying_duration_seconds"],
            "duration_requirement_seconds": duration,
            "classification": result["classification"].value,
        }

    progress_outcomes = [
        {
            **outcome(value, cfg.hip_first_min_duration_seconds, value),
            "is_production_value": value == cfg.hip_first_progress_threshold,
        }
        for value in PROGRESS_DIFFERENCE_SENSITIVITY_VALUES
    ]
    duration_outcomes = [
        {
            **outcome(cfg.hip_first_progress_threshold, value, value),
            "is_production_value": value == cfg.hip_first_min_duration_seconds,
        }
        for value in MINIMUM_DURATION_SENSITIVITY_VALUES
    ]
    return {
        "performed": True,
        "method": "one_factor_at_a_time",
        "production_parameters": {
            "progress_difference": cfg.hip_first_progress_threshold,
            "minimum_duration_seconds": cfg.hip_first_min_duration_seconds,
            "early_ascent_fraction": cfg.hip_first_ascent_fraction,
        },
        "parameter_sweeps": {
            "progress_difference": _summarise_sweep(
                "progress_difference",
                cfg.hip_first_progress_threshold,
                PROGRESS_DIFFERENCE_SENSITIVITY_VALUES,
                progress_outcomes,
                production,
            ),
            "minimum_duration_seconds": _summarise_sweep(
                "minimum_duration_seconds",
                cfg.hip_first_min_duration_seconds,
                MINIMUM_DURATION_SENSITIVITY_VALUES,
                duration_outcomes,
                production,
            ),
        },
        "production_classification": production,
        "classification_stability": {
            "progress_difference": all(
                item["classification"] == production for item in progress_outcomes
            ),
            "minimum_duration_seconds": all(
                item["classification"] == production for item in duration_outcomes
            ),
        },
        "limitations": [
            "Experimental sweep values are not calibration data and do not select or update production parameters.",
            "Each parameter varies alone while all other production parameters remain fixed; interactions are not evaluated.",
            "The already-computed early-ascent signal and phases are reused; pose inference is not repeated.",
        ],
    }


def calculate_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Return the angle at point b formed by points a-b-c, in degrees."""
    ab_x = float(a["x"]) - float(b["x"])
    ab_y = float(a["y"]) - float(b["y"])
    cb_x = float(c["x"]) - float(b["x"])
    cb_y = float(c["y"]) - float(b["y"])
    ab_length = math.hypot(ab_x, ab_y)
    cb_length = math.hypot(cb_x, cb_y)
    if ab_length == 0 or cb_length == 0:
        return float("nan")
    cosine = max(-1.0, min(1.0, (ab_x * cb_x + ab_y * cb_y) / (ab_length * cb_length)))
    return math.degrees(math.acos(cosine))


def _has_visible_joints(
    landmarks: LandmarkFrame, joints: Iterable[str], visibility_threshold: float
) -> bool:
    return all(
        (lm := landmarks.get(joint)) is not None
        and float(lm.get("visibility", 0.0)) >= visibility_threshold
        and float(lm.get("presence", 1.0)) >= visibility_threshold
        for joint in joints
    )


def _torso_lean_degrees(shoulder: Landmark, hip: Landmark) -> float:
    dx = float(shoulder["x"]) - float(hip["x"])
    dy = float(shoulder["y"]) - float(hip["y"])
    if dx == 0 and dy == 0:
        return float("nan")
    return abs(math.degrees(math.atan2(dx, -dy)))


def _median_smooth(values: list[float | None], window: int = 5) -> list[float | None]:
    if window < 3 or window % 2 == 0:
        return values
    half = window // 2
    out: list[float | None] = []
    for i in range(len(values)):
        sample = [
            v
            for v in values[max(0, i - half) : i + half + 1]
            if v is not None and math.isfinite(v)
        ]
        out.append(float(median(sample)) if sample else None)
    return out


def _finite_values(values: Iterable[Any]) -> list[float]:
    return [
        float(v)
        for v in values
        if isinstance(v, (float, int)) and math.isfinite(float(v))
    ]


def _stats(
    metrics: list[dict[str, Any]], indices: Iterable[int], key: str
) -> dict[str, float | None]:
    vals = _finite_values(metrics[i].get(key) for i in indices if 0 <= i < len(metrics))
    return {
        "median": round(float(median(vals)), 1) if vals else None,
        "min": round(min(vals), 1) if vals else None,
        "max": round(max(vals), 1) if vals else None,
    }


def _median_metric(
    metrics: list[dict[str, Any]], indices: Iterable[int], key: str
) -> float | None:
    vals = _finite_values(metrics[i].get(key) for i in indices if 0 <= i < len(metrics))
    return float(median(vals)) if vals else None


def calculate_frame_metrics(
    frame_landmarks: LandmarkFrame,
    visibility_threshold: float = 0.6,
    side: str = "left",
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "valid": False,
        "landmark_confidence": None,
        "left_knee_angle": None,
        "left_hip_angle": None,
        "torso_lean_degrees": None,
        "hip_y": None,
        "shoulder_y": None,
        "knee_y": None,
        "knee_x_relative_to_toe": None,
        "available_landmarks": [],
        "landmark_quality": {},
    }
    names = {joint: f"{side}_{joint}" for joint in SQUAT_LANDMARK_BASES}
    available = {
        joint: frame_landmarks[name]
        for joint, name in names.items()
        if _has_visible_joints(frame_landmarks, (name,), visibility_threshold)
    }
    metrics["available_landmarks"] = [names[joint] for joint in available]
    metrics["landmark_quality"] = {
        names[joint]: {
            "visibility": round(float(landmark.get("visibility", 0.0)), 3),
            "presence": round(float(landmark.get("presence", 1.0)), 3),
        }
        for joint, landmark in available.items()
    }
    confidences = [float(lm.get("visibility", 0.0)) for lm in available.values()]
    metrics["landmark_confidence"] = (
        round(float(median(confidences)), 3) if confidences else None
    )
    # Hip availability alone defines phase-trajectory usability. Every other signal
    # has its own dependency set and remains independent of ankle availability.
    if (hip := available.get("hip")) is not None:
        metrics.update({"valid": True, "hip_y": float(hip["y"])})
    if (shoulder := available.get("shoulder")) is not None:
        metrics["shoulder_y"] = float(shoulder["y"])
    if (knee := available.get("knee")) is not None:
        metrics["knee_y"] = float(knee["y"])
    if shoulder is not None and hip is not None:
        metrics["torso_lean_degrees"] = _torso_lean_degrees(shoulder, hip)
    if (
        hip is not None
        and knee is not None
        and (ankle := available.get("ankle")) is not None
    ):
        metrics["left_knee_angle"] = calculate_angle(hip, knee, ankle)
    if shoulder is not None and hip is not None and knee is not None:
        metrics["left_hip_angle"] = calculate_angle(shoulder, hip, knee)
    toe = available.get("foot_index")
    if knee is not None and toe is not None:
        metrics["knee_x_relative_to_toe"] = float(knee["x"]) - float(toe["x"])
    return metrics


def _velocity(values: list[float | None], fps: float) -> list[float | None]:
    out = [None] * len(values)
    for i in range(1, len(values)):
        if values[i] is not None and values[i - 1] is not None:
            out[i] = (values[i] - values[i - 1]) * fps
    return out


def _runs(mask: list[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for i, ok in enumerate(mask + [False]):
        if ok and start is None:
            start = i
        if not ok and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def _segmentation_debug_template(cfg: SquatAnalysisConfig) -> dict[str, Any]:
    return {
        "baseline_hip_y": 0.0,
        "baseline_start_frame": 0,
        "baseline_end_frame": 0,
        "baseline_calculation_method": "median smoothed hip_y from fixed initial stable setup window",
        "minimum_descent_depth_threshold": cfg.minimum_descent_depth,
        "descent_velocity_threshold_per_second": cfg.descent_velocity_threshold_per_second,
        "ascent_velocity_threshold_per_second": cfg.ascent_velocity_threshold_per_second,
        "required_sustained_frames": 0,
        "maximum_depth_from_baseline": 0.0,
        "first_descent_candidate_frame": None,
        "first_bottom_candidate_frame": None,
        "first_ascent_candidate_frame": None,
        "final_standing_candidate_frame": None,
        "failure_reason": None,
    }


def _first_sustained(
    mask: list[bool], min_run: int, start_after: int = -1
) -> tuple[int, int] | None:
    for s, e in _runs(mask):
        if s > start_after and e - s + 1 >= min_run:
            return s, e
    return None


def _phase_state_machine(
    hip_y: list[float | None],
    fps: float,
    cfg: SquatAnalysisConfig,
    valid_indices: list[int],
) -> tuple[list[Phase], dict[str, Any]]:
    n = len(hip_y)
    phases: list[Phase] = ["unknown"] * n
    debug = _segmentation_debug_template(cfg)
    if not valid_indices:
        debug["failure_reason"] = "no_valid_frames"
        return phases, {"full_squat": False, "segmentation_debug": debug}

    vel = _velocity(hip_y, fps)  # normalized image-coordinate units per second
    min_run = max(1, round(cfg.sustained_motion_seconds * fps))
    debug["required_sustained_frames"] = min_run

    # Freeze the standing baseline once from an initial setup window. Prefer 0.5-1.0s;
    # for short clips/tests fall back to the first configured setup-baseline duration.
    preferred_start = round(cfg.setup_baseline_start_seconds * fps)
    preferred_end = max(
        preferred_start, round(cfg.setup_baseline_end_seconds * fps) - 1
    )
    initial_end = min(valid_indices[-1], preferred_end)
    stable_initial = [
        i <= initial_end
        and hip_y[i] is not None
        and (
            vel[i] is None
            or abs(float(vel[i])) <= cfg.stable_velocity_threshold_per_second
        )
        for i in range(n)
    ]
    stable_runs = [
        r
        for r in _runs(stable_initial)
        if r[1] - r[0] + 1 >= max(1, round(cfg.setup_baseline_seconds * fps))
    ]
    if stable_runs:
        s, e = stable_runs[0]
        baseline_window = [
            i for i in valid_indices if s <= i <= e and hip_y[i] is not None
        ]
    elif valid_indices[-1] >= preferred_end:
        baseline_window = [
            i
            for i in valid_indices
            if preferred_start <= i <= preferred_end and hip_y[i] is not None
        ]
    else:
        baseline_window = []
    if not baseline_window:
        fallback_count = max(
            1,
            min(
                round(cfg.setup_baseline_seconds * fps), max(2, len(valid_indices) // 4)
            ),
        )
        baseline_window = [
            i for i in valid_indices[:fallback_count] if hip_y[i] is not None
        ]
    baseline = (
        float(median(_finite_values(hip_y[i] for i in baseline_window)))
        if baseline_window
        else float(hip_y[valid_indices[0]] or 0.0)
    )
    debug.update(
        {
            "baseline_hip_y": baseline,
            "baseline_start_frame": (
                baseline_window[0] if baseline_window else valid_indices[0]
            ),
            "baseline_end_frame": (
                baseline_window[-1] if baseline_window else valid_indices[0]
            ),
            "maximum_depth_from_baseline": max(
                [
                    float(hip_y[i]) - baseline
                    for i in valid_indices
                    if hip_y[i] is not None
                ]
                or [0.0]
            ),
        }
    )

    depth_ok = [
        hip_y[i] is not None
        and (float(hip_y[i]) - baseline) >= cfg.minimum_descent_depth
        for i in range(n)
    ]
    down = [
        depth_ok[i]
        and vel[i] is not None
        and float(vel[i]) >= cfg.descent_velocity_threshold_per_second
        for i in range(n)
    ]
    descent_run = _first_sustained(down, min_run)
    if descent_run is None:
        for i in valid_indices:
            phases[i] = "setup"
        debug["failure_reason"] = "no_sustained_descent_from_frozen_baseline"
        return phases, {
            "full_squat": False,
            "segmentation_debug": debug,
        }
    descent_start = descent_run[0]
    debug["first_descent_candidate_frame"] = descent_start

    post_descent = [
        i for i in valid_indices if i >= descent_start and hip_y[i] is not None
    ]
    peak_value = max(float(hip_y[i]) for i in post_descent)
    peak_frame = max(i for i in post_descent if float(hip_y[i]) == peak_value)
    max_y = float(hip_y[peak_frame])
    travel = max_y - baseline
    near_max_tol = max(cfg.bottom_near_max_hip_y_ratio * max(travel, 1e-6), 0.01)
    bottom_mask = [
        i >= descent_start
        and hip_y[i] is not None
        and max_y - float(hip_y[i]) <= near_max_tol
        and (
            vel[i] is None
            or abs(float(vel[i])) <= cfg.stable_velocity_threshold_per_second
        )
        for i in range(n)
    ]
    bottom_runs = [
        r for r in _runs(bottom_mask) if r[1] >= peak_frame - max(1, round(0.25 * fps))
    ]
    if bottom_runs:
        bottom_start, bottom_end = min(
            bottom_runs, key=lambda r: abs(((r[0] + r[1]) // 2) - peak_frame)
        )
    else:
        # Make bottom contiguous around the deepest point using the near-maximum depth band.
        bottom_start = peak_frame
        bottom_end = peak_frame
        while (
            bottom_start > descent_start
            and hip_y[bottom_start - 1] is not None
            and max_y - float(hip_y[bottom_start - 1]) <= near_max_tol
        ):
            bottom_start -= 1
        while (
            bottom_end + 1 < n
            and hip_y[bottom_end + 1] is not None
            and max_y - float(hip_y[bottom_end + 1]) <= near_max_tol
        ):
            bottom_end += 1
    debug["first_bottom_candidate_frame"] = bottom_start

    up = [
        i > bottom_start
        and vel[i] is not None
        and float(vel[i]) <= -cfg.ascent_velocity_threshold_per_second
        for i in range(n)
    ]
    ascent_run = _first_sustained(up, min_run, start_after=bottom_start)
    if ascent_run is None:
        ascent_start = min(bottom_end + 1, valid_indices[-1])
    else:
        ascent_start = max(bottom_start + 1, ascent_run[0])
        bottom_end = min(bottom_end, ascent_start - 1)
    debug["first_ascent_candidate_frame"] = ascent_start if ascent_start < n else None

    final_tol = max(cfg.standing_height_tolerance_ratio * max(travel, 1e-6), 0.02)
    stable_run = max(1, round(cfg.final_standing_window_seconds * fps))
    final_position_mask = [
        i > ascent_start
        and hip_y[i] is not None
        and abs(float(hip_y[i]) - baseline) <= final_tol
        for i in range(n)
    ]
    final_start = None
    for s, e in _runs(final_position_mask):
        if s <= ascent_start:
            continue
        # Prefer at least 0.5s of stable standing. If the clip ends while already
        # standing, accept the available trailing stable run instead of rejecting a
        # visibly completed single rep solely because there are fewer post-return frames.
        required = stable_run if e < n - 1 else min(stable_run, e - s + 1)
        low_speed_frames = [
            i
            for i in range(s, e + 1)
            if vel[i] is None
            or abs(float(vel[i])) <= cfg.stable_velocity_threshold_per_second
        ]
        if e - s + 1 >= required and (
            len(low_speed_frames) >= max(1, required - 1) or e == n - 1
        ):
            final_start = s
            break
    debug["final_standing_candidate_frame"] = final_start

    for i in valid_indices:
        if i < descent_start:
            phases[i] = "setup"
        elif i < bottom_start:
            phases[i] = "descent"
        elif i <= bottom_end:
            phases[i] = "bottom"
        elif final_start is not None and i >= final_start:
            phases[i] = "final_standing"
        else:
            phases[i] = "ascent"

    full_squat = (
        travel >= cfg.minimum_rep_hip_travel
        and final_start is not None
        and ascent_start > bottom_start
    )
    if not full_squat:
        debug["failure_reason"] = "incomplete_rep_or_insufficient_travel"
    return phases, {
        "full_squat": full_squat,
        "bottom_window_frames": [bottom_start, bottom_end],
        "representative_bottom_window_frames": [
            max(0, peak_frame - 2),
            min(n - 1, peak_frame + 2),
        ],
        "representative_bottom_frame": peak_frame,
        "setup_baseline_frames": [
            debug["baseline_start_frame"],
            debug["baseline_end_frame"],
        ],
        "baseline_hip_y": baseline,
        "final_standing_established": final_start is not None,
        "hip_travel": travel,
        "segmentation_debug": debug,
    }


def _phase_summaries(
    phases: list[Phase], metrics: list[dict[str, Any]], fps: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    timings = {}
    angles = {}
    for phase in PHASES:
        idx = [i for i, p in enumerate(phases) if p == phase]
        timings[phase] = {
            "start_frame": idx[0] if idx else None,
            "end_frame": idx[-1] if idx else None,
            "start_timestamp_seconds": round(idx[0] / fps, 3) if idx and fps else None,
            "end_timestamp_seconds": round(idx[-1] / fps, 3) if idx and fps else None,
            "duration_seconds": round((len(idx) / fps), 3) if fps else None,
        }
        angles[phase] = {
            "knee_angle": _stats(metrics, idx, "left_knee_angle"),
            "hip_angle": _stats(metrics, idx, "left_hip_angle"),
            "torso_lean_degrees": _stats(metrics, idx, "torso_lean_degrees"),
        }
    return timings, angles


def _movement_angle_summary(
    phases: list[Phase], metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarise angle evidence only across descent, bottom, and ascent."""
    movement_phases = ("descent", "bottom", "ascent")
    interval = [i for i, phase in enumerate(phases) if phase in movement_phases]
    bounds = {
        "start_frame": interval[0] if interval else None,
        "end_frame": interval[-1] if interval else None,
        "frame_count": len(interval),
        "phases": list(movement_phases),
    }

    def minimum(metric_key: str) -> dict[str, Any]:
        observations = [
            (i, metrics[i].get(metric_key))
            for i in interval
            if isinstance(metrics[i].get(metric_key), (int, float))
            and math.isfinite(float(metrics[i][metric_key]))
        ]
        observed, expected = len(observations), len(interval)
        if not expected:
            status = "unavailable"
            reason = "No detected descent-to-ascent movement interval was available."
        elif not observed:
            status = "unavailable"
            reason = "No usable angle observations were available during the detected movement interval."
        elif observed < expected:
            status = "partial"
            reason = "Minimum among available observations; angle evidence covers only part of the movement interval."
        else:
            status, reason = "complete", None
        return {
            "value_degrees": round(min(value for _, value in observations), 1) if observations else None,
            "evidence_status": status,
            "observed_frames": observed,
            "interval_frames": expected,
            "coverage_ratio": round(observed / expected, 3) if expected else 0.0,
            "source_frames": [i for i, _ in observations],
            "description": "Minimum among available observations in the movement interval" if observations else "Unavailable",
            "unavailable_or_partial_reason": reason,
        }

    return {
        "movement_interval": bounds,
        "minimum_knee_angle": minimum("left_knee_angle"),
        "minimum_hip_angle": minimum("left_hip_angle"),
    }


def _finding(
    priority: int,
    name: str,
    status: str,
    confidence: str,
    evidence: str,
    numeric: dict[str, Any],
) -> dict[str, Any]:
    return {
        "priority": priority,
        "name": name,
        "status": status,
        "confidence": confidence,
        "evidence": evidence,
        "numeric_evidence": numeric,
    }


def _metric_landmark_quality(
    metrics: list[dict[str, Any]],
    source_frames: Iterable[int],
    landmark_names: Iterable[str],
) -> dict[str, Any]:
    """Expose per-frame pose quality without filling missing observations."""

    frames = [i for i in source_frames if 0 <= i < len(metrics)]
    return {
        name: {
            "source_frame_count": len(frames),
            "usable_frame_count": sum(
                1 for i in frames if name in metrics[i].get("landmark_quality", {})
            ),
            "by_frame": {
                str(i): metrics[i].get("landmark_quality", {}).get(name)
                for i in frames
                if name in metrics[i].get("landmark_quality", {})
            },
        }
        for name in landmark_names
    }


def _landmark_states_from_quality(
    landmark_quality: dict[str, Any],
) -> dict[str, LandmarkState]:
    """Map actual pose observations to evidence state; interpolation is not used."""

    return {
        name: (
            LandmarkState.OBSERVED
            if details.get("usable_frame_count", 0) > 0
            else LandmarkState.UNAVAILABLE
        )
        for name, details in landmark_quality.items()
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return a linearly interpolated percentile without adding a dependency."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metric_trajectory_reliability(
    frames: list[LandmarkFrame],
    evaluation_frames: Iterable[int],
    landmark_names: tuple[str, str],
    visibility_threshold: float,
) -> dict[str, Any]:
    """Describe raw metric dependencies without applying a reliability gate.

    Availability deliberately uses the same visibility/presence rule as existing
    metric evidence. Quality summaries retain every emitted MediaPipe observation,
    including observations below that rule. Displacements only join consecutive
    available frames, so a dropout is not misreported as a single-frame jump.
    """

    window = sorted({i for i in evaluation_frames if 0 <= i < len(frames)})

    def raw_landmark(frame_index: int, name: str) -> Landmark | None:
        landmark = frames[frame_index].get(name)
        if landmark is None or not all(
            isinstance(landmark.get(axis), (int, float))
            and math.isfinite(float(landmark[axis]))
            for axis in ("x", "y")
        ):
            return None
        return landmark

    def available(frame_index: int, name: str) -> bool:
        landmark = raw_landmark(frame_index, name)
        return bool(
            landmark is not None
            and float(landmark.get("visibility", 0.0)) >= visibility_threshold
            and float(landmark.get("presence", 1.0)) >= visibility_threshold
        )

    segment_by_frame: dict[int, float] = {}
    first, second = landmark_names
    for i in window:
        if available(i, first) and available(i, second):
            a, b = frames[i][first], frames[i][second]
            segment_by_frame[i] = math.hypot(
                float(a["x"]) - float(b["x"]),
                float(a["y"]) - float(b["y"]),
            )
    segment_values = list(segment_by_frame.values())
    normaliser = float(median(segment_values)) if segment_values else None

    availability: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    temporal: dict[str, Any] = {}
    raw_trajectories: dict[str, Any] = {}
    for name in landmark_names:
        available_frames = [i for i in window if available(i, name)]
        missing_runs = _runs([not available(i, name) for i in window])
        longest = max(missing_runs, key=lambda run: run[1] - run[0], default=None)
        availability[name] = {
            "evaluation_window_frame_count": len(window),
            "available_frame_count": len(available_frames),
            "availability_ratio": (
                round(len(available_frames) / len(window), 6) if window else None
            ),
            "longest_consecutive_dropout_frames": (
                longest[1] - longest[0] + 1 if longest else 0
            ),
            "longest_dropout_start_frame": window[longest[0]] if longest else None,
            "longest_dropout_end_frame": window[longest[1]] if longest else None,
        }
        observed = [
            frames[i][name] for i in window if raw_landmark(i, name) is not None
        ]
        visibilities = _finite_values(item.get("visibility") for item in observed)
        presences = _finite_values(item.get("presence", 1.0) for item in observed)
        quality[name] = {
            "observed_frame_count": len(observed),
            "median_visibility": (
                round(float(median(visibilities)), 6) if visibilities else None
            ),
            "minimum_visibility": round(min(visibilities), 6) if visibilities else None,
            "median_presence": (
                round(float(median(presences)), 6) if presences else None
            ),
            "minimum_presence": round(min(presences), 6) if presences else None,
            "interpretation": "MediaPipe quality signal; not a calibrated correctness probability",
        }
        raw_trajectories[name] = {
            str(i): {
                "x": round(float(frames[i][name]["x"]), 6),
                "y": round(float(frames[i][name]["y"]), 6),
            }
            for i in available_frames
        }
        movements: list[tuple[int, float]] = []
        available_set = set(available_frames)
        for i in available_frames:
            if i - 1 in available_set:
                current, previous = frames[i][name], frames[i - 1][name]
                movements.append(
                    (
                        i,
                        math.hypot(
                            float(current["x"]) - float(previous["x"]),
                            float(current["y"]) - float(previous["y"]),
                        ),
                    )
                )
        raw_values = [value for _, value in movements]
        normalised = (
            [value / normaliser for value in raw_values]
            if normaliser and normaliser > 0
            else []
        )
        largest = max(movements, key=lambda item: item[1], default=None)
        temporal[name] = {
            "consecutive_frame_pair_count": len(movements),
            "normalisation_segment": f"{first}-{second}",
            "normalisation_median_segment_length": (
                round(normaliser, 6) if normaliser is not None else None
            ),
            "median_normalised_frame_displacement": (
                round(float(median(normalised)), 6) if normalised else None
            ),
            "p95_normalised_frame_displacement": (
                round(float(_percentile(normalised, 0.95)), 6) if normalised else None
            ),
            "maximum_normalised_frame_displacement": (
                round(max(normalised), 6) if normalised else None
            ),
            "largest_single_frame_jump": (
                round(largest[1] / normaliser, 6) if largest and normaliser else None
            ),
            "largest_single_frame_jump_raw": round(largest[1], 6) if largest else None,
            "largest_single_frame_jump_frame": largest[0] if largest else None,
            "raw_displacement_median": (
                round(float(median(raw_values)), 6) if raw_values else None
            ),
            "raw_displacement_p95": (
                round(float(_percentile(raw_values, 0.95)), 6) if raw_values else None
            ),
            "raw_displacement_maximum": (
                round(max(raw_values), 6) if raw_values else None
            ),
        }

    segment_median = normaliser
    deviations = (
        [abs(value - segment_median) for value in segment_values]
        if segment_median is not None
        else []
    )
    changes = [
        (i, abs(segment_by_frame[i] - segment_by_frame[i - 1]))
        for i in window
        if i in segment_by_frame and i - 1 in segment_by_frame
    ]
    largest_change = max(changes, key=lambda item: item[1], default=None)
    geometry = {
        "segment": f"{first}-{second}",
        "frames_with_segment": len(segment_values),
        "median_segment_length": (
            round(segment_median, 6) if segment_median is not None else None
        ),
        "relative_median_absolute_deviation": (
            round(float(median(deviations)) / segment_median, 6)
            if deviations and segment_median and segment_median > 0
            else None
        ),
        "p95_absolute_deviation_from_median": (
            round(float(_percentile(deviations, 0.95)), 6) if deviations else None
        ),
        "maximum_frame_to_frame_segment_length_change": (
            round(largest_change[1], 6) if largest_change else None
        ),
        "largest_segment_length_change_frame": (
            largest_change[0] if largest_change else None
        ),
        "segment_length_by_frame": {
            str(i): round(value, 6) for i, value in segment_by_frame.items()
        },
    }
    return {
        "existing_requirements_met": None,
        "reliability_gate_applied": False,
        "evaluation_window": {
            "start_frame": window[0] if window else None,
            "end_frame": window[-1] if window else None,
            "frame_count": len(window),
            "frames": window,
        },
        "availability": availability,
        "mediapipe_quality": quality,
        "temporal_stability": temporal,
        "geometric_stability": geometry,
        "raw_trajectories": raw_trajectories,
    }


def _insufficient_metric_evidence(
    spec: MetricSpecification,
    side: str | None,
    warning: str,
    *,
    repetition_id: int | None = None,
    phase: str | None = None,
    source_frames: list[int] | None = None,
) -> MetricEvidence:
    landmarks = [f"{side}_{name}" if side else name for name in spec.required_landmarks]
    return MetricEvidence(
        metric_id=spec.metric_id,
        repetition_id=repetition_id,
        phase=phase or spec.required_phase,
        source_frames=source_frames or [],
        landmarks_used=landmarks,
        landmark_quality={},
        landmark_state={name: LandmarkState.UNAVAILABLE for name in landmarks},
        landmark_origin={name: "MEDIAPIPE" for name in landmarks},
        measurement=None,
        normalisation=spec.normalisation_method,
        decision_rule=spec.decision_rule,
        threshold={
            **spec.threshold,
            **spec.threshold_source,
            "threshold_version": spec.threshold_version,
        },
        distance_from_boundary=None,
        reliability={"requirements_met": False, "reason": warning},
        sensitivity_result={
            "performed": False,
            "reason": "insufficient_metric_evidence",
        },
        evidence_status=EvidenceStatus.INSUFFICIENT,
        classification=None,
        warnings=[warning],
    )


def _unavailable_squat_metric_results(
    cfg: SquatAnalysisConfig, side: str | None, warning: str
) -> list[dict[str, Any]]:
    return [
        _insufficient_metric_evidence(spec, side, warning).to_dict()
        for spec in squat_metric_specifications(cfg).values()
    ]


def analyse_squat_landmarks(
    frames: list[LandmarkFrame],
    fps: float = 30.0,
    visibility_threshold: float | None = None,
    pose_counts: list[int] | None = None,
    processing_metadata: dict[str, Any] | None = None,
    config: SquatAnalysisConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SquatAnalysisConfig()
    metric_specs = squat_metric_specifications(cfg)
    threshold = (
        cfg.visibility_threshold
        if visibility_threshold is None
        else visibility_threshold
    )
    warnings = []
    context = prepare_lift_analysis_context(
        lift="squat",
        frames=frames,
        fps=fps,
        visibility_threshold=threshold,
        landmarks=SQUAT_LANDMARK_CONFIGURATION,
        processing_metadata=processing_metadata,
        minimum_side_visibility_score=cfg.minimum_side_visibility_score,
        minimum_usable_frame_ratio=cfg.minimum_usable_frame_ratio,
        phase_tracking_bases=("hip",),
    )
    frames = context.frames
    total_frames = len(frames)
    extraction_information = context.extraction_information
    visibility = context.visibility
    recording_reliability = context.recording_reliability
    selected_side = context.selected_side or ""
    kinematic_self_calibration = attach_calibration_usage(
        context.calibration_profile,
        body_scale_consumers=[],
        body_scale_classification_input=False,
        body_scale_coaching_input=False,
    )
    # Phase side is selected by the usable hip motion itself, not by aggregate
    # body-chain visibility.  Evaluate both whole-repetition candidates once and
    # keep the winner fixed for all phase boundaries.
    phase_candidates: list[tuple[bool, int, str, list[dict[str, Any]], list[float | None], list[Phase], dict[str, Any]]] = []
    for candidate_side in ("left", "right"):
        candidate_metrics = [
            calculate_frame_metrics(frame, threshold, candidate_side) for frame in frames
        ]
        candidate_valid = [i for i, value in enumerate(candidate_metrics) if value["valid"]]
        candidate_hip = _median_smooth(
            [value["hip_y"] if isinstance(value["hip_y"], float) else None for value in candidate_metrics],
            cfg.smoothing_window,
        )
        candidate_phases, candidate_state = _phase_state_machine(
            candidate_hip, fps, cfg, candidate_valid
        )
        phase_candidates.append((
            bool(candidate_state.get("full_squat")), len(candidate_valid), candidate_side,
            candidate_metrics, candidate_hip, candidate_phases, candidate_state,
        ))
    phase_candidates.sort(key=lambda item: (item[0], item[1], item[2] == "left"), reverse=True)
    best_phase = phase_candidates[0]
    if best_phase[1]:
        selected_side = best_phase[2]
        visibility["selected_side"] = selected_side
        recording_reliability["selected_side"] = selected_side
        recording_reliability["selected_side_usable_frames"] = best_phase[1]
        recording_reliability["selected_side_usable_frame_ratio"] = round(best_phase[1] / total_frames, 3) if total_frames else 0.0
        recording_reliability["rejection_reasons"] = []
    metrics = (
        best_phase[3]
        if selected_side
        else []
    )
    valid = [i for i, m in enumerate(metrics) if m["valid"]]
    missing_ratio = 1.0 - (len(valid) / total_frames) if total_frames else 1.0
    if missing_ratio > (1.0 - cfg.warning_usable_frame_ratio):
        warnings.append(
            "Some frames have missing or low-confidence hip landmarks required for phase detection; other measurements retain their own landmark availability."
        )
    if recording_reliability.get("warnings"):
        warnings.append("The declared camera view may limit individual side-view metrics; it does not reject phase tracking.")
    if (
        pose_counts
        and sum(1 for c in pose_counts if c > 1) / max(1, len(pose_counts)) > 0.2
    ):
        warnings.append(
            "More than one person was consistently detected; analysis assumes the first detected person."
        )
    if recording_reliability["rejection_reasons"]:
        reasons = list(warnings)
        if not valid:
            reasons.append(
                "No reliable landmark time series was available for squat analysis."
            )
        else:
            reasons.append(
                "Too few frames contained a usable selected-side landmark set."
            )
        return {
            "lift": "squat",
            "analysis_status": "invalid",
            "selected_side": selected_side or None,
            "visibility": visibility,
            "recording_reliability": recording_reliability,
            "phase_frames": {},
            "measurements": {},
            "metric_specifications": {
                key: spec.to_dict() for key, spec in metric_specs.items()
            },
            "metric_results": _unavailable_squat_metric_results(
                cfg,
                selected_side or None,
                "Phase detection could not obtain a sufficiently complete hip trajectory.",
            ),
            "detected_faults": [],
            "kinematic_self_calibration": kinematic_self_calibration,
            "processing_information": {
                "total_frames": total_frames,
                "usable_frames": len(valid),
                "usable_frame_definition": "selected-side hip available for phase trajectory",
                "usable_frame_ratio": (
                    round(len(valid) / total_frames, 3) if total_frames else 0.0
                ),
                "visibility_threshold": threshold,
                "minimum_side_visibility_score": cfg.minimum_side_visibility_score,
                "minimum_usable_frame_ratio": cfg.minimum_usable_frame_ratio,
                "consumed_landmarks": list(SQUAT_LANDMARKS),
                **extraction_information,
            },
            "quality": {
                "analysis_valid": False,
                "video_valid": False,
                "confidence": "low",
                "tracking_valid_frame_ratio": 0.0,
            },
            "reps_detected": 0,
            "rep": {},
            "reps": [],
            "warnings": reasons,
            "frames": [
                {
                    "frame": i,
                    "timestamp_seconds": round(i / fps, 3) if fps else None,
                    "phase": "unknown",
                    **metrics[i],
                }
                for i in range(len(metrics))
            ],
        }
    hip_sm = best_phase[4]
    phases, state = best_phase[5], best_phase[6]
    for i, m in enumerate(metrics):
        m["smoothed_hip_y"] = hip_sm[i]
    full = bool(state.get("full_squat"))
    if not full:
        warnings.append(
            "No full squat was confidently detected from hip-height change."
        )
    if not state.get("final_standing_established"):
        warnings.append(
            "Final standing could not be established; the clip may end early or not return close to baseline standing posture."
        )
    segmentation_debug = state.get(
        "segmentation_debug", _segmentation_debug_template(cfg)
    )
    if not full:
        confidence = "medium" if missing_ratio <= 0.25 else "low"
        max_t = _finite_values(m["torso_lean_degrees"] for m in metrics)
        movement_angles = _movement_angle_summary([], metrics)
        return {
            "lift": "squat",
            "analysis_status": "invalid",
            "selected_side": selected_side,
            "visibility": visibility,
            "recording_reliability": {
                **recording_reliability,
                "rejection_reasons": [
                    *recording_reliability["rejection_reasons"],
                    "incomplete_repetition",
                ],
            },
            "phase_frames": {},
            "measurements": {
                "minimum_knee_angle": None,
                "minimum_hip_angle": None,
                "movement_angle_summary": movement_angles,
            },
            "metric_specifications": {
                key: spec.to_dict() for key, spec in metric_specs.items()
            },
            "metric_results": _unavailable_squat_metric_results(
                cfg,
                selected_side,
                "A complete squat phase sequence was not detected, so the required metric phases are unavailable.",
            ),
            "detected_faults": [],
            "kinematic_self_calibration": kinematic_self_calibration,
            "processing_information": {
                "total_frames": total_frames,
                "usable_frames": len(valid),
                "usable_frame_definition": "selected-side hip available for phase trajectory",
                "usable_frame_ratio": round(len(valid) / total_frames, 3),
                "visibility_threshold": threshold,
                "minimum_side_visibility_score": cfg.minimum_side_visibility_score,
                "minimum_usable_frame_ratio": cfg.minimum_usable_frame_ratio,
                "consumed_landmarks": list(SQUAT_LANDMARKS),
                **extraction_information,
            },
            "quality": {
                "analysis_valid": False,
                "video_valid": bool(total_frames and len(valid) / total_frames >= 0.75),
                "confidence": confidence,
                "tracking_valid_frame_ratio": (
                    round(len(valid) / total_frames, 3) if total_frames else 0.0
                ),
            },
            "reps_detected": 0,
            "rep": {},
            "reps": [],
            "minimum_knee_angle": None,
            "minimum_hip_angle": None,
            "movement_angle_summary": movement_angles,
            "maximum_torso_lean_degrees": round(max(max_t), 1) if max_t else None,
            "bottom_frame": None,
            "bottom_timestamp_seconds": None,
            "depth_assessment": "not_assessed",
            "confidence": confidence,
            "warnings": warnings,
            "segmentation_debug": segmentation_debug,
            "frames": [
                {
                    "frame": i,
                    "timestamp_seconds": round(i / fps, 3) if fps else None,
                    "phase": phases[i],
                    **metrics[i],
                }
                for i in range(total_frames)
            ],
        }
    timings, angles = _phase_summaries(phases, metrics, fps)
    movement_angles = _movement_angle_summary(phases, metrics)
    bottom_idx = [i for i, p in enumerate(phases) if p == "bottom"] or list(
        range(
            state.get("bottom_window_frames", [valid[0], valid[0]])[0],
            state.get("bottom_window_frames", [valid[0], valid[0]])[1] + 1,
        )
    )
    depth_selection = select_evidence_side(frames, ("hip", "knee"), threshold, bottom_idx)
    depth_side = depth_selection["selected_side"]
    depth_metrics = (
        [calculate_frame_metrics(frame, threshold, depth_side) for frame in frames]
        if depth_side else metrics
    )
    depth_source_frames = [
        i
        for i in bottom_idx
        if depth_metrics[i].get("hip_y") is not None
        and depth_metrics[i].get("knee_y") is not None
    ]
    hip_bottom = _median_metric(depth_metrics, depth_source_frames, "hip_y")
    knee_bottom = _median_metric(depth_metrics, depth_source_frames, "knee_y")
    if isinstance(hip_bottom, float) and isinstance(knee_bottom, float):
        delta = hip_bottom - knee_bottom
        depth = (
            "likely_sufficient_depth"
            if delta > cfg.depth_margin
            else (
                "borderline_depth"
                if delta >= -cfg.depth_margin
                else "likely_insufficient_depth"
            )
        )
    else:
        delta = None
        depth = "unknown"
    setup_k = angles["setup"]["knee_angle"]["median"]
    bottom_k = angles["bottom"]["knee_angle"]["median"]
    setup_h = angles["setup"]["hip_angle"]["median"]
    bottom_h = angles["bottom"]["hip_angle"]["median"]
    knee_change = (
        round(setup_k - bottom_k, 1)
        if setup_k is not None and bottom_k is not None
        else None
    )
    hip_change = (
        round(setup_h - bottom_h, 1)
        if setup_h is not None and bottom_h is not None
        else None
    )
    setup_t = angles["setup"]["torso_lean_degrees"]["median"]
    bottom_t = angles["bottom"]["torso_lean_degrees"]["median"]
    torso_change = (
        round(bottom_t - setup_t, 1)
        if setup_t is not None and bottom_t is not None
        else None
    )
    desc_d = timings["descent"]["duration_seconds"] or 0
    asc_d = timings["ascent"]["duration_seconds"] or 0
    ratio = round(asc_d / desc_d, 2) if desc_d else None
    bottom_duration = timings["bottom"]["duration_seconds"] or 0
    # sticking low-speed segment inside ascent
    vel = _velocity(hip_sm, fps)
    asc_idx = [i for i, p in enumerate(phases) if p == "ascent"]
    max_up = max(
        [abs(v) for i, v in enumerate(vel) if i in asc_idx and v is not None] or [0]
    )
    low_thr = max_up * cfg.sticking_low_speed_ratio
    low_runs = [
        r
        for r in _runs(
            [
                i in asc_idx and vel[i] is not None and abs(float(vel[i])) <= low_thr
                for i in range(total_frames)
            ]
        )
        if (r[1] - r[0] + 1) / fps >= cfg.sticking_min_duration_seconds
    ]
    sticking = {
        "detected": bool(low_runs),
        "duration_seconds": (
            round((low_runs[0][1] - low_runs[0][0] + 1) / fps, 3) if low_runs else 0.0
        ),
        "start_timestamp_seconds": round(low_runs[0][0] / fps, 3) if low_runs else None,
        "end_timestamp_seconds": round(low_runs[0][1] / fps, 3) if low_runs else None,
    }
    # hip first
    hip_first = {
        "detected": False,
        "duration_seconds": 0.0,
        "max_relative_progress_difference": 0.0,
        "source_frames": [],
        "hip_progress_by_frame": {},
        "shoulder_progress_by_frame": {},
    }
    early_ascent_evaluation_frames: list[int] = []
    progress_differences: list[float | None] = []
    if asc_idx:
        early_count = max(1, math.ceil(len(asc_idx) * cfg.hip_first_ascent_fraction))
        end = asc_idx[min(len(asc_idx) - 1, early_count - 1)]
        start = asc_idx[0]
        early_ascent_evaluation_frames = list(range(start, end + 1))
        coordination_selection = select_evidence_side(
            frames, ("hip", "shoulder"), threshold, early_ascent_evaluation_frames
        )
        coordination_side = coordination_selection["selected_side"]
        coordination_metrics = (
            [calculate_frame_metrics(frame, threshold, coordination_side) for frame in frames]
            if coordination_side else metrics
        )
        coordination_hip_sm = _median_smooth(
            [m["hip_y"] if isinstance(m["hip_y"], float) else None for m in coordination_metrics],
            cfg.smoothing_window,
        )
        hip0 = coordination_hip_sm[start]
        hip_end = coordination_hip_sm[asc_idx[-1]]
        sh_sm = _median_smooth(
            [
                m["shoulder_y"] if isinstance(m["shoulder_y"], float) else None
                for m in coordination_metrics
            ],
            cfg.smoothing_window,
        )
        sh0 = sh_sm[start]
        sh_end = sh_sm[asc_idx[-1]]
        raw_coordination_frames = {
            i
            for i, frame_metrics in enumerate(coordination_metrics)
            if isinstance(frame_metrics["hip_y"], float)
            and isinstance(frame_metrics["shoulder_y"], float)
        }
        raw_normalisation_available = {
            start,
            asc_idx[-1],
        }.issubset(raw_coordination_frames)
        hip_progress_by_frame = {}
        shoulder_progress_by_frame = {}
        coordination_source_frames = []
        for i in range(total_frames):
            difference: float | None = None
            if (
                start <= i <= end
                and raw_normalisation_available
                and i in raw_coordination_frames
                and None not in (
                    hip0,
                    hip_end,
                    coordination_hip_sm[i],
                    sh0,
                    sh_end,
                    sh_sm[i],
                )
            ):
                hp = (float(hip0) - float(coordination_hip_sm[i])) / max(
                    float(hip0) - float(hip_end), 1e-6
                )
                sp = (float(sh0) - float(sh_sm[i])) / max(
                    float(sh0) - float(sh_end), 1e-6
                )
                difference = hp - sp
                coordination_source_frames.append(i)
                hip_progress_by_frame[str(i)] = round(hp, 3)
                shoulder_progress_by_frame[str(i)] = round(sp, 3)
            progress_differences.append(difference)
        production_coordination = classify_early_ascent(
            progress_differences,
            fps,
            cfg.hip_first_progress_threshold,
            cfg.hip_first_min_duration_seconds,
        )
        hip_first = {
            "detected": production_coordination["detected"],
            "duration_seconds": production_coordination["qualifying_duration_seconds"],
            "max_relative_progress_difference": production_coordination[
                "maximum_progress_difference"
            ],
            "source_frames": coordination_source_frames,
            "hip_progress_by_frame": hip_progress_by_frame,
            "shoulder_progress_by_frame": shoulder_progress_by_frame,
        }
    depth_side = depth_side or selected_side
    coordination_side = locals().get("coordination_side") or selected_side
    side_hip = f"{depth_side}_hip"
    side_knee = f"{depth_side}_knee"
    coordination_hip = f"{coordination_side}_hip"
    side_shoulder = f"{coordination_side}_shoulder"
    depth_reliability = _metric_trajectory_reliability(
        frames, bottom_idx, (side_hip, side_knee), threshold
    )
    coordination_reliability = _metric_trajectory_reliability(
        frames,
        early_ascent_evaluation_frames,
        (coordination_hip, side_shoulder),
        threshold,
    )
    depth_spec = metric_specs["squat_depth_2d_proxy"]
    if delta is None or not depth_source_frames:
        depth_evidence = _insufficient_metric_evidence(
            depth_spec,
            depth_side,
            "Hip and knee did not coexist with sufficient quality in the detected bottom frames.",
            repetition_id=1,
            phase="bottom",
            source_frames=list(bottom_idx),
        )
        depth_evidence.landmark_quality = _metric_landmark_quality(
            depth_metrics, bottom_idx, (side_hip, side_knee)
        )
        depth_evidence.landmark_state = _landmark_states_from_quality(
            depth_evidence.landmark_quality
        )
        depth_reliability["existing_requirements_met"] = False
        depth_evidence.reliability = depth_reliability
    else:
        depth_classification = classify_squat_depth(delta, cfg.depth_margin)
        depth_evidence = MetricEvidence(
            metric_id=depth_spec.metric_id,
            repetition_id=1,
            phase="bottom",
            source_frames=depth_source_frames,
            landmarks_used=[side_hip, side_knee],
            landmark_quality=_metric_landmark_quality(
                depth_metrics, depth_source_frames, (side_hip, side_knee)
            ),
            landmark_state={
                side_hip: LandmarkState.OBSERVED,
                side_knee: LandmarkState.OBSERVED,
            },
            landmark_origin={side_hip: "MEDIAPIPE", side_knee: "MEDIAPIPE"},
            measurement={
                "bottom_hip_y_median": round(hip_bottom, 6),
                "bottom_knee_y_median": round(knee_bottom, 6),
                "hip_minus_knee_y": round(delta, 6),
                "legacy_depth_assessment": depth,
            },
            normalisation=depth_spec.normalisation_method,
            decision_rule=depth_spec.decision_rule,
            threshold={
                **depth_spec.threshold,
                **depth_spec.threshold_source,
                "threshold_version": depth_spec.threshold_version,
            },
            distance_from_boundary=round(
                min(abs(delta - cfg.depth_margin), abs(delta + cfg.depth_margin)), 6
            ),
            reliability=depth_reliability,
            sensitivity_result=_depth_sensitivity(delta, cfg),
            evidence_status=EvidenceStatus.SUFFICIENT,
            classification=depth_classification,
            warnings=[
                "This is a 2D side-view proxy, not a competition judging decision."
            ],
        )
        depth_evidence.reliability["existing_requirements_met"] = True
    coordination_spec = metric_specs["squat_early_ascent_coordination"]
    coordination_frames = hip_first.get("source_frames", [])
    if not coordination_frames:
        coordination_evidence = _insufficient_metric_evidence(
            coordination_spec,
            coordination_side,
            "Hip and shoulder evidence was insufficient to calculate normalised early-ascent coordination.",
            repetition_id=1,
            phase="ascent",
            source_frames=early_ascent_evaluation_frames,
        )
        coordination_evidence.landmark_quality = _metric_landmark_quality(
            coordination_metrics, early_ascent_evaluation_frames, (coordination_hip, side_shoulder)
        )
        coordination_evidence.landmark_state = _landmark_states_from_quality(
            coordination_evidence.landmark_quality
        )
        coordination_reliability["existing_requirements_met"] = False
        coordination_evidence.reliability = coordination_reliability
    else:
        max_difference = float(hip_first["max_relative_progress_difference"])
        coordination_evidence = MetricEvidence(
            metric_id=coordination_spec.metric_id,
            repetition_id=1,
            phase="ascent",
            source_frames=coordination_frames,
            landmarks_used=[coordination_hip, side_shoulder],
            landmark_quality=_metric_landmark_quality(
                coordination_metrics, coordination_frames, (coordination_hip, side_shoulder)
            ),
            landmark_state={
                coordination_hip: LandmarkState.OBSERVED,
                side_shoulder: LandmarkState.OBSERVED,
            },
            landmark_origin={coordination_hip: "MEDIAPIPE", side_shoulder: "MEDIAPIPE"},
            measurement={
                "ascent_frame_range": [asc_idx[0], asc_idx[-1]],
                "early_ascent_frame_range": [
                    coordination_frames[0],
                    coordination_frames[-1],
                ],
                "hip_progress_by_frame": hip_first["hip_progress_by_frame"],
                "shoulder_progress_by_frame": hip_first["shoulder_progress_by_frame"],
                "maximum_progress_difference": max_difference,
                "qualifying_duration_seconds": hip_first["duration_seconds"],
                "duration_requirement_met": hip_first["detected"],
            },
            normalisation=coordination_spec.normalisation_method,
            decision_rule=coordination_spec.decision_rule,
            threshold={
                **coordination_spec.threshold,
                **coordination_spec.threshold_source,
                "threshold_version": coordination_spec.threshold_version,
            },
            distance_from_boundary=round(
                max_difference - cfg.hip_first_progress_threshold, 6
            ),
            reliability=coordination_reliability,
            sensitivity_result=_early_ascent_sensitivity(
                progress_differences, fps, cfg
            ),
            evidence_status=EvidenceStatus.SUFFICIENT,
            classification=(
                EarlyAscentClassification.HIP_FIRST_PATTERN_DETECTED
                if hip_first["detected"]
                else EarlyAscentClassification.HIP_FIRST_PATTERN_NOT_DETECTED
            ),
            warnings=[],
        )
        coordination_evidence.reliability["existing_requirements_met"] = True
    metric_results = [depth_evidence.to_dict(), coordination_evidence.to_dict()]
    findings = []
    findings.append(
        _finding(
            6 if depth == "likely_insufficient_depth" else 12,
            "depth",
            depth,
            "medium",
            (
                f"Bottom median hip_y {hip_bottom:.3f} vs knee_y {knee_bottom:.3f}."
                if delta is not None
                else "Depth could not be measured."
            ),
            {
                "bottom_hip_y_median": hip_bottom,
                "bottom_knee_y_median": knee_bottom,
                "hip_minus_knee_y": delta,
            },
        )
    )
    if hip_first["detected"]:
        findings.append(
            _finding(
                8,
                "hip_first_ascent",
                "observed",
                "medium",
                "Hip upward progress exceeded shoulder progress early in ascent for a sustained segment.",
                hip_first,
            )
        )
    if (
        knee_change is not None
        and hip_change is not None
        and knee_change >= cfg.limited_hip_min_knee_flexion_change
        and hip_change / max(knee_change, 1e-6) <= cfg.limited_hip_max_hip_to_knee_ratio
    ):
        findings.append(
            _finding(
                10,
                "limited_hip_contribution",
                "observed",
                "low",
                "Knee flexion changed substantially more than hip flexion from setup to bottom.",
                {
                    "knee_flexion_change_degrees": knee_change,
                    "hip_flexion_change_degrees": hip_change,
                },
            )
        )
    if (ratio is not None and ratio >= cfg.prolonged_ascent_ratio) or sticking[
        "detected"
    ]:
        findings.append(
            _finding(
                11,
                "prolonged_ascent_or_sticking_region",
                "observed",
                "medium",
                "Ascent was longer than descent and/or contained a sustained low-speed segment.",
                {"ascent_to_descent_ratio": ratio, "sticking_region": sticking},
            )
        )
    if not state.get("final_standing_established"):
        findings.append(
            _finding(
                5,
                "rep_completion",
                "warning",
                "medium",
                "Final standing could not be established from the available frames.",
                {"final_standing_established": False},
            )
        )
    findings = sorted(findings, key=lambda f: f["priority"])
    derived = {
        "knee_flexion_change_degrees": knee_change,
        "hip_flexion_change_degrees": hip_change,
        "bottom_torso_lean_change_degrees": torso_change,
        "ascent_to_descent_ratio": ratio,
        "bottom_pause_duration_seconds": round(bottom_duration, 3),
        "bottom_pause_detected": bottom_duration > cfg.bottom_pause_threshold_seconds,
        "ascent_low_speed_region": sticking,
        "hip_first_ascent": hip_first,
    }
    confidence = (
        "high"
        if not warnings and missing_ratio <= 0.10 and full
        else "medium" if missing_ratio <= 0.25 and full else "low"
    )
    rep = {
        "rep_number": 1,
        "start_frame": timings["setup"]["start_frame"],
        "bottom_frame": state.get(
            "representative_bottom_frame", max(bottom_idx) if bottom_idx else None
        ),
        "end_frame": timings["final_standing"]["end_frame"],
        "bottom_window_frames": state.get("bottom_window_frames"),
        "representative_bottom_window_frames": state.get(
            "representative_bottom_window_frames"
        ),
        "depth_assessment": depth,
        "phase_timings": timings,
        "phase_angles": angles,
        "derived_metrics": derived,
        "metric_results": metric_results,
        "findings": findings,
        "warnings": warnings,
        "confidence": confidence,
        "segmentation_debug": segmentation_debug,
    }
    max_t = _finite_values(m["torso_lean_degrees"] for m in metrics)
    return {
        "lift": "squat",
        "analysis_status": "warning" if warnings else "valid",
        "selected_side": selected_side,
        "visibility": visibility,
        "recording_reliability": recording_reliability,
        "phase_frames": {
            phase: {
                "start_frame": timing["start_frame"],
                "end_frame": timing["end_frame"],
            }
            for phase, timing in timings.items()
        },
        "measurements": {
            "minimum_knee_angle": movement_angles["minimum_knee_angle"]["value_degrees"],
            "minimum_hip_angle": movement_angles["minimum_hip_angle"]["value_degrees"],
            "movement_angle_summary": movement_angles,
            "maximum_torso_lean_degrees": round(max(max_t), 1) if max_t else None,
            "depth_assessment": depth,
            **derived,
        },
        "metric_specifications": {
            key: spec.to_dict() for key, spec in metric_specs.items()
        },
        "metric_results": metric_results,
        "detected_faults": [
            finding
            for finding in findings
            if finding.get("status") in {"observed", "likely_insufficient_depth"}
        ],
        "kinematic_self_calibration": kinematic_self_calibration,
        "processing_information": {
            "total_frames": total_frames,
            "usable_frames": len(valid),
            "usable_frame_definition": "selected-side hip available for phase trajectory",
            "usable_frame_ratio": round(len(valid) / total_frames, 3),
            "visibility_threshold": threshold,
            "minimum_side_visibility_score": cfg.minimum_side_visibility_score,
            "minimum_usable_frame_ratio": cfg.minimum_usable_frame_ratio,
            "warning_usable_frame_ratio": cfg.warning_usable_frame_ratio,
            "consumed_landmarks": list(SQUAT_LANDMARKS),
            **extraction_information,
        },
        "quality": {
            "analysis_valid": bool(full),
            "video_valid": bool(total_frames and len(valid) / total_frames >= 0.75),
            "confidence": confidence,
            "tracking_valid_frame_ratio": (
                round(len(valid) / total_frames, 3) if total_frames else 0.0
            ),
        },
        "reps_detected": 1 if full else 0,
        "rep": rep,
        "reps": [rep] if full else [],
        "minimum_knee_angle": movement_angles["minimum_knee_angle"]["value_degrees"],
        "minimum_hip_angle": movement_angles["minimum_hip_angle"]["value_degrees"],
        "movement_angle_summary": movement_angles,
        "maximum_torso_lean_degrees": round(max(max_t), 1) if max_t else None,
        "bottom_frame": rep["bottom_frame"],
        "bottom_timestamp_seconds": (
            round((rep["bottom_frame"] or 0) / fps, 3)
            if fps and rep["bottom_frame"] is not None
            else None
        ),
        "depth_assessment": depth,
        "confidence": confidence,
        "warnings": warnings,
        "segmentation_debug": segmentation_debug,
        "frames": [
            {
                "frame": i,
                "timestamp_seconds": round(i / fps, 3) if fps else None,
                "phase": phases[i],
                **metrics[i],
            }
            for i in range(total_frames)
        ],
    }


def save_analysis_json(analysis: dict[str, Any], output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return str(output_path)
