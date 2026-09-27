"""Deterministic DEADLIFT-01 descriptive analyser using the shared lift context."""

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

DEADLIFT_LANDMARK_BASES = (
    "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "foot_index"
)
DEADLIFT_SIDE_SELECTION_BASES = ("shoulder", "hip", "knee", "ankle", "wrist")
DEADLIFT_LANDMARKS = tuple(
    f"{side}_{joint}" for side in ("left", "right") for joint in DEADLIFT_LANDMARK_BASES
)
DEADLIFT_OVERLAY_CONNECTION_BASES = (
    ("shoulder", "elbow"),
    ("elbow", "wrist"),
    ("shoulder", "hip"),
    ("hip", "knee"),
    ("knee", "ankle"),
    ("ankle", "heel"),
    ("heel", "foot_index"),
    ("ankle", "foot_index"),
)
DEADLIFT_OVERLAY_CONNECTIONS = tuple(
    (f"{side}_{start}", f"{side}_{end}")
    for side in ("left", "right")
    for start, end in DEADLIFT_OVERLAY_CONNECTION_BASES
)
DEADLIFT_CALIBRATION_SEGMENTS = {
    "shoulder_hip": ("shoulder", "hip"),
    "hip_knee": ("hip", "knee"),
    "knee_ankle": ("knee", "ankle"),
}
DEADLIFT_LANDMARK_CONFIGURATION = LiftLandmarkConfiguration(
    retained_landmarks=DEADLIFT_LANDMARKS,
    side_selection_bases=DEADLIFT_SIDE_SELECTION_BASES,
    calibration_segments=DEADLIFT_CALIBRATION_SEGMENTS,
)


@dataclass(frozen=True)
class DeadliftAnalysisConfig:
    """Engineering parameters for segmentation, not technique thresholds."""

    visibility_threshold: float = 0.6
    minimum_side_visibility_score: float = 0.6
    minimum_usable_frame_ratio: float = 0.75
    smoothing_window: int = 5
    setup_fraction: float = 0.2
    minimum_setup_frames: int = 4
    setup_stability_range: float = 0.018
    sustained_motion_frames: int = 3
    minimum_lift_off_travel: float = 0.025
    upward_step_tolerance: float = 0.002
    knee_passing_tolerance: float = 0.015
    minimum_total_wrist_travel: float = 0.10
    terminal_window_frames: int = 5
    terminal_stability_range: float = 0.018
    metric_window_radius: int = 2
    # Provisional project thresholds for the existing body-scale-normalised
    # hip-minus-shoulder progress metric. They are not literature cut-offs.
    early_coordination_threshold: float = 0.08
    early_coordination_uncertainty_band: float = 0.02
    early_coordination_minimum_persistent_frames: int = 3
    finish_knee_angle_threshold_degrees: float = 165.0
    finish_knee_angle_uncertainty_degrees: float = 5.0
    finish_trunk_thigh_angle_threshold_degrees: float = 160.0
    finish_trunk_thigh_uncertainty_degrees: float = 5.0
    finish_torso_inclination_threshold_degrees: float = 12.0
    finish_torso_inclination_uncertainty_degrees: float = 5.0
    finish_minimum_evidence_frames: int = 3


def assess_early_hip_shoulder_coordination(
    frame_differences: list[float] | None, config: DeadliftAnalysisConfig | None = None
) -> str:
    """Classify persistent hip-leading progress in the existing metric's units."""
    cfg = config or DeadliftAnalysisConfig()
    if not frame_differences or len(frame_differences) < cfg.early_coordination_minimum_persistent_frames:
        return "INSUFFICIENT_EVIDENCE"
    upper = cfg.early_coordination_threshold + cfg.early_coordination_uncertainty_band
    lower = cfg.early_coordination_threshold - cfg.early_coordination_uncertainty_band
    run = best_run = 0
    for value in frame_differences:
        run = run + 1 if value >= upper else 0
        best_run = max(best_run, run)
    if best_run >= cfg.early_coordination_minimum_persistent_frames:
        return "EARLY_HIP_RISE_DETECTED"
    if max(frame_differences) < lower:
        return "EARLY_HIP_RISE_NOT_DETECTED"
    return "UNCERTAIN"


def assess_finish_posture(
    measurement: dict[str, Any] | None,
    config: DeadliftAnalysisConfig | None = None,
    evidence_frame_count: int | None = None,
) -> str:
    """Classify a 2D finish-posture proxy; this is not lockout judging."""
    cfg = config or DeadliftAnalysisConfig()
    if evidence_frame_count is not None and evidence_frame_count < cfg.finish_minimum_evidence_frames:
        return "INSUFFICIENT_EVIDENCE"
    if not measurement or any(
        measurement.get(key) is None
        for key in (
            "projected_knee_angle_degrees",
            "projected_trunk_thigh_angle_degrees",
            "torso_inclination_from_vertical_degrees",
        )
    ):
        return "INSUFFICIENT_EVIDENCE"
    knee = float(measurement["projected_knee_angle_degrees"])
    hip = float(measurement["projected_trunk_thigh_angle_degrees"])
    torso = float(measurement["torso_inclination_from_vertical_degrees"])
    if (
        knee < cfg.finish_knee_angle_threshold_degrees - cfg.finish_knee_angle_uncertainty_degrees
        or hip < cfg.finish_trunk_thigh_angle_threshold_degrees - cfg.finish_trunk_thigh_uncertainty_degrees
        or torso > cfg.finish_torso_inclination_threshold_degrees + cfg.finish_torso_inclination_uncertainty_degrees
    ):
        return "INCOMPLETE_FINISH"
    if (
        knee >= cfg.finish_knee_angle_threshold_degrees + cfg.finish_knee_angle_uncertainty_degrees
        and hip >= cfg.finish_trunk_thigh_angle_threshold_degrees + cfg.finish_trunk_thigh_uncertainty_degrees
        and torso <= cfg.finish_torso_inclination_threshold_degrees - cfg.finish_torso_inclination_uncertainty_degrees
    ):
        return "COMPLETE_FINISH"
    return "UNCERTAIN"


def deadlift_metric_specifications(
    config: DeadliftAnalysisConfig | None = None,
) -> dict[str, MetricSpecification]:
    cfg = config or DeadliftAnalysisConfig()
    reliability = {
        "minimum_landmark_visibility": cfg.visibility_threshold,
        "missing_landmarks": "not_permitted_in_metric_source_frames",
    }
    none = {
        "threshold_source_type": "none_descriptive_metric",
        "threshold_source_reference": None,
    }
    supported = "conventional_deadlift_side_or_near-side_one_complete_repetition"
    return {
        "deadlift_start_geometry_2d": MetricSpecification(
            "deadlift_start_geometry_2d",
            "deadlift",
            ("shoulder", "hip", "knee", "ankle"),
            "lift_off",
            supported,
            "start_geometry_2d",
            "aspect-corrected projected image-plane angles",
            "descriptive start geometry; no technique classification",
            None,
            none,
            "deadlift-01",
            reliability,
            "insufficient unless the body chain is co-observed around lift-off",
            None,
            ("pose_estimation", "phase_state_machine", "robust_boundary_window"),
        ),
        "deadlift_phase_extension_2d": MetricSpecification(
            "deadlift_phase_extension_2d",
            "deadlift",
            ("shoulder", "hip", "knee", "ankle", "wrist"),
            "lift_off_to_knee_passing_to_top_position",
            supported,
            "phase_extension_changes_2d",
            "aspect-corrected projected image-plane angles",
            "descriptive boundary-window angle changes; no technique classification",
            None,
            none,
            "deadlift-01",
            reliability,
            "insufficient unless wrist-derived ordered phase boundaries and body-chain boundary windows exist",
            None,
            ("pose_estimation", "median_smoothed_wrist_proxy", "phase_state_machine"),
        ),
        "deadlift_finish_geometry_2d": MetricSpecification(
            "deadlift_finish_geometry_2d",
            "deadlift",
            ("shoulder", "hip", "knee", "ankle", "wrist"),
            "top_position",
            supported,
            "finish_geometry_2d",
            "aspect-corrected angles and normalised image y",
            "complete when all terminal geometry clears the provisional bands; incomplete when any measure clears its adverse band; otherwise uncertain; not competition lockout judging",
            {
                "knee_angle_degrees": cfg.finish_knee_angle_threshold_degrees,
                "trunk_thigh_angle_degrees": cfg.finish_trunk_thigh_angle_threshold_degrees,
                "torso_inclination_degrees": cfg.finish_torso_inclination_threshold_degrees,
                "uncertainty_degrees": {"knee": cfg.finish_knee_angle_uncertainty_degrees, "trunk_thigh": cfg.finish_trunk_thigh_uncertainty_degrees, "torso": cfg.finish_torso_inclination_uncertainty_degrees},
                "minimum_evidence_frames": cfg.finish_minimum_evidence_frames,
            },
            {"threshold_source_type": "provisional_project_threshold", "threshold_source_reference": None},
            "deadlift-01",
            reliability,
            "insufficient unless body chain and wrist are co-observed during the stable top position",
            None,
            ("pose_estimation", "median_smoothed_wrist_proxy", "phase_state_machine"),
        ),
        "deadlift_early_coordination_2d": MetricSpecification(
            "deadlift_early_coordination_2d",
            "deadlift",
            ("shoulder", "hip", "wrist"),
            "lift_off_to_knee_passing",
            supported,
            "hip_minus_shoulder_vertical_progress",
            "vertical displacement divided by per-video selected-side body scale",
            "hip-leading progress must exceed the upper uncertainty boundary for consecutive frames; values wholly below the lower boundary are negative; otherwise uncertain",
            {"hip_minus_shoulder_progress": cfg.early_coordination_threshold, "uncertainty_band": cfg.early_coordination_uncertainty_band, "minimum_persistent_frames": cfg.early_coordination_minimum_persistent_frames},
            {"threshold_source_type": "provisional_project_threshold", "threshold_source_reference": None},
            "deadlift-01",
            reliability,
            "insufficient unless phase-boundary wrist evidence, shoulder/hip trajectory, and body scale exist",
            None,
            ("pose_estimation", "phase_state_machine", "kinematic_self_calibration"),
        ),
    }


def _visible(frame: dict[str, dict[str, float]], name: str, threshold: float) -> bool:
    value = frame.get(name)
    return bool(
        value
        and float(value.get("visibility", 0)) >= threshold
        and float(value.get("presence", 1)) >= threshold
    )


def _smooth(values: list[float | None], window: int) -> list[float | None]:
    half = window // 2
    result = []
    for index in range(len(values)):
        observed = [
            v for v in values[max(0, index - half) : index + half + 1] if v is not None
        ]
        result.append(float(median(observed)) if observed else None)
    return result


def _detect_phases(
    wrist: list[float | None], knee: list[float | None], cfg: DeadliftAnalysisConfig
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    available = [i for i, value in enumerate(wrist) if value is not None]
    diagnostics: dict[str, Any] = {
        "proxy": "selected wrist / hand-and-bar-end proxy; not direct barbell tracking"
    }
    if len(available) < cfg.minimum_setup_frames + 2 * cfg.sustained_motion_frames:
        return {}, {**diagnostics, "failure_reason": "insufficient_wrist_trajectory"}
    # Locate a local, stable setup immediately before an ascent rather than using
    # the opening fraction of the recording.  The latter can contain the
    # lifter's approach and made "lift-off" mean "already near the knee".
    pull_candidates: list[tuple[int, int, int, float, float]] = []
    for lift_off in available:
        setup_end = lift_off - 1
        setup_start = setup_end - cfg.minimum_setup_frames + 1
        if setup_start < 0:
            continue
        setup_window = list(range(setup_start, setup_end + 1))
        if any(wrist[i] is None for i in setup_window):
            continue
        setup_values = [float(wrist[i]) for i in setup_window]
        stable_range = max(setup_values) - min(setup_values)
        stable_steps = all(
            abs(b - a) <= cfg.upward_step_tolerance
            for a, b in zip(setup_values, setup_values[1:])
        )
        if stable_range > cfg.setup_stability_range or not stable_steps:
            continue
        baseline = float(median(setup_values))
        if (
            wrist[lift_off] is None
            or baseline - float(wrist[lift_off]) <= cfg.upward_step_tolerance
        ):
            continue

        knee_candidates = [
            i
            for i in available
            if i > lift_off
            and knee[i] is not None
            and float(wrist[i]) <= float(knee[i]) + cfg.knee_passing_tolerance
        ]
        if not knee_candidates:
            continue
        knee_passing = knee_candidates[0]
        if knee_passing < lift_off + 2:
            continue
        ascent_frames = [i for i in available if lift_off <= i <= knee_passing]
        if len(ascent_frames) < cfg.sustained_motion_frames:
            continue
        sustained_upward = all(
            float(wrist[b]) <= float(wrist[a]) + cfg.upward_step_tolerance
            for a, b in zip(ascent_frames, ascent_frames[1:])
        )
        confirmed_travel = baseline - min(float(wrist[i]) for i in ascent_frames)
        if sustained_upward and confirmed_travel >= cfg.minimum_lift_off_travel:
            pull_candidates.append(
                (lift_off, knee_passing, setup_start, baseline, stable_range)
            )

    if not pull_candidates:
        return {}, {
            **diagnostics,
            "failure_reason": "lift_off_not_observed",
        }
    # The latest qualifying local setup avoids selecting an earlier approach
    # movement that happens to continue in the same direction as the pull.
    lift_off, knee_passing, setup_start, baseline, stable_range = pull_candidates[-1]
    # Resolve the whole post-knee turning region before accepting a top.  A
    # locally quiet window is not terminal evidence when the wrist subsequently
    # travels meaningfully higher (smaller image y), which is common while the
    # lifter slows through the last part of extension.
    post_knee = [i for i in available if i > knee_passing]
    if not post_knee:
        return {}, {
            **diagnostics,
            "failure_reason": "stable_top_position_not_observed",
            "lift_off_frame": lift_off,
            "knee_passing_frame": knee_passing,
        }
    trajectory_minimum = min(float(wrist[i]) for i in post_knee)
    minimum_frames = [
        i for i in post_knee if float(wrist[i]) == trajectory_minimum
    ]

    # Find the first movement away from the genuine minimum which remains
    # downward long enough to accumulate meaningful travel.  Looking forward
    # for confirmation lets gradual descents begin at their actual onset rather
    # than only when a short window has already covered the travel threshold.
    descent_start = None
    for start in range(minimum_frames[0] + 1, len(wrist)):
        if (
            wrist[start] is None
            or float(wrist[start])
            <= trajectory_minimum + cfg.upward_step_tolerance
        ):
            continue
        run = [float(wrist[start])]
        for index in range(start + 1, len(wrist)):
            if wrist[index] is None:
                break
            value = float(wrist[index])
            if value < run[-1] - cfg.upward_step_tolerance:
                break
            run.append(value)
            if (
                len(run) >= cfg.sustained_motion_frames
                and value - trajectory_minimum >= cfg.minimum_lift_off_travel
            ):
                descent_start = start
                break
        if descent_start is not None:
            break

    if descent_start is not None:
        top_confirmed = descent_start - 1
        top_start = max(
            knee_passing + 1,
            top_confirmed - cfg.terminal_window_frames + 1,
        )
    else:
        top_start = minimum_frames[0]
        top_confirmed = min(
            available[-1], top_start + cfg.terminal_window_frames - 1
        )
    top_window_indices = list(range(top_start, top_confirmed + 1))
    top_window = [
        float(wrist[i]) for i in top_window_indices if wrist[i] is not None
    ]
    top_is_stable = (
        len(top_window) >= cfg.sustained_motion_frames
        and max(top_window) - min(top_window) <= cfg.terminal_stability_range
        and max(top_window) <= trajectory_minimum + cfg.terminal_stability_range
        and baseline - trajectory_minimum >= cfg.minimum_total_wrist_travel
    )
    if not top_is_stable:
        return {}, {
            **diagnostics,
            "failure_reason": "stable_top_position_not_observed",
            "lift_off_frame": lift_off,
            "knee_passing_frame": knee_passing,
        }
    top_range = max(top_window) - min(top_window)
    if not (setup_start <= lift_off - 1 < lift_off + 1 < knee_passing < top_start):
        return {}, {**diagnostics, "failure_reason": "incoherent_phase_order"}

    # A top position is an ascent boundary, not repetition completion.  Require
    # sustained downward proxy travel before assigning a descent, then require a
    # separate stable window close to the local setup baseline before completion.
    return_start = return_end = None
    if descent_start is not None:
        for end in range(
            descent_start + cfg.sustained_motion_frames - 1, len(wrist)
        ):
            start = end - cfg.sustained_motion_frames + 1
            window = wrist[start : end + 1]
            if any(value is None for value in window):
                continue
            values = [float(value) for value in window if value is not None]
            if (
                max(values) - min(values) <= cfg.setup_stability_range
                and all(abs(value - baseline) <= cfg.setup_stability_range for value in values)
            ):
                return_start, return_end = start, end
                break

    top_end = (descent_start - 1) if descent_start is not None else available[-1]
    phases = {
        "setup": {"start_frame": setup_start, "end_frame": lift_off - 1},
        "lift_off": {"start_frame": lift_off, "end_frame": lift_off},
        "below_knee_ascent": {"start_frame": lift_off + 1, "end_frame": knee_passing - 1},
        "knee_passing_ascent": {"start_frame": knee_passing, "end_frame": top_start - 1},
        "top_position": {"start_frame": top_start, "end_frame": top_end},
    }
    if descent_start is not None:
        phases["controlled_descent"] = {
            "start_frame": descent_start,
            "end_frame": (return_start - 1) if return_start is not None else available[-1],
        }
    if return_start is not None and return_end is not None:
        phases["return_to_bottom"] = {
            "start_frame": return_start,
            "end_frame": available[-1],
        }
    return phases, {
        **diagnostics,
        "failure_reason": None,
        "setup_baseline_y": round(baseline, 6),
        "setup_range": round(stable_range, 6),
        "lift_off_evidence": {
            "frame": lift_off,
            "travel_from_setup": round(baseline - float(wrist[lift_off]), 6),
            "sustained_frames": cfg.sustained_motion_frames,
        },
        "knee_passing_evidence": {
            "frame": knee_passing,
            "wrist_y": wrist[knee_passing],
            "knee_y": knee[knee_passing],
            "tolerance": cfg.knee_passing_tolerance,
        },
        "top_position_evidence": {
            "start_frame": top_start,
            "confirmed_frame": top_confirmed,
            "trajectory_minimum_y": round(trajectory_minimum, 6),
            "wrist_proxy_values": [round(value, 6) for value in top_window],
            "total_upward_travel": round(baseline - trajectory_minimum, 6),
            "terminal_range": round(top_range, 6),
            "terminal_window_frames": cfg.terminal_window_frames,
        },
        "descent_evidence": {
            "start_frame": descent_start,
            "sustained_frames": cfg.sustained_motion_frames,
        },
        "return_evidence": {
            "start_frame": return_start,
            "confirmed_frame": return_end,
            "baseline_tolerance": cfg.setup_stability_range,
            "repetition_complete": return_end is not None,
        },
    }


def _angle(
    a: dict[str, float], b: dict[str, float], c: dict[str, float], aspect: float
) -> float:
    """Projected angle with normalised x scaled to the source frame aspect ratio."""
    ab = ((float(a["x"]) - float(b["x"])) * aspect, float(a["y"]) - float(b["y"]))
    cb = ((float(c["x"]) - float(b["x"])) * aspect, float(c["y"]) - float(b["y"]))
    lengths = math.hypot(*ab), math.hypot(*cb)
    if min(lengths) == 0:
        return float("nan")
    cosine = max(
        -1.0, min(1.0, (ab[0] * cb[0] + ab[1] * cb[1]) / (lengths[0] * lengths[1]))
    )
    return math.degrees(math.acos(cosine))


def _inclination(
    shoulder: dict[str, float], hip: dict[str, float], aspect: float
) -> float:
    dx = (float(shoulder["x"]) - float(hip["x"])) * aspect
    dy = float(shoulder["y"]) - float(hip["y"])
    return math.degrees(math.atan2(abs(dx), abs(dy)))  # zero is image vertical


def _window(center: int, radius: int, length: int) -> list[int]:
    return list(range(max(0, center - radius), min(length, center + radius + 1)))


def _quality(
    frames: list[dict[str, dict[str, float]]], names: list[str], source: list[int]
) -> tuple[dict[str, Any], dict[str, LandmarkState], dict[str, str]]:
    quality, states, origins = {}, {}, {}
    for name in names:
        values = [frames[i][name] for i in source if name in frames[i]]
        quality[name] = {
            "source_frame_coverage": round(len(values) / len(source), 3)
            if source
            else 0.0,
            "median_visibility": round(
                float(median(float(v.get("visibility", 0)) for v in values)), 3
            )
            if values
            else None,
        }
        states[name] = (
            LandmarkState.OBSERVED
            if source and len(values) == len(source)
            else LandmarkState.UNAVAILABLE
        )
        origins[name] = "mediapipe_pose_observation" if values else "unavailable"
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
        {n: LandmarkState.UNAVAILABLE for n in names},
        {n: "unavailable" for n in names},
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


def _evidence(
    spec: MetricSpecification,
    frames: list[dict[str, dict[str, float]]],
    names: list[str],
    source: list[int],
    measurement: dict[str, Any],
    classification: str | None = None,
) -> dict[str, Any]:
    quality, states, origins = _quality(frames, names, source)
    evidence_status = (
        EvidenceStatus.INSUFFICIENT
        if classification == "INSUFFICIENT_EVIDENCE"
        else EvidenceStatus.SUFFICIENT
    )
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
        {"sufficient": evidence_status == EvidenceStatus.SUFFICIENT, "metric_specific": True},
        None,
        evidence_status,
        classification,
        [],
    ).to_dict()


def analyse_deadlift_landmarks(
    frames: list[dict[str, dict[str, float]]],
    fps: float = 30.0,
    processing_metadata: dict[str, Any] | None = None,
    config: DeadliftAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Analyse one conventional deadlift repetition with provisional coordination and finish classifications."""
    cfg = config or DeadliftAnalysisConfig()
    specs = deadlift_metric_specifications(cfg)
    context = prepare_lift_analysis_context(
        lift="deadlift",
        frames=frames,
        fps=fps,
        visibility_threshold=cfg.visibility_threshold,
        landmarks=DEADLIFT_LANDMARK_CONFIGURATION,
        processing_metadata=processing_metadata,
        minimum_side_visibility_score=cfg.minimum_side_visibility_score,
        minimum_usable_frame_ratio=cfg.minimum_usable_frame_ratio,
        phase_tracking_bases=("wrist", "knee"),
    )
    side = context.selected_side
    phase_candidates = []
    for candidate_side in ("left", "right"):
        wrist_name, knee_name = f"{candidate_side}_wrist", f"{candidate_side}_knee"
        candidate_wrist = [
            float(frame[wrist_name]["y"])
            if _visible(frame, wrist_name, cfg.visibility_threshold) else None
            for frame in context.frames
        ]
        candidate_knee = [
            float(frame[knee_name]["y"])
            if _visible(frame, knee_name, cfg.visibility_threshold) else None
            for frame in context.frames
        ]
        candidate_smooth = _smooth(candidate_wrist, cfg.smoothing_window)
        candidate_phases, candidate_diagnostics = _detect_phases(candidate_smooth, candidate_knee, cfg)
        complete_count = sum(
            wrist is not None and knee is not None
            for wrist, knee in zip(candidate_wrist, candidate_knee)
        )
        phase_candidates.append((
            bool(candidate_phases), complete_count, candidate_side, candidate_wrist,
            candidate_knee, candidate_smooth, candidate_phases, candidate_diagnostics,
        ))
    phase_candidates.sort(key=lambda item: (item[0], item[1], item[2] == "left"), reverse=True)
    best_phase = phase_candidates[0]
    if best_phase[1]:
        side = best_phase[2]
        context.visibility["selected_side"] = side
        context.recording_reliability["selected_side"] = side
        context.recording_reliability["selected_side_usable_frames"] = best_phase[1]
        context.recording_reliability["selected_side_usable_frame_ratio"] = round(best_phase[1] / len(context.frames), 3) if context.frames else 0.0
        context.recording_reliability["rejection_reasons"] = []
    names = {base: f"{side}_{base}" if side else "" for base in DEADLIFT_LANDMARK_BASES}
    raw_wrist = [
        float(f[names["wrist"]]["y"])
        if names["wrist"] and _visible(f, names["wrist"], cfg.visibility_threshold)
        else None
        for f in context.frames
    ]
    raw_knee = [
        float(f[names["knee"]]["y"])
        if names["knee"] and _visible(f, names["knee"], cfg.visibility_threshold)
        else None
        for f in context.frames
    ]
    smooth_wrist = best_phase[5] if side else _smooth(raw_wrist, cfg.smoothing_window)
    phases, phase_diagnostics = (best_phase[6], best_phase[7]) if side else _detect_phases(smooth_wrist, raw_knee, cfg)
    recording_reliability = context.recording_reliability
    if recording_reliability["rejection_reasons"]:
        phases = {}
        phase_diagnostics = {
            **phase_diagnostics,
            "failure_reason": recording_reliability["rejection_reasons"][0],
        }
    metadata = processing_metadata or {}
    width, height = metadata.get("frame_width"), metadata.get("frame_height")
    aspect = float(width) / float(height) if width and height else 1.0
    warnings: list[str] = []
    if not width or not height:
        warnings.append(
            "Source frame aspect ratio was unavailable; projected angles use normalised coordinates and may be distorted for non-square video."
        )
    results: list[dict[str, Any]] = []
    if not phases:
        warning = "A coherent SETUP–LIFT_OFF–KNEE_PASSING–TOP_POSITION wrist-proxy sequence was not identified."
        warnings.append(warning)
        results = [_insufficient(spec, side, warning) for spec in specs.values()]
        if not recording_reliability["rejection_reasons"]:
            recording_reliability = {
                **recording_reliability,
                "rejection_reasons": ["incomplete_repetition"],
            }
    else:
        boundaries = {
            "lift_off": phases["lift_off"]["start_frame"],
            "knee_passing": phases["knee_passing_ascent"]["start_frame"],
            "top_position": phases["top_position"]["start_frame"],
        }
        def geometry(
            center: int,
            require_wrist: bool = False,
            metric_side: str | None = None,
            source_frames: list[int] | None = None,
        ) -> tuple[list[int], dict[str, Any] | None]:
            metric_names = {
                base: f"{metric_side or side}_{base}" for base in DEADLIFT_LANDMARK_BASES
            }
            required = [metric_names[b] for b in ("shoulder", "hip", "knee", "ankle")]
            required += [metric_names["wrist"]] if require_wrist else []
            source = [
                i
                for i in (
                    source_frames
                    if source_frames is not None
                    else _window(center, cfg.metric_window_radius, len(context.frames))
                )
                if all(
                    _visible(context.frames[i], n, cfg.visibility_threshold)
                    for n in required
                )
            ]
            if not source:
                return [], None
            knee_angles = [
                _angle(
                    context.frames[i][metric_names["hip"]],
                    context.frames[i][metric_names["knee"]],
                    context.frames[i][metric_names["ankle"]],
                    aspect,
                )
                for i in source
            ]
            trunk_angles = [
                _angle(
                    context.frames[i][metric_names["shoulder"]],
                    context.frames[i][metric_names["hip"]],
                    context.frames[i][metric_names["knee"]],
                    aspect,
                )
                for i in source
            ]
            inclinations = [
                _inclination(
                    context.frames[i][metric_names["shoulder"]],
                    context.frames[i][metric_names["hip"]],
                    aspect,
                )
                for i in source
            ]
            return source, {
                "projected_knee_angle_degrees": round(float(median(knee_angles)), 2),
                "projected_trunk_thigh_angle_degrees": round(
                    float(median(trunk_angles)), 2
                ),
                "torso_inclination_from_vertical_degrees": round(
                    float(median(inclinations)), 2
                ),
                "aspect_ratio_correction": round(aspect, 6),
                "angle_window_min_max": {
                    "knee": [round(min(knee_angles), 2), round(max(knee_angles), 2)],
                    "trunk_thigh": [
                        round(min(trunk_angles), 2),
                        round(max(trunk_angles), 2),
                    ],
                },
            }

        start_window = _window(boundaries["lift_off"], cfg.metric_window_radius, len(context.frames))
        start_side = select_evidence_side(context.frames, ("shoulder", "hip", "knee", "ankle"), cfg.visibility_threshold, start_window)["selected_side"] or side
        start_names = [f"{start_side}_{base}" for base in ("shoulder", "hip", "knee", "ankle")]
        start_source, start = geometry(boundaries["lift_off"], metric_side=start_side)
        spec = specs["deadlift_start_geometry_2d"]
        results.append(
            _evidence(spec, context.frames, start_names, start_source, start)
            if start
            else _insufficient(
                spec, start_side, "Body-chain geometry is unavailable around lift-off."
            )
        )
        boundary_source = sorted(set(sum((_window(v, cfg.metric_window_radius, len(context.frames)) for v in boundaries.values()), [])))
        extension_side = select_evidence_side(context.frames, ("shoulder", "hip", "knee", "ankle", "wrist"), cfg.visibility_threshold, boundary_source)["selected_side"] or side
        extension_names = [f"{extension_side}_{base}" for base in ("shoulder", "hip", "knee", "ankle", "wrist")]
        boundary_geometry = {key: geometry(frame, metric_side=extension_side)[1] for key, frame in boundaries.items()}
        spec = specs["deadlift_phase_extension_2d"]
        if all(boundary_geometry.values()):
            a, b, c = (
                boundary_geometry[k]
                for k in ("lift_off", "knee_passing", "top_position")
            )
            source = sorted(
                set(
                    sum(
                        (
                            _window(v, cfg.metric_window_radius, len(context.frames))
                            for v in boundaries.values()
                        ),
                        [],
                    )
                )
            )
            measurement = {
                "knee_extension_below_knee_degrees": round(
                    b["projected_knee_angle_degrees"]
                    - a["projected_knee_angle_degrees"],
                    2,
                ),
                "knee_extension_above_knee_degrees": round(
                    c["projected_knee_angle_degrees"]
                    - b["projected_knee_angle_degrees"],
                    2,
                ),
                "trunk_thigh_change_below_knee_degrees": round(
                    b["projected_trunk_thigh_angle_degrees"]
                    - a["projected_trunk_thigh_angle_degrees"],
                    2,
                ),
                "trunk_thigh_change_above_knee_degrees": round(
                    c["projected_trunk_thigh_angle_degrees"]
                    - b["projected_trunk_thigh_angle_degrees"],
                    2,
                ),
                "boundary_geometry": boundary_geometry,
            }
            results.append(
                _evidence(
                    spec,
                    context.frames,
                    extension_names,
                    source,
                    measurement,
                )
            )
        else:
            results.append(
                _insufficient(
                    spec,
                    extension_side,
                    "Body-chain geometry is unavailable at one or more wrist-derived phase boundaries.",
                )
            )
        top_evidence = phase_diagnostics["top_position_evidence"]
        finish_window = list(
            range(top_evidence["start_frame"], top_evidence["confirmed_frame"] + 1)
        )
        finish_side = select_evidence_side(context.frames, ("shoulder", "hip", "knee", "ankle", "wrist"), cfg.visibility_threshold, finish_window)["selected_side"] or side
        finish_names = [f"{finish_side}_{base}" for base in ("shoulder", "hip", "knee", "ankle", "wrist")]
        finish_source, finish = geometry(
            boundaries["top_position"], True, finish_side, finish_window
        )
        spec = specs["deadlift_finish_geometry_2d"]
        if finish:
            terminal_values = [
                smooth_wrist[i] for i in finish_source if smooth_wrist[i] is not None
            ]
            finish.update(
                {
                    "terminal_wrist_vertical_range": round(
                        max(terminal_values) - min(terminal_values), 6
                    ),
                    "top_position_proxy_only": True,
                }
            )
            finish_classification = assess_finish_posture(finish, cfg, len(finish_source))
            results.append(
                _evidence(
                    spec,
                    context.frames,
                    finish_names,
                    finish_source,
                    finish,
                    finish_classification,
                )
            )
        else:
            results.append(
                _insufficient(
                    spec,
                    finish_side,
                    "Stable top-position body-chain or wrist evidence is unavailable.",
                )
            )
        spec = specs["deadlift_early_coordination_2d"]
        body_scale = context.calibration_profile.get("profile", {}).get("body_scale")
        coordination_source = [
            i
            for i in range(boundaries["lift_off"], boundaries["knee_passing"] + 1)
            if _visible(context.frames[i], names["hip"], cfg.visibility_threshold)
            and _visible(context.frames[i], names["shoulder"], cfg.visibility_threshold)
        ]
        if body_scale and len(coordination_source) >= 2:
            first = coordination_source[0]
            diffs, inclinations = [], []
            for i in coordination_source:
                hip_progress = (
                    float(context.frames[first][names["hip"]]["y"])
                    - float(context.frames[i][names["hip"]]["y"])
                ) / body_scale
                shoulder_progress = (
                    float(context.frames[first][names["shoulder"]]["y"])
                    - float(context.frames[i][names["shoulder"]]["y"])
                ) / body_scale
                diffs.append(hip_progress - shoulder_progress)
                inclinations.append(
                    _inclination(
                        context.frames[i][names["shoulder"]],
                        context.frames[i][names["hip"]],
                        aspect,
                    )
                )
            measurement = {
                "maximum_hip_minus_shoulder_vertical_progress": round(max(diffs), 4),
                "median_hip_minus_shoulder_vertical_progress": round(
                    float(median(diffs)), 4
                ),
                "endpoint_hip_minus_shoulder_vertical_progress": round(diffs[-1], 4),
                "torso_inclination_change_degrees": round(
                    inclinations[-1] - inclinations[0], 2
                ),
                "body_scale": body_scale,
                "experimental_descriptive": True,
                "framewise_hip_minus_shoulder_vertical_progress": [round(v, 4) for v in diffs],
            }
            coordination_classification = assess_early_hip_shoulder_coordination(diffs, cfg)
            results.append(
                _evidence(
                    spec,
                    context.frames,
                    [names["shoulder"], names["hip"], names["wrist"]],
                    coordination_source,
                    measurement,
                    coordination_classification,
                )
            )
        else:
            results.append(
                _insufficient(
                    spec,
                    side,
                    "Early coordination requires shoulder/hip observations and a non-null body scale.",
                )
            )
    sufficient = sum(
        r["evidence_status"] == EvidenceStatus.SUFFICIENT.value for r in results
    )
    repetition_complete = "return_to_bottom" in phases
    if phases and not repetition_complete:
        warnings.append(
            "The upright position was observed, but the return to the setup baseline was not fully captured; the repetition is partial."
        )
    status = (
        "invalid"
        if not phases
        else (
            "warning"
            if sufficient < len(results) or not repetition_complete
            else "valid"
        )
    )
    faults = []
    classifications = {r["metric_id"]: r.get("classification") for r in results}
    if classifications.get("deadlift_early_coordination_2d") == "EARLY_HIP_RISE_DETECTED":
        faults.append("early_hip_rise")
    if classifications.get("deadlift_finish_geometry_2d") == "INCOMPLETE_FINISH":
        faults.append("incomplete_finish")
    calibration = attach_calibration_usage(
        context.calibration_profile,
        body_scale_consumers=[
            "deadlift_early_coordination_2d measurement",
            "deadlift_early_coordination_2d classification",
            "deadlift_early_coordination_2d coaching",
        ],
        body_scale_classification_input=True,
        body_scale_coaching_input=True,
    )
    return {
        "lift": "deadlift",
        "analysis_status": status,
        "selected_side": side,
        "visibility": context.visibility,
        "recording_reliability": recording_reliability,
        "phase_frames": phases,
        "measurements": {r["metric_id"]: r["measurement"] for r in results},
        "metric_specifications": {k: v.to_dict() for k, v in specs.items()},
        "metric_results": results,
        "detected_faults": faults,
        "calibration": calibration,
        "kinematic_self_calibration": calibration,
        "warnings": warnings,
        "processing_information": {
            "total_frames": len(context.frames),
            "usable_frames": recording_reliability["selected_side_usable_frames"],
            "usable_frame_ratio": recording_reliability["selected_side_usable_frame_ratio"],
            "fps": fps,
            "visibility_threshold": cfg.visibility_threshold,
            "consumed_landmarks": list(DEADLIFT_LANDMARKS),
            **context.extraction_information,
        },
        "diagnostics": {
            "scope": "one conventional deadlift; side or near-side; one complete repetition",
            "phase_signal": "selected-side wrist normalised image y (hand/bar-end proxy)",
            "raw_wrist_y": raw_wrist,
            "smoothed_wrist_y": smooth_wrist,
            "raw_knee_y": raw_knee,
            "phase_detection": phase_diagnostics,
            "repetition_complete": repetition_complete,
            "operational_parameters": cfg.__dict__,
            "angle_geometry": "x scaled by source width/height before projected angle calculations"
            if width and height
            else "normalised-coordinate fallback; aspect-ratio limitation warned",
        },
    }


def deadlift_diagnostics_view(analysis: dict[str, Any]) -> dict[str, Any]:
    """Project the complete deterministic result into a manual-inspection view."""
    evidence = {
        r.get("metric_id"): {
            k: r.get(k)
            for k in (
                "evidence_status",
                "reliability",
                "source_frames",
                "landmark_quality",
                "measurement",
                "landmark_state",
                "landmark_origin",
                "warnings",
            )
        }
        for r in analysis.get("metric_results", [])
    }
    calibration = analysis.get("calibration", {})
    return {
        "selected_side": analysis.get("selected_side"),
        "visibility": analysis.get("visibility"),
        "phase_frames": analysis.get("phase_frames"),
        "trajectory": {
            "signal": analysis.get("diagnostics", {}).get("phase_signal"),
            "raw": analysis.get("diagnostics", {}).get("raw_wrist_y"),
            "smoothed": analysis.get("diagnostics", {}).get("smoothed_wrist_y"),
            "knee_y": analysis.get("diagnostics", {}).get("raw_knee_y"),
            "phase_detection": analysis.get("diagnostics", {}).get("phase_detection"),
        },
        "metric_evidence": evidence,
        "calibration_summary": {
            "diagnostic_only": calibration.get("diagnostic_only"),
            "usage": calibration.get("usage", {}),
            "profile": calibration.get("profile"),
            "reacquisition_events": calibration.get("reacquisition_events", []),
        },
        "warnings": analysis.get("warnings", []),
        "provenance": {
            "wrist_proxy": "hand/bar-end proxy; not direct barbell tracking",
            "classifications": "deterministic provisional early-coordination and finish-posture proxy assessments",
        },
    }
