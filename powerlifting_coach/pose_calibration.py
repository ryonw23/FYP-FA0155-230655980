"""Per-video calibration of raw MediaPipe pose trajectories.

The distributions in this module describe what is typical for one subject in
one recording.  They are not anatomical models, correctness decisions, or a
source of replacement landmarks. Lift analysers declare how they consume the
shared body scale; deviation and reacquisition evidence remains diagnostic-only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Any


PROFILE_LANDMARKS = ("shoulder", "hip", "knee", "ankle")
PROFILE_SEGMENTS = {
    "shoulder_hip": ("shoulder", "hip"),
    "hip_knee": ("hip", "knee"),
    "knee_ankle": ("knee", "ankle"),
}
CalibrationSegments = dict[str, tuple[str, str]]
MAD_DIVISOR_EPSILON = 1e-12
POST_REACQUISITION_FRAME_STEPS = 3


def attach_calibration_usage(
    calibration: dict[str, Any],
    *,
    body_scale_consumers: list[str],
    body_scale_classification_input: bool,
    body_scale_coaching_input: bool,
) -> dict[str, Any]:
    """Copy a calibration result and declare its downstream lift-specific use."""

    result = deepcopy(calibration)
    body_scale_diagnostic_only = not body_scale_consumers
    result["usage"] = {
        "body_scale": {
            "diagnostic_only": body_scale_diagnostic_only,
            "consumers": list(body_scale_consumers),
            "classification_input": body_scale_classification_input,
            "coaching_input": body_scale_coaching_input,
        },
        "segment_deviations": {
            "diagnostic_only": True,
            "repairs_or_filters_landmarks": False,
        },
        "reacquisition": {
            "diagnostic_only": True,
            "repairs_or_filters_landmarks": False,
        },
    }
    result["diagnostic_only"] = body_scale_diagnostic_only
    return result


@dataclass(frozen=True)
class CalibrationDistribution:
    sample_count: int
    median: float | None
    mad: float | None
    p05: float | None
    p25: float | None
    p75: float | None
    p95: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class KinematicProfile:
    source_frame_count: int
    calibration_frame_count: int
    calibration_frames_used: list[int]
    frames_excluded_due_to_existing_quality_requirements: list[int]
    selected_side: str | None
    body_scale_method: str
    body_scale_segments: list[str]
    body_scale: float | None
    segment_profiles: dict[str, CalibrationDistribution]
    motion_profiles: dict[str, CalibrationDistribution]
    segment_change_profiles: dict[str, CalibrationDistribution]
    warnings: list[str]
    provenance: dict[str, str]


@dataclass(frozen=True)
class LandmarkReacquisitionEvent:
    """A MediaPipe observation that follows one or more unavailable frames."""

    landmark: str
    last_valid_frame: int
    reacquired_frame: int
    gap_frame_count: int
    unavailable_frames: list[int]
    previous_position: tuple[float, float]
    reacquired_position: tuple[float, float]
    origin: str
    observation_context: str
    evidence: dict[str, Any]


def _percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: list[float]) -> CalibrationDistribution:
    if not values:
        return CalibrationDistribution(0, None, None, None, None, None, None, None, None)
    centre = float(median(values))
    return CalibrationDistribution(
        sample_count=len(values),
        median=centre,
        mad=float(median(abs(value - centre) for value in values)),
        p05=_percentile(values, 0.05),
        p25=_percentile(values, 0.25),
        p75=_percentile(values, 0.75),
        p95=_percentile(values, 0.95),
        minimum=min(values),
        maximum=max(values),
    )


def _deviation(value: float, profile: CalibrationDistribution) -> dict[str, float | str | None]:
    deviation = None
    limitation = None
    if profile.median is None or profile.mad is None:
        limitation = "profile_has_no_samples"
    elif profile.mad <= MAD_DIVISOR_EPSILON:
        limitation = "profile_mad_zero_or_too_small"
    else:
        deviation = abs(value - profile.median) / profile.mad
    return {
        "value": value,
        "profile_median": profile.median,
        "profile_mad": profile.mad,
        "robust_deviation": deviation,
        "deviation_limitation": limitation,
    }


def build_kinematic_self_calibration(
    frames: list[dict[str, dict[str, float]]],
    selected_side: str | None,
    visibility_threshold: float,
    segments: CalibrationSegments | None = None,
) -> dict[str, Any]:
    """Learn and evaluate a descriptive profile from raw MediaPipe x/y values.

    The existing visibility/presence rule determines availability. No
    observation is removed because of a value learned by this function.
    """

    configured_segments = dict(segments or PROFILE_SEGMENTS)
    profile_landmarks = tuple(
        dict.fromkeys(joint for endpoints in configured_segments.values() for joint in endpoints)
    )
    warnings: list[str] = []
    if selected_side not in {"left", "right"}:
        warnings.append("No analysis side was selected; calibration distributions are unavailable.")
    names = {
        base: f"{selected_side}_{base}" if selected_side in {"left", "right"} else ""
        for base in profile_landmarks
    }

    def available(frame_index: int, base: str) -> bool:
        landmark = frames[frame_index].get(names[base]) if names[base] else None
        return bool(
            landmark
            and all(isinstance(landmark.get(axis), (int, float)) and math.isfinite(float(landmark[axis])) for axis in ("x", "y"))
            and float(landmark.get("visibility", 0.0)) >= visibility_threshold
            and float(landmark.get("presence", 1.0)) >= visibility_threshold
        )

    used = [i for i in range(len(frames)) if any(available(i, base) for base in profile_landmarks)]
    excluded = [i for i in range(len(frames)) if i not in set(used)]
    segment_values: dict[str, dict[int, float]] = {key: {} for key in configured_segments}
    for key, (first, second) in configured_segments.items():
        for i in range(len(frames)):
            if available(i, first) and available(i, second):
                a, b = frames[i][names[first]], frames[i][names[second]]
                segment_values[key][i] = math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

    complete_scales = [
        sum(segment_values[key][i] for key in configured_segments)
        for i in range(len(frames))
        if all(i in segment_values[key] for key in configured_segments)
    ]
    body_scale = float(median(complete_scales)) if complete_scales else None
    if body_scale is None or body_scale <= 0:
        body_scale = None
        if configured_segments == PROFILE_SEGMENTS:
            warnings.append("No positive linked shoulder-to-ankle segment scale was available; normalised motion profiles are unavailable.")
        else:
            warnings.append("No positive configured linked-segment scale was available; normalised motion profiles are unavailable.")

    motions: dict[str, dict[int, float]] = {base: {} for base in profile_landmarks}
    if body_scale:
        for base in profile_landmarks:
            for i in range(1, len(frames)):
                if available(i - 1, base) and available(i, base):
                    previous, current = frames[i - 1][names[base]], frames[i][names[base]]
                    motions[base][i] = math.hypot(float(current["x"]) - float(previous["x"]), float(current["y"]) - float(previous["y"])) / body_scale

    changes: dict[str, dict[int, float]] = {key: {} for key in configured_segments}
    for key, values in segment_values.items():
        for i, value in values.items():
            if i - 1 in values:
                changes[key][i] = abs(value - values[i - 1])

    segment_profiles = {key: _distribution(list(values.values())) for key, values in segment_values.items()}
    motion_profiles = {key: _distribution(list(values.values())) for key, values in motions.items()}
    change_profiles = {key: _distribution(list(values.values())) for key, values in changes.items()}
    if used and len(used) < len(frames):
        warnings.append("Some frames were excluded only because none of the selected-side calibration landmarks met the existing quality requirements.")

    observations = []
    adjacent = {
        base: tuple(key for key, endpoints in configured_segments.items() if base in endpoints)
        for base in profile_landmarks
    }
    reacquired_keys: set[tuple[str, int]] = set()
    reacquisition_events: list[LandmarkReacquisitionEvent] = []
    for base in profile_landmarks:
        last_valid: int | None = None
        unavailable_since_valid: list[int] = []
        for i in range(len(frames)):
            if not available(i, base):
                if last_valid is not None:
                    unavailable_since_valid.append(i)
                continue
            if last_valid is not None and unavailable_since_valid:
                previous = frames[last_valid][names[base]]
                current = frames[i][names[base]]
                raw_displacement = math.hypot(
                    float(current["x"]) - float(previous["x"]),
                    float(current["y"]) - float(previous["y"]),
                )
                elapsed_steps = i - last_valid
                normalized_total = raw_displacement / body_scale if body_scale else None
                normalized_per_step = normalized_total / elapsed_steps if normalized_total is not None else None
                motion_evidence: dict[str, Any] = {
                    "raw_displacement": raw_displacement,
                    "normalized_total_displacement": normalized_total,
                    "elapsed_frame_steps": elapsed_steps,
                    "normalized_displacement_per_elapsed_frame": normalized_per_step,
                }
                if normalized_per_step is not None:
                    motion_evidence.update(_deviation(normalized_per_step, motion_profiles[base]))
                    motion_evidence["profile_p95"] = motion_profiles[base].p95
                else:
                    motion_evidence.update({
                        "profile_median": motion_profiles[base].median,
                        "profile_mad": motion_profiles[base].mad,
                        "profile_p95": motion_profiles[base].p95,
                        "robust_deviation": None,
                        "deviation_limitation": "body_scale_unavailable",
                    })

                segment_consistency = {}
                for key in adjacent[base]:
                    if i in segment_values[key]:
                        segment_consistency[key] = {
                            **_deviation(segment_values[key][i], segment_profiles[key]),
                            "observed_segment_length": segment_values[key][i],
                            "profile_p05": segment_profiles[key].p05,
                            "profile_p95": segment_profiles[key].p95,
                        }

                continuous_frames = []
                post_motion: dict[str, dict[str, float | str | None]] = {}
                post_segment_changes: dict[
                    str, dict[str, dict[str, float | str | None]]
                ] = {}
                for step in range(1, POST_REACQUISITION_FRAME_STEPS + 1):
                    post_frame = i + step
                    if post_frame >= len(frames) or not available(post_frame, base):
                        break
                    continuous_frames.append(post_frame)
                    frame_key = str(post_frame)
                    if post_frame in motions[base]:
                        post_motion[frame_key] = _deviation(
                            motions[base][post_frame], motion_profiles[base]
                        )
                    for key in adjacent[base]:
                        if post_frame in changes[key]:
                            post_segment_changes.setdefault(key, {})[frame_key] = _deviation(
                                changes[key][post_frame], change_profiles[key]
                            )
                evidence = {
                    "reacquisition_motion": motion_evidence,
                    "segment_consistency": segment_consistency,
                    "post_reacquisition": {
                        "diagnostic_window_frame_steps": POST_REACQUISITION_FRAME_STEPS,
                        "consecutive_frames_available": len(continuous_frames),
                        "available_frames": continuous_frames,
                        "motion_evidence": post_motion,
                        "segment_change_evidence": post_segment_changes,
                    },
                }
                reacquisition_events.append(LandmarkReacquisitionEvent(
                    landmark=names[base], last_valid_frame=last_valid, reacquired_frame=i,
                    gap_frame_count=len(unavailable_since_valid),
                    unavailable_frames=list(unavailable_since_valid),
                    previous_position=(float(previous["x"]), float(previous["y"])),
                    reacquired_position=(float(current["x"]), float(current["y"])),
                    origin="MEDIAPIPE", observation_context="REACQUIRED", evidence=evidence,
                ))
                reacquired_keys.add((base, i))
            last_valid = i
            unavailable_since_valid = []

    for i in used:
        for base in profile_landmarks:
            if not available(i, base):
                continue
            observations.append({
                "frame": i,
                "landmark": names[base],
                "provenance": "MEDIAPIPE",
                "observation_context": "REACQUIRED" if (base, i) in reacquired_keys else "CONTINUOUS_OBSERVATION",
                "motion": _deviation(motions[base][i], motion_profiles[base]) if i in motions[base] else None,
                "segments": {key: _deviation(segment_values[key][i], segment_profiles[key]) for key in adjacent[base] if i in segment_values[key]},
                "segment_changes": {key: _deviation(changes[key][i], change_profiles[key]) for key in adjacent[base] if i in changes[key]},
            })

    profile = KinematicProfile(
        source_frame_count=len(frames), calibration_frame_count=len(used),
        calibration_frames_used=used,
        frames_excluded_due_to_existing_quality_requirements=excluded,
        selected_side=selected_side if selected_side in {"left", "right"} else None,
        body_scale_method=(
            "median per-frame sum of raw shoulder-hip, hip-knee, and knee-ankle lengths"
            if configured_segments == PROFILE_SEGMENTS
            else "median per-frame sum of configured raw linked-segment lengths"
        ),
        body_scale_segments=list(configured_segments),
        body_scale=body_scale, segment_profiles=segment_profiles,
        motion_profiles=motion_profiles, segment_change_profiles=change_profiles,
        warnings=warnings,
        provenance={"observation_origin": "MEDIAPIPE", "derivation": "descriptive_statistics_only"},
    )
    return {"profile": asdict(profile), "observations": observations,
            "reacquisition_events": [asdict(event) for event in reacquisition_events],
            "reacquisition_definition": "A valid selected-side MediaPipe landmark observation preceded by one or more unavailable frames and an earlier valid observation.",
            "post_reacquisition_window": f"Up to {POST_REACQUISITION_FRAME_STEPS} immediately consecutive available frame steps after reacquisition; it ends at the first unavailable frame or video end.",
            "diagnostic_only": True,
            "interpretation": "A deviation means the observation differs from behaviour typically observed for this subject in this video; it is not a confirmed tracking error."}
