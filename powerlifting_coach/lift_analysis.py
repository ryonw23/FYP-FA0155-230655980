"""Shared pose-session preparation for lift-specific analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pose_calibration import CalibrationSegments, build_kinematic_self_calibration

Landmark = dict[str, float]
LandmarkFrame = dict[str, Landmark]


@dataclass(frozen=True)
class LiftLandmarkConfiguration:
    retained_landmarks: tuple[str, ...]
    side_selection_bases: tuple[str, ...]
    calibration_segments: CalibrationSegments


@dataclass(frozen=True)
class LiftAnalysisContext:
    lift: str
    fps: float
    frames: list[LandmarkFrame]
    selected_side: str | None
    visibility: dict[str, Any]
    visibility_threshold: float
    calibration_profile: dict[str, Any]
    extraction_information: dict[str, Any]
    recording_reliability: dict[str, Any]


def filter_landmarks(
    frames: list[LandmarkFrame], retained_landmarks: tuple[str, ...]
) -> list[LandmarkFrame]:
    """Retain only the landmarks explicitly configured for a lift."""

    retained = set(retained_landmarks)
    return [
        {name: landmark for name, landmark in frame.items() if name in retained}
        for frame in frames
    ]


def side_visibility(
    frames: list[LandmarkFrame], side: str, landmark_bases: tuple[str, ...]
) -> tuple[float, dict[str, float]]:
    """Aggregate visibility/presence for one side, counting absence as zero."""

    by_landmark: dict[str, float] = {}
    for base in landmark_bases:
        name = f"{side}_{base}"
        values = [
            min(
                float(frame.get(name, {}).get("visibility", 0.0)),
                float(frame.get(name, {}).get("presence", 1.0)),
            )
            for frame in frames
        ]
        by_landmark[name] = round(sum(values) / len(values), 3) if values else 0.0
    score = sum(by_landmark.values()) / len(by_landmark) if by_landmark else 0.0
    return round(score, 3), by_landmark


def select_analysis_side(
    frames: list[LandmarkFrame],
    landmark_bases: tuple[str, ...],
    minimum_score: float = 0.0,
) -> dict[str, Any]:
    """Select the better-observed side while exposing both aggregate scores."""

    left_score, left_landmarks = side_visibility(frames, "left", landmark_bases)
    right_score, right_landmarks = side_visibility(frames, "right", landmark_bases)
    best_score = max(left_score, right_score)
    selected = (
        ("left" if left_score >= right_score else "right")
        if best_score >= minimum_score
        else None
    )
    return {
        "left_score": left_score,
        "right_score": right_score,
        "selected_side": selected,
        "landmark_scores": {**left_landmarks, **right_landmarks},
    }


def select_evidence_side(
    frames: list[LandmarkFrame],
    landmark_bases: tuple[str, ...],
    visibility_threshold: float,
    source_frames: list[int] | None = None,
) -> dict[str, Any]:
    """Choose the side with the most complete, co-observed evidence."""

    source = source_frames if source_frames is not None else list(range(len(frames)))
    evidence: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        names = tuple(f"{side}_{base}" for base in landmark_bases)
        complete_frames = []
        quality_sum = 0.0
        for frame_index in source:
            joint_scores = []
            for name in names:
                landmark = frames[frame_index].get(name, {})
                score = min(
                    float(landmark.get("visibility", 0.0)),
                    float(landmark.get("presence", 1.0)),
                )
                joint_scores.append(score)
            if joint_scores and min(joint_scores) >= visibility_threshold:
                complete_frames.append(frame_index)
                quality_sum += sum(joint_scores) / len(joint_scores)

        evidence[side] = {
            "complete_frames": complete_frames,
            "complete_frame_count": len(complete_frames),
            "complete_frame_ratio": (
                round(len(complete_frames) / len(source), 3) if source else 0.0
            ),
            "quality_sum": round(quality_sum, 3),
        }
    selected = max(
        ("left", "right"),
        key=lambda side: (
            evidence[side]["complete_frame_count"],
            evidence[side]["quality_sum"],
            side == "left",
        ),
    )
    if not evidence[selected]["complete_frames"]:
        selected = None
    return {"selected_side": selected, "by_side": evidence}


def prepare_lift_analysis_context(
    *,
    lift: str,
    frames: list[LandmarkFrame],
    fps: float,
    visibility_threshold: float,
    landmarks: LiftLandmarkConfiguration,
    processing_metadata: dict[str, Any] | None = None,
    minimum_side_visibility_score: float = 0.0,
    minimum_usable_frame_ratio: float = 0.0,
    phase_tracking_bases: tuple[str, ...] = (),
) -> LiftAnalysisContext:
    """Prepare shared pose/session evidence without applying lift technique rules."""

    filtered_frames = filter_landmarks(frames, landmarks.retained_landmarks)
    visibility = select_analysis_side(filtered_frames, landmarks.side_selection_bases)
    phase_selection = select_evidence_side(
        filtered_frames, phase_tracking_bases, visibility_threshold
    )
    selected_side = phase_selection["selected_side"]
    usable_frames = (
        phase_selection["by_side"][selected_side]["complete_frame_count"]
        if selected_side else 0
    )
    usable_ratio = usable_frames / len(filtered_frames) if filtered_frames else 0.0
    metadata = processing_metadata or {}
    declared_view = str(metadata.get("camera_view", "")).upper()
    reasons = []
    if not selected_side:
        reasons.append("no_usable_phase_trajectory")
    warnings = []
    if declared_view.startswith("FRONT"):
        warnings.append("declared_front_view_may_limit_side_view_metrics")

    reliability = {
        "left_score": visibility["left_score"],
        "right_score": visibility["right_score"],
        "selected_side": selected_side,
        "selected_side_usable_frames": usable_frames,
        "selected_side_usable_frame_ratio": round(usable_ratio, 3),
        "minimum_side_visibility_score": minimum_side_visibility_score,
        "minimum_usable_frame_ratio": minimum_usable_frame_ratio,
        "phase_tracking_landmarks": list(phase_tracking_bases),
        "phase_tracking_side_evidence": phase_selection["by_side"],
        "rejection_reasons": reasons,
        "warnings": warnings,
    }

    extraction_information = {
        key: value
        for key, value in metadata.items()
        if key
        in {
            "fps",
            "pose_counts_by_frame",
            "total_processed_frames",
            "processing_time_seconds",
            "frame_width",
            "frame_height",
        }
    }
    calibration = build_kinematic_self_calibration(
        filtered_frames,
        selected_side,
        visibility_threshold,
        landmarks.calibration_segments,
    )
    return LiftAnalysisContext(
        lift=lift,
        fps=fps,
        frames=filtered_frames,
        selected_side=selected_side,
        visibility=visibility,
        visibility_threshold=visibility_threshold,
        calibration_profile=calibration,
        extraction_information=extraction_information,
        recording_reliability=reliability,
    )
