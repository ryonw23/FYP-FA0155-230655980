"""Pose extraction using OpenCV and MediaPipe Tasks Pose Landmarker."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .annotate_pose_video import (
    LandmarkFrame,
    POSE_LANDMARK_NAMES,
    _load_task_dependencies,
    draw_landmarks_on_bgr,
    task_result_to_landmarks,
)
from .pose_config import DEFAULT_MEDIAPIPE_POSE_CONFIG, cpu_base_options
from .video_io import open_oriented_video, read_oriented_frame

REQUIRED_JOINTS = ("left_shoulder", "left_hip", "left_knee", "left_ankle")


@dataclass(frozen=True)
class PoseExtractionResult:
    """Frames plus source metadata already observed during pose extraction."""

    frames: list[LandmarkFrame]
    metadata: dict[str, Any]


def render_pose_overlay(
    video_path: str | Path,
    output_path: str | Path,
    frames: list[LandmarkFrame],
    connections: Collection[tuple[str, str]],
    visibility_threshold: float,
) -> None:
    """Render already-extracted landmarks without repeating pose inference."""

    cv2, _, _, _ = _load_task_dependencies()
    capture, video_metadata = open_oriented_video(cv2, video_path)
    fps = video_metadata.fps
    width = video_metadata.display_width
    height = video_metadata.display_height
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create output MP4 with OpenCV VideoWriter.")

    displayed_landmarks = {name for connection in connections for name in connection}
    frame_index = 0
    try:
        while True:
            success, frame_bgr = read_oriented_frame(
                cv2, capture, video_metadata.rotation_degrees
            )
            if not success:
                break
            landmarks = frames[frame_index] if frame_index < len(frames) else {}
            writer.write(
                draw_landmarks_on_bgr(
                    cv2,
                    frame_bgr,
                    landmarks,
                    connections,
                    visibility_threshold,
                    displayed_landmarks,
                )
                if landmarks and connections
                else frame_bgr
            )
            frame_index += 1
    finally:
        capture.release()
        writer.release()


def extract_pose(
    video_path: str | Path,
    required_joints: tuple[str, ...] = REQUIRED_JOINTS,
    visibility_threshold: float = 0.5,
    log_path: str | Path | None = None,
    annotated_video_path: str | Path | None = None,
    task_model_path: str | Path = "models/pose_landmarker_full.task",
    num_poses: int = 1,
    retained_landmarks: Collection[str] | None = None,
    overlay_connections: Collection[tuple[str, str]] | None = None,
    return_result: bool = False,
    min_pose_detection_confidence: float | None = None,
    min_pose_presence_confidence: float | None = None,
    min_tracking_confidence: float | None = None,
) -> list[LandmarkFrame] | PoseExtractionResult:
    """Extract named landmarks for each frame with the MediaPipe Tasks API."""

    started_at = perf_counter()
    cv2, mp_module, tasks_python, vision = _load_task_dependencies()
    video_path = Path(video_path)
    model_path = Path(task_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

    capture, video_metadata = open_oriented_video(cv2, video_path)
    fps = video_metadata.fps
    width = video_metadata.display_width
    height = video_metadata.display_height

    pose_frames: list[LandmarkFrame] = []
    low_visibility_frames: list[dict[str, Any]] = []
    visibility_log: list[dict[str, float]] = []
    landmark_time_series: list[dict[str, Any]] = []
    pose_counts: list[int] = []
    writer = None
    if annotated_video_path is not None:
        if overlay_connections is None:
            capture.release()
            raise ValueError(
                "overlay_connections must be provided when annotated_video_path is set"
            )
        output_path = Path(annotated_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create output MP4 with OpenCV VideoWriter.")

    # Inference confidence is independent of downstream landmark visibility.
    detection_confidence = (
        DEFAULT_MEDIAPIPE_POSE_CONFIG.pose_detection_confidence
        if min_pose_detection_confidence is None
        else min_pose_detection_confidence
    )
    presence_confidence = (
        DEFAULT_MEDIAPIPE_POSE_CONFIG.pose_presence_confidence
        if min_pose_presence_confidence is None
        else min_pose_presence_confidence
    )
    tracking_confidence = (
        DEFAULT_MEDIAPIPE_POSE_CONFIG.tracking_confidence
        if min_tracking_confidence is None
        else min_tracking_confidence
    )
    options = vision.PoseLandmarkerOptions(
        # MediaPipe 1.0.1's macOS Metal path can abort the whole process while
        # opening this graph. Keep the supported 0.10.21 runtime on its CPU
        # delegate; a fatal native abort cannot be recovered with try/except.
        base_options=cpu_base_options(tasks_python, str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=detection_confidence,
        min_pose_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
        output_segmentation_masks=False,
    )

    def record_frame(
        frame_index: int, timestamp_ms: int, landmarks: LandmarkFrame, pose_count: int
    ) -> None:
        pose_frames.append(landmarks)
        pose_counts.append(pose_count)
        landmark_time_series.append(
            {
                "frame": frame_index,
                "timestamp_ms": timestamp_ms,
                "timestamp_seconds": round(timestamp_ms / 1000, 3),
                "person_count": pose_count,
                "landmarks": landmarks,
            }
        )
        frame_visibility = {
            joint: float(landmarks.get(joint, {}).get("visibility", 0.0))
            for joint in required_joints
        }
        visibility_log.append(frame_visibility)
        below_threshold = [
            joint
            for joint, visibility in frame_visibility.items()
            if visibility < visibility_threshold
        ]
        if below_threshold:
            low_visibility_frames.append(
                {"frame": frame_index, "joints": below_threshold}
            )

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                success, frame_bgr = read_oriented_frame(
                    cv2, capture, video_metadata.rotation_degrees
                )
                if not success:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp_module.Image(
                    image_format=mp_module.ImageFormat.SRGB,
                    data=frame_rgb,
                )
                timestamp_ms = int((frame_index / fps) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                landmarks = task_result_to_landmarks(result, retained_landmarks)
                pose_count = len(getattr(result, "pose_landmarks", []) or [])
                record_frame(frame_index, timestamp_ms, landmarks, pose_count)

                if writer is not None:
                    output_frame = frame_bgr
                    if getattr(result, "pose_landmarks", None):
                        output_frame = draw_landmarks_on_bgr(
                            cv2,
                            frame_bgr,
                            landmarks,
                            connections=overlay_connections,
                            visibility_threshold=visibility_threshold,
                        )
                    writer.write(output_frame)

                frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    total = len(pose_frames)
    processing_time = round(perf_counter() - started_at, 3)
    metadata = {
        "fps": float(fps),
        "frame_width": int(width),
        "frame_height": int(height),
        "source_frame_width": video_metadata.encoded_width,
        "source_frame_height": video_metadata.encoded_height,
        "display_rotation_degrees": video_metadata.rotation_degrees,
        "orientation_applied_before_pose_inference": bool(
            video_metadata.rotation_degrees
        ),
        "pose_counts_by_frame": pose_counts,
        "total_processed_frames": total,
        "processing_time_seconds": processing_time,
        "mediapipe_confidence_thresholds": {
            "pose_detection": detection_confidence,
            "pose_presence": presence_confidence,
            "tracking": tracking_confidence,
        },
    }
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        reliable = total - len(low_visibility_frames)
        summary = {
            "video_path": str(video_path),
            "pose_backend": "mediapipe_tasks_pose_landmarker",
            "model_path": str(model_path),
            "total_frames": total,
            "fps": float(fps),
            "frame_width": int(width),
            "frame_height": int(height),
            "source_frame_width": video_metadata.encoded_width,
            "source_frame_height": video_metadata.encoded_height,
            "display_rotation_degrees": video_metadata.rotation_degrees,
            "orientation_applied_before_pose_inference": bool(
                video_metadata.rotation_degrees
            ),
            "retained_landmark_names": list(retained_landmarks or POSE_LANDMARK_NAMES),
            "landmark_filtering": {
                "applied": retained_landmarks is not None,
                "stage": "immediately_after_pose_inference",
                "model_output_modified": False,
            },
            "required_joints": list(required_joints),
            "visibility_threshold": visibility_threshold,
            "low_visibility_frames": low_visibility_frames,
            "visibility_by_frame": visibility_log,
            "pose_counts_by_frame": pose_counts,
            "processing_time_seconds": processing_time,
            "landmark_time_series": landmark_time_series,
            "percentage_frames_above_threshold": (
                (reliable / total * 100) if total else 0.0
            ),
        }
        log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if return_result:
        return PoseExtractionResult(frames=pose_frames, metadata=metadata)
    return pose_frames
