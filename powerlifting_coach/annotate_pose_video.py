"""Annotate MP4 files with MediaPipe Tasks Pose Landmarker joints."""

from __future__ import annotations

import argparse
from collections.abc import Collection
import importlib
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .motion_analysis import SQUAT_LANDMARKS, SQUAT_OVERLAY_CONNECTIONS
from .pose_config import DEFAULT_MEDIAPIPE_POSE_CONFIG, cpu_base_options
from .video_io import open_oriented_video, read_oriented_frame

POSE_LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)

LandmarkFrame = dict[str, dict[str, float]]
SUPPORTED_MEDIAPIPE_VERSION = "0.10.21"


def _load_task_dependencies() -> tuple[Any, Any, Any, Any]:
    """Load OpenCV and MediaPipe Tasks dependencies when annotation runs."""

    installed_version = version("mediapipe")
    if installed_version != SUPPORTED_MEDIAPIPE_VERSION:
        raise RuntimeError(
            "Unsupported MediaPipe version "
            f"{installed_version}; install requirements.txt to use the tested "
            f"{SUPPORTED_MEDIAPIPE_VERSION} CPU runtime. MediaPipe 1.0.1 can abort "
            "the process in its Apple-silicon Metal pose graph."
        )
    cv2_module = importlib.import_module("cv2")
    mp_module = importlib.import_module("mediapipe")
    tasks_python = importlib.import_module("mediapipe.tasks.python")
    vision = importlib.import_module("mediapipe.tasks.python.vision")
    return cv2_module, mp_module, tasks_python, vision


def task_result_to_landmarks(
    result: Any, retained_landmarks: Collection[str] | None = None
) -> LandmarkFrame:
    """Name the first pose and optionally retain an explicit downstream subset."""

    if not getattr(result, "pose_landmarks", None):
        return {}

    retained = set(retained_landmarks) if retained_landmarks is not None else None
    return {
        POSE_LANDMARK_NAMES[index]: {
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(getattr(landmark, "z", 0.0)),
            "visibility": float(getattr(landmark, "visibility", 1.0)),
            "presence": float(getattr(landmark, "presence", 1.0)),
        }
        for index, landmark in enumerate(result.pose_landmarks[0])
        if index < len(POSE_LANDMARK_NAMES)
        and (retained is None or POSE_LANDMARK_NAMES[index] in retained)
    }


def draw_landmarks_on_bgr(
    cv2: Any,
    frame_bgr: Any,
    landmarks: LandmarkFrame,
    connections: Collection[tuple[str, str]],
    visibility_threshold: float = 0.5,
    displayed_landmarks: Collection[str] | None = None,
) -> Any:
    """Draw retained pose dots and named links on an OpenCV BGR frame."""

    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    points: dict[str, tuple[int, int]] = {}

    displayed = set(displayed_landmarks) if displayed_landmarks is not None else None
    for name, landmark in landmarks.items():
        if displayed is not None and name not in displayed:
            continue
        visibility = float(landmark.get("visibility", 1.0))
        presence = float(landmark.get("presence", 1.0))
        if visibility < visibility_threshold or presence < visibility_threshold:
            continue
        x = min(max(int(landmark["x"] * width), 0), width - 1)
        y = min(max(int(landmark["y"] * height), 0), height - 1)
        points[name] = (x, y)

    for start, end in connections:
        if start in points and end in points:
            cv2.line(annotated, points[start], points[end], (0, 255, 0), 2)

    for point in points.values():
        cv2.circle(annotated, point, 4, (0, 0, 255), -1)

    return annotated


def annotate_pose_video(
    input_video: str | Path,
    output_video: str | Path,
    model_path: str | Path = "models/pose_landmarker_full.task",
    num_poses: int = 1,
    min_confidence: float = DEFAULT_MEDIAPIPE_POSE_CONFIG.pose_detection_confidence,
) -> str:
    """Read an MP4, overlay Pose Landmarker joints, and write an annotated MP4."""

    cv2, mp_module, tasks_python, vision = _load_task_dependencies()
    input_video = Path(input_video)
    output_video = Path(output_video)
    model_path = Path(model_path)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    capture, video_metadata = open_oriented_video(cv2, input_video)
    fps = video_metadata.fps
    width = video_metadata.display_width
    height = video_metadata.display_height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create output MP4 with OpenCV VideoWriter.")

    options = vision.PoseLandmarkerOptions(
        base_options=cpu_base_options(tasks_python, str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
        output_segmentation_masks=False,
    )

    frame_index = 0
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
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

                annotated_bgr = frame_bgr
                landmarks = task_result_to_landmarks(result, SQUAT_LANDMARKS)
                if landmarks:
                    annotated_bgr = draw_landmarks_on_bgr(
                        cv2,
                        frame_bgr,
                        landmarks,
                        connections=SQUAT_OVERLAY_CONNECTIONS,
                        visibility_threshold=min_confidence,
                    )

                writer.write(annotated_bgr)
                frame_index += 1
    finally:
        capture.release()
        writer.release()

    return str(output_video)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate an MP4 with MediaPipe Tasks Pose Landmarker joints."
    )
    parser.add_argument("--input", required=True, help="Path to source MP4")
    parser.add_argument("--output", required=True, help="Path for annotated MP4")
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_full.task",
        help="Path to MediaPipe .task model bundle",
    )
    args = parser.parse_args()

    result_path = annotate_pose_video(
        input_video=args.input,
        output_video=args.output,
        model_path=args.model,
    )
    print(f"Annotated video written to: {result_path}")


if __name__ == "__main__":
    main()
