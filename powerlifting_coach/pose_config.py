"""Central production configuration for MediaPipe pose inference."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MediaPipePoseConfig:
    """Confidence thresholds used by every normal pose-inference path."""

    pose_detection_confidence: float = 0.5
    pose_presence_confidence: float = 0.5
    tracking_confidence: float = 0.5


DEFAULT_MEDIAPIPE_POSE_CONFIG = MediaPipePoseConfig()


def cpu_base_options(tasks_python, model_path: str):
    """Build explicit CPU options so pose inference never selects Metal/GPU."""

    return tasks_python.BaseOptions(
        model_asset_path=model_path,
        delegate=tasks_python.BaseOptions.Delegate.CPU,
    )
