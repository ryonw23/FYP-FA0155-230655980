"""OpenCV video decoding helpers that honour display-orientation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoDecodeMetadata:
    """Geometry applied consistently to inference frames and rendered output."""

    fps: float
    encoded_width: int
    encoded_height: int
    display_width: int
    display_height: int
    rotation_degrees: int


def _normalise_rotation(value: float) -> int:
    """Return the nearest supported clockwise display rotation."""

    return int(round(float(value) / 90.0) * 90) % 360


def open_oriented_video(
    cv2: Any, video_path: str | Path
) -> tuple[Any, VideoDecodeMetadata]:
    """Open *video_path* with explicit, deterministic display rotation.

    OpenCV's FFmpeg and AVFoundation backends can apply stream rotation
    automatically.  Disable that behaviour and apply ``ORIENTATION_META``
    ourselves so inference and either overlay-writing path see identical pixels.
    """

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    orientation_meta = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    rotation = (
        _normalise_rotation(capture.get(orientation_meta))
        if orientation_meta is not None
        else 0
    )
    if orientation_auto is not None:
        # Read the metadata first, then force raw coded frames. This avoids
        # backend-dependent implicit rotation and, importantly, double rotation.
        capture.set(orientation_auto, 0)

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    encoded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    encoded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if encoded_width <= 0 or encoded_height <= 0:
        capture.release()
        raise RuntimeError("Video dimensions could not be read.")

    if rotation in {90, 270}:
        display_width, display_height = encoded_height, encoded_width
    else:
        display_width, display_height = encoded_width, encoded_height
    return capture, VideoDecodeMetadata(
        fps=fps,
        encoded_width=encoded_width,
        encoded_height=encoded_height,
        display_width=display_width,
        display_height=display_height,
        rotation_degrees=rotation,
    )


def read_oriented_frame(cv2: Any, capture: Any, rotation_degrees: int) -> tuple[bool, Any]:
    """Read one frame and rotate it into the stream's intended display view."""

    success, frame = capture.read()
    if not success:
        return False, frame
    rotate_codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotation_degrees in rotate_codes:
        frame = cv2.rotate(frame, rotate_codes[rotation_degrees])
    return True, frame
