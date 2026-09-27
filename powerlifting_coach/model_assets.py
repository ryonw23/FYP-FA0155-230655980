"""Download and cache the two model assets used by the application."""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path


MODEL_DIRECTORY = Path(os.getenv("POWERLIFTING_MODEL_DIR", "models"))
POSE_MODEL_PATH = MODEL_DIRECTORY / "pose_landmarker_full.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# Meta's own repository requires accepting its gated terms before download.
# This public llama.cpp conversion is downloadable without a Hugging Face login,
# but remains governed by the Llama 3.2 Community License (not Apache-2.0).
# Pin the quantisation filename instead of discovering an arbitrary asset.
LANGUAGE_MODEL_REPOSITORY = "bartowski/Llama-3.2-3B-Instruct-GGUF"
LANGUAGE_MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LANGUAGE_MODEL_PATH = MODEL_DIRECTORY / LANGUAGE_MODEL_FILENAME
MINIMUM_GGUF_BYTES = 1_000_000_000


class ModelDownloadError(RuntimeError):
    """A required model could not be obtained from its official source."""


def valid_gguf(path: Path) -> bool:
    """Reject HTML/error bodies and truncated files masquerading as weights."""

    if not path.is_file() or path.stat().st_size < MINIMUM_GGUF_BYTES:
        return False
    with path.open("rb") as model_file:
        return model_file.read(4) == b"GGUF"


def valid_pose_model(path: Path) -> bool:
    """Apply the existing non-empty-file validation used for pose assets."""

    return path.is_file() and bool(path.stat().st_size)


def download_file(
    url: str,
    destination: Path,
    display_name: str,
    validator: Callable[[Path], bool] | None = None,
) -> Path:
    """Download *url* atomically, reusing an existing non-empty file."""

    is_valid = validator or (
        lambda candidate: candidate.is_file() and bool(candidate.stat().st_size)
    )
    if is_valid(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        print(f"[models] Downloading {display_name} to {destination} …")
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        if not is_valid(partial):
            raise OSError("the downloaded file is incomplete or has an invalid format")
        partial.replace(destination)
        print(f"[models] {display_name} is ready ({destination.stat().st_size:,} bytes).")
        return destination
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Could not download {display_name} from {url} to {destination}: {exc}. "
            "Check the internet connection and write permissions, then try again."
        ) from exc


def ensure_pose_model(path: Path = POSE_MODEL_PATH) -> Path:
    """Return the cached MediaPipe Pose Landmarker model, downloading if needed."""

    return download_file(
        POSE_MODEL_URL,
        path,
        "MediaPipe Pose Landmarker Full",
        valid_pose_model,
    )


def ensure_language_model(path: Path = LANGUAGE_MODEL_PATH) -> Path:
    """Return the cached public Llama 3.2 GGUF model, downloading if needed."""

    url = (
        f"https://huggingface.co/{LANGUAGE_MODEL_REPOSITORY}/resolve/main/"
        f"{LANGUAGE_MODEL_FILENAME}?download=true"
    )
    return download_file(url, path, "Llama 3.2 3B Instruct Q4_K_M", valid_gguf)
