"""Install the model assets required for offline application use."""

from __future__ import annotations

import os
from pathlib import Path

from .model_assets import (
    LANGUAGE_MODEL_PATH,
    POSE_MODEL_PATH,
    ModelDownloadError,
    ensure_language_model,
    ensure_pose_model,
    valid_gguf,
    valid_pose_model,
)


def _prepare(name: str, path: Path, validator, downloader) -> None:
    if validator(path):
        print(f"[setup] Reusing valid {name}: {path}")
        return
    if path.exists():
        print(f"[setup] Existing {name} is invalid and will be replaced: {path}")
    else:
        print(f"[setup] {name} is missing: {path}")
    downloader(path)


def main() -> int:
    """Ensure required models are valid and cached before the app is launched."""

    language_path = Path(os.getenv("POWERLIFTING_LLM_PATH", str(LANGUAGE_MODEL_PATH)))
    print("[setup] Preparing models for offline analysis.")
    try:
        _prepare("MediaPipe Pose Landmarker Full model", POSE_MODEL_PATH, valid_pose_model, ensure_pose_model)
        _prepare("Llama 3.2 3B GGUF model", language_path, valid_gguf, ensure_language_model)
    except ModelDownloadError as exc:
        print(f"[setup] FAILED: {exc}")
        print("[setup] Setup is incomplete. Fix the problem above and run python -m powerlifting_coach.setup again.")
        return 1
    print("[setup] Complete. All required models are cached and valid; the app can now run offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
