"""Offline sensitivity evaluation for deadlift body-scale normalisation."""

from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from ..deadlift_analysis import (
    DEADLIFT_CALIBRATION_SEGMENTS,
    DeadliftAnalysisConfig,
    analyse_deadlift_landmarks,
    assess_early_hip_shoulder_coordination,
)

LIMITATIONS = [
    "Normalisation invariance does not establish tracking or biomechanical accuracy.",
    "No accuracy claim is possible without independent reference labels.",
    "Comparisons on previously inspected clips are exploratory.",
    "The setup-window result is a fixed-threshold sensitivity comparison, not a separately validated classifier.",
]


def load_landmark_export(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {"recording": path.stem}
    if isinstance(payload.get("frames"), list):
        return payload["frames"], {
            **payload,
            "recording": payload.get("video_or_session_id", path.stem),
        }
    series = payload.get("landmark_time_series")
    if isinstance(series, list):
        metadata = {
            key: payload.get(key)
            for key in ("fps", "frame_width", "frame_height", "camera_view")
        }
        metadata["recording"] = payload.get("video_path", path.stem)
        return [item["landmarks"] for item in series], metadata
    raise ValueError(
        "Input must be a frame list or contain frames/landmark_time_series"
    )


def _eligible(landmark: Any, threshold: float) -> bool:
    return bool(
        isinstance(landmark, dict)
        and all(
            isinstance(landmark.get(axis), (int, float))
            and math.isfinite(float(landmark[axis]))
            for axis in ("x", "y")
        )
        and float(landmark.get("visibility", 0)) >= threshold
        and float(landmark.get("presence", 1)) >= threshold
    )


def linked_chain_scale(
    frames: list[dict[str, Any]], side: str, source_frames: list[int], threshold: float
) -> dict[str, Any]:
    samples = []
    for index in source_frames:
        lengths = []
        for first, second in DEADLIFT_CALIBRATION_SEGMENTS.values():
            a, b = frames[index].get(f"{side}_{first}"), frames[index].get(
                f"{side}_{second}"
            )
            if not (_eligible(a, threshold) and _eligible(b, threshold)):
                break
            lengths.append(
                math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))
            )
        if len(lengths) == len(DEADLIFT_CALIBRATION_SEGMENTS) and sum(lengths) > 0:
            samples.append(sum(lengths))
    return {
        "value": float(median(samples)) if samples else None,
        "sample_count": len(samples),
    }


def coordination_measurement(
    frames: list[dict[str, Any]],
    side: str,
    movement_window: list[int],
    threshold: float,
    scale: float | None,
    config: DeadliftAnalysisConfig,
) -> dict[str, Any]:
    source = [
        i
        for i in movement_window
        if _eligible(frames[i].get(f"{side}_hip"), threshold)
        and _eligible(frames[i].get(f"{side}_shoulder"), threshold)
    ]
    if not scale or len(source) < 2:
        return {
            "status": "unavailable",
            "source_frames": source,
            "classification": "INSUFFICIENT_EVIDENCE",
        }
    first, hip, shoulder = source[0], f"{side}_hip", f"{side}_shoulder"
    values = [
        (
            (float(frames[first][hip]["y"]) - float(frames[i][hip]["y"]))
            - (float(frames[first][shoulder]["y"]) - float(frames[i][shoulder]["y"]))
        )
        / scale
        for i in source
    ]
    return {
        "status": "available",
        "source_frames": source,
        "maximum_hip_minus_shoulder_vertical_progress": max(values),
        "median_hip_minus_shoulder_vertical_progress": float(median(values)),
        "endpoint_hip_minus_shoulder_vertical_progress": values[-1],
        "framewise_hip_minus_shoulder_vertical_progress": values,
        "classification": assess_early_hip_shoulder_coordination(values, config),
    }


def transform_coordinates(
    frames: list[dict[str, Any]], factor: float, translate_x: float, translate_y: float
) -> list[dict[str, Any]]:
    transformed = []
    for frame in frames:
        changed = {}
        for name, landmark in frame.items():
            item = dict(landmark)
            if isinstance(item.get("x"), (int, float)) and math.isfinite(
                float(item["x"])
            ):
                item["x"] = float(item["x"]) * factor + translate_x
            if isinstance(item.get("y"), (int, float)) and math.isfinite(
                float(item["y"])
            ):
                item["y"] = float(item["y"]) * factor + translate_y
            changed[name] = item
        transformed.append(changed)
    return transformed


def _invariance(
    original: dict[str, Any], changed: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    keys = (
        "maximum_hip_minus_shoulder_vertical_progress",
        "median_hip_minus_shoulder_vertical_progress",
        "endpoint_hip_minus_shoulder_vertical_progress",
    )
    differences = {
        key: abs(float(original[key]) - float(changed[key]))
        for key in keys
        if key in original and key in changed
    }
    return {
        "available": len(differences) == len(keys),
        "tolerance": tolerance,
        "absolute_differences": differences,
        "invariant_within_tolerance": len(differences) == len(keys)
        and max(differences.values()) <= tolerance,
    }


def evaluate_recording(
    frames: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    config: DeadliftAnalysisConfig | None = None,
    transform_factor: float = 1.7,
    translate_x: float = 0.13,
    translate_y: float = -0.09,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    cfg, metadata = config or DeadliftAnalysisConfig(), metadata or {}
    analysis = analyse_deadlift_landmarks(
        frames, float(metadata.get("fps") or 30), metadata, cfg
    )
    side, phases = analysis.get("selected_side"), analysis.get("phase_frames", {})
    production = next(
        (
            item.get("classification")
            for item in analysis.get("metric_results", [])
            if item.get("metric_id") == "deadlift_early_coordination_2d"
        ),
        None,
    )
    result = {
        "recording": metadata.get("recording"),
        "analysis_status": analysis.get("analysis_status"),
        "selected_side": side,
        "reference_window": None,
        "movement_window": None,
        "production_classification": production,
        "limitations": list(LIMITATIONS),
    }
    if not side or not phases:
        result.update(
            status="unavailable",
            reason="Production analysis did not provide a selected side and complete phase boundaries.",
        )
        return result
    setup, lift_off, knee = (
        phases["setup"],
        phases["lift_off"]["start_frame"],
        phases["knee_passing_ascent"]["start_frame"],
    )
    reference = list(range(setup["start_frame"], setup["end_frame"] + 1))
    movement = list(range(lift_off, knee + 1))
    whole = list(range(len(frames)))
    result["reference_window"] = {**setup, "frames": reference}
    result["movement_window"] = {
        "start_frame": lift_off,
        "end_frame": knee,
        "frames": movement,
    }
    current = linked_chain_scale(frames, side, whole, cfg.visibility_threshold)
    baseline = linked_chain_scale(frames, side, reference, cfg.visibility_threshold)
    cm = coordination_measurement(
        frames, side, movement, cfg.visibility_threshold, current["value"], cfg
    )
    bm = coordination_measurement(
        frames, side, movement, cfg.visibility_threshold, baseline["value"], cfg
    )
    result["scales"] = {
        "current_whole_video": current,
        "baseline_setup_window": baseline,
    }
    if current["value"] is not None and baseline["value"] is not None:
        absolute = abs(current["value"] - baseline["value"])
        result["scale_difference"] = {
            "absolute": absolute,
            "relative_to_current": absolute / current["value"],
        }
    else:
        result["scale_difference"] = {"absolute": None, "relative_to_current": None}
    result["measurements"] = {
        "current_whole_video": cm,
        "baseline_setup_window_fixed_threshold_sensitivity": bm,
    }
    changed = transform_coordinates(frames, transform_factor, translate_x, translate_y)
    tc = linked_chain_scale(changed, side, whole, cfg.visibility_threshold)
    tb = linked_chain_scale(changed, side, reference, cfg.visibility_threshold)
    tcm = coordination_measurement(
        changed, side, movement, cfg.visibility_threshold, tc["value"], cfg
    )
    tbm = coordination_measurement(
        changed, side, movement, cfg.visibility_threshold, tb["value"], cfg
    )
    result["coordinate_transformation_check"] = {
        "factor": transform_factor,
        "translation": {"x": translate_x, "y": translate_y},
        "selected_side_and_windows_held_fixed": True,
        "clipping_or_pose_reextraction": False,
        "current_whole_video": _invariance(cm, tcm, tolerance),
        "baseline_setup_window": _invariance(bm, tbm, tolerance),
        "limitation": LIMITATIONS[0],
    }
    result["status"] = (
        "available" if cm["status"] == bm["status"] == "available" else "unavailable"
    )
    return result


def comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        scales, measures = result.get("scales", {}), result.get("measurements", {})
        current, baseline = scales.get("current_whole_video", {}), scales.get(
            "baseline_setup_window", {}
        )
        cm, bm = measures.get("current_whole_video", {}), measures.get(
            "baseline_setup_window_fixed_threshold_sensitivity", {}
        )
        label = lambda window: (
            f'{window["start_frame"]}-{window["end_frame"]}' if window else None
        )
        rows.append(
            {
                "recording": result.get("recording"),
                "status": result.get("status"),
                "selected_side": result.get("selected_side"),
                "reference_window": label(result.get("reference_window")),
                "movement_window": label(result.get("movement_window")),
                "current_scale": current.get("value"),
                "current_samples": current.get("sample_count"),
                "setup_scale": baseline.get("value"),
                "setup_samples": baseline.get("sample_count"),
                "absolute_scale_difference": result.get("scale_difference", {}).get(
                    "absolute"
                ),
                "relative_scale_difference": result.get("scale_difference", {}).get(
                    "relative_to_current"
                ),
                "current_max_progress": cm.get(
                    "maximum_hip_minus_shoulder_vertical_progress"
                ),
                "setup_max_progress": bm.get(
                    "maximum_hip_minus_shoulder_vertical_progress"
                ),
                "production_classification": result.get("production_classification"),
                "setup_fixed_threshold_classification": bm.get("classification"),
                "current_transform_invariant": result.get(
                    "coordinate_transformation_check", {}
                )
                .get("current_whole_video", {})
                .get("invariant_within_tolerance"),
                "setup_transform_invariant": result.get(
                    "coordinate_transformation_check", {}
                )
                .get("baseline_setup_window", {})
                .get("invariant_within_tolerance"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline deadlift body-scale sensitivity evaluation"
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, required=True, dest="json_path")
    parser.add_argument("--csv", type=Path, required=True, dest="csv_path")
    args = parser.parse_args()
    results = [evaluate_recording(*load_landmark_export(path)) for path in args.inputs]
    args.json_path.write_text(
        json.dumps(
            {
                "evaluation": "deadlift_body_scale_normalisation",
                "results": results,
                "limitations": LIMITATIONS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = comparison_rows(results)
    with args.csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
