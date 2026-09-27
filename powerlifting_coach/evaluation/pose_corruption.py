"""Controlled, offline corruption of raw selected-side pose trajectories.

This module evaluates the *existing* per-video calibration by changing deep
copies of MediaPipe observations. Perturbation levels are experimental inputs,
not anatomical quantities or production plausibility thresholds. Synthetic
responses do not establish MediaPipe accuracy, real-world error frequency,
biomechanical ground truth, or coaching accuracy.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal

from ..pose_calibration import PROFILE_SEGMENTS, build_kinematic_self_calibration

CorruptionType = Literal[
    "single_frame_offset", "persistent_offset", "dropout", "dropout_reacquisition_offset"
]
Direction = Literal["horizontal", "vertical", "diagonal"]


@dataclass(frozen=True)
class CorruptionScenario:
    """One deterministic corruption applied in isolation to a clean control."""

    scenario_id: str
    landmark: str
    source_frame: int
    corruption_type: CorruptionType
    magnitude_body_scale: float = 0.0
    direction: Direction = "horizontal"
    run_length: int = 1
    random_seed: int | None = None


@dataclass(frozen=True)
class CorruptionRecord:
    video_or_session_id: str | None
    source_frame: int
    landmark: str
    corruption_type: str
    magnitude_body_scale: float
    direction: str | None
    original_coordinates: tuple[float, float] | None
    corrupted_coordinates: tuple[float, float] | None
    random_seed: int | None


@dataclass
class CorruptionEvaluationResult:
    video_or_session_id: str | None
    selected_side: str
    visibility_threshold: float
    experimental_body_scale: float | None
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=lambda: [
        "Perturbation severities are experimental levels, not production thresholds.",
        "Synthetic corruption does not measure MediaPipe accuracy or real-world error frequency.",
        "This evaluation does not establish biomechanical ground truth or coaching accuracy.",
    ])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compact_rows(self) -> list[dict[str, Any]]:
        rows = []
        for item in self.scenarios:
            primary = item.get("primary_motion_comparison", {})
            rows.append({
                "scenario": item["scenario"]["scenario_id"],
                "landmark": item["scenario"]["landmark"],
                "frame": item["scenario"]["source_frame"],
                "severity_body_scale": item["scenario"]["magnitude_body_scale"],
                "original_deviation": primary.get("original_robust_deviation"),
                "corrupted_deviation": primary.get("corrupted_robust_deviation"),
                "deviation_amplification": primary.get("deviation_amplification"),
                "rank_percentile": primary.get("corrupted_rank_percentile"),
                "top_1_percent": primary.get("top_percent_presence", {}).get("1"),
                "top_5_percent": primary.get("top_percent_presence", {}).get("5"),
                "top_10_percent": primary.get("top_percent_presence", {}).get("10"),
                "reacquisition_event": bool(item.get("reacquisition", {}).get("corrupted")),
                "relevant_segments": json.dumps(item.get("segment_comparisons", {}), sort_keys=True),
            })
        return rows


def knee_evaluation_matrix(frame: int, severities: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)) -> list[CorruptionScenario]:
    """Return a small deterministic matrix; callers choose a clean source frame."""

    scenarios: list[CorruptionScenario] = []
    for severity in severities:
        label = str(severity).replace(".", "_")
        scenarios.extend([
            CorruptionScenario(f"knee_single_{label}", "knee", frame, "single_frame_offset", severity),
            CorruptionScenario(f"knee_persistent_{label}", "knee", frame, "persistent_offset", severity, "horizontal", 3),
            CorruptionScenario(f"knee_reacquisition_{label}", "knee", frame, "dropout_reacquisition_offset", severity, "horizontal", 3),
        ])
    for gap in (1, 2, 3):
        scenarios.append(CorruptionScenario(f"knee_dropout_{gap}", "knee", frame, "dropout", run_length=gap))
    return scenarios


def _offset(body_scale: float, magnitude: float, direction: Direction) -> tuple[float, float]:
    distance = body_scale * magnitude
    if direction == "horizontal":
        return distance, 0.0
    if direction == "vertical":
        return 0.0, distance
    component = distance / math.sqrt(2.0)
    return component, component


def _observation(calibration: dict[str, Any], frame: int, landmark: str) -> dict[str, Any] | None:
    return next((item for item in calibration["observations"] if item["frame"] == frame and item["landmark"] == landmark), None)


def _percentile(value: float | None, values: list[float]) -> float | None:
    if value is None or not values:
        return None
    return round(100.0 * sum(candidate <= value for candidate in values) / len(values), 3)


def _signal_comparison(original: dict[str, Any] | None, corrupted: dict[str, Any] | None, distribution: list[float]) -> dict[str, Any]:
    original_deviation = original.get("robust_deviation") if original else None
    corrupted_deviation = corrupted.get("robust_deviation") if corrupted else None
    percentile = _percentile(corrupted_deviation, distribution)
    return {
        "original_value": original.get("value") if original else None,
        "original_robust_deviation": original_deviation,
        "corrupted_value": corrupted.get("value") if corrupted else None,
        "corrupted_robust_deviation": corrupted_deviation,
        "deviation_amplification": (
            corrupted_deviation - original_deviation
            if isinstance(original_deviation, (int, float)) and isinstance(corrupted_deviation, (int, float)) else None
        ),
        "corrupted_rank_percentile": percentile,
        "top_percent_presence": {
            "1": percentile is not None and percentile >= 99,
            "5": percentile is not None and percentile >= 95,
            "10": percentile is not None and percentile >= 90,
        },
    }


def _deviation_distribution(
    calibration: dict[str, Any], section: str, key: str | None = None,
    landmark: str | None = None,
) -> list[float]:
    values = []
    for observation in calibration["observations"]:
        if landmark is not None and observation["landmark"] != landmark:
            continue
        evidence = observation.get(section)
        evidence = evidence.get(key) if key is not None and isinstance(evidence, dict) else evidence
        deviation = evidence.get("robust_deviation") if isinstance(evidence, dict) else None
        if isinstance(deviation, (int, float)) and math.isfinite(deviation):
            values.append(float(deviation))
    return values


def _apply_scenario(
    frames: list[dict[str, dict[str, float]]], scenario: CorruptionScenario,
    full_name: str, body_scale: float, session_id: str | None,
) -> tuple[list[dict[str, dict[str, float]]], list[CorruptionRecord], int]:
    changed = copy.deepcopy(frames)
    if scenario.run_length < 1 or scenario.source_frame < 0:
        raise ValueError("run_length must be positive and source_frame must be non-negative")
    if scenario.magnitude_body_scale < 0:
        raise ValueError("magnitude_body_scale must be non-negative")
    if scenario.corruption_type == "single_frame_offset":
        affected = [scenario.source_frame]
        evidence_frame = scenario.source_frame
    elif scenario.corruption_type == "persistent_offset":
        affected = list(range(scenario.source_frame, scenario.source_frame + scenario.run_length))
        evidence_frame = scenario.source_frame
    elif scenario.corruption_type == "dropout":
        affected = list(range(scenario.source_frame, scenario.source_frame + scenario.run_length))
        evidence_frame = scenario.source_frame + scenario.run_length
    else:
        affected = list(range(scenario.source_frame, scenario.source_frame + scenario.run_length))
        evidence_frame = scenario.source_frame + scenario.run_length
    if any(index >= len(changed) for index in affected + [evidence_frame]):
        raise IndexError(f"Scenario {scenario.scenario_id!r} extends beyond the trajectory")

    records = []
    dx, dy = _offset(body_scale, scenario.magnitude_body_scale, scenario.direction)
    for index in affected:
        landmark = changed[index].get(full_name)
        if landmark is None:
            raise ValueError(f"{full_name} is absent at source frame {index}")
        original = (float(landmark["x"]), float(landmark["y"]))
        if scenario.corruption_type in {"dropout", "dropout_reacquisition_offset"}:
            landmark["visibility"] = 0.0
            landmark["presence"] = 0.0
            corrupted_coordinates = None
        else:
            landmark["x"], landmark["y"] = original[0] + dx, original[1] + dy
            corrupted_coordinates = (landmark["x"], landmark["y"])
        records.append(CorruptionRecord(session_id, index, full_name, scenario.corruption_type,
                                        scenario.magnitude_body_scale, scenario.direction, original,
                                        corrupted_coordinates, scenario.random_seed))
    if scenario.corruption_type == "dropout_reacquisition_offset":
        landmark = changed[evidence_frame].get(full_name)
        if landmark is None:
            raise ValueError(f"{full_name} is absent at reacquisition frame {evidence_frame}")
        original = (float(landmark["x"]), float(landmark["y"]))
        landmark["x"], landmark["y"] = original[0] + dx, original[1] + dy
        records.append(CorruptionRecord(session_id, evidence_frame, full_name, scenario.corruption_type,
                                        scenario.magnitude_body_scale, scenario.direction, original,
                                        (landmark["x"], landmark["y"]), scenario.random_seed))
    return changed, records, evidence_frame


def evaluate_pose_corruptions(
    frames: list[dict[str, dict[str, float]]], selected_side: str,
    visibility_threshold: float, scenarios: list[CorruptionScenario],
    video_or_session_id: str | None = None,
) -> CorruptionEvaluationResult:
    """Compare clean control evidence with independently corrupted deep copies."""

    if selected_side not in {"left", "right"}:
        raise ValueError("selected_side must be 'left' or 'right'")
    original = build_kinematic_self_calibration(frames, selected_side, visibility_threshold)
    body_scale = original["profile"]["body_scale"]
    if not isinstance(body_scale, (int, float)) or body_scale <= 0:
        raise ValueError("The clean trajectory has no positive calibration body scale")
    result = CorruptionEvaluationResult(video_or_session_id, selected_side, visibility_threshold, body_scale)

    adjacent = {base: [key for key, pair in PROFILE_SEGMENTS.items() if base in pair]
                for base in ("shoulder", "hip", "knee", "ankle")}
    for scenario in scenarios:
        base = scenario.landmark.removeprefix(f"{selected_side}_")
        if base not in adjacent:
            raise ValueError(f"Unsupported selected-side calibration landmark: {scenario.landmark}")
        full_name = f"{selected_side}_{base}"
        corrupted_frames, records, evidence_frame = _apply_scenario(
            frames, scenario, full_name, body_scale, video_or_session_id
        )
        corrupted = build_kinematic_self_calibration(corrupted_frames, selected_side, visibility_threshold)
        control_observation = _observation(original, evidence_frame, full_name)
        changed_observation = _observation(corrupted, evidence_frame, full_name)
        motion = _signal_comparison(
            control_observation.get("motion") if control_observation else None,
            changed_observation.get("motion") if changed_observation else None,
            _deviation_distribution(corrupted, "motion", landmark=full_name),
        )
        segment_comparisons = {}
        segment_change_comparisons = {}
        for segment in adjacent[base]:
            segment_comparisons[segment] = _signal_comparison(
                control_observation.get("segments", {}).get(segment) if control_observation else None,
                changed_observation.get("segments", {}).get(segment) if changed_observation else None,
                _deviation_distribution(corrupted, "segments", segment),
            )
            segment_change_comparisons[segment] = _signal_comparison(
                control_observation.get("segment_changes", {}).get(segment) if control_observation else None,
                changed_observation.get("segment_changes", {}).get(segment) if changed_observation else None,
                _deviation_distribution(corrupted, "segment_changes", segment),
            )
        def matching_reacquisition_event(calibration):
            return next(
                (
                    item
                    for item in calibration["reacquisition_events"]
                    if item["landmark"] == full_name
                    and item["reacquired_frame"] == evidence_frame
                ),
                None,
            )

        original_event = matching_reacquisition_event(original)
        corrupted_event = matching_reacquisition_event(corrupted)
        result.scenarios.append({
            "scenario": asdict(scenario), "evidence_frame": evidence_frame,
            "corruption_records": [asdict(record) for record in records],
            "primary_motion_comparison": motion,
            "segment_comparisons": segment_comparisons,
            "segment_change_comparisons": segment_change_comparisons,
            "multi_signal_response": {"motion": motion["corrupted_rank_percentile"],
                                      **{f"segment:{key}": value["corrupted_rank_percentile"] for key, value in segment_comparisons.items()},
                                      **{f"segment_change:{key}": value["corrupted_rank_percentile"] for key, value in segment_change_comparisons.items()}},
            "reacquisition": {
                "event_created_by_corruption": original_event is None and corrupted_event is not None,
                "original": original_event,
                "corrupted": corrupted_event,
                "control_continuous_observation_at_reacquired_frame": control_observation,
            },
            "control_profile": original["profile"],
            "corrupted_profile": corrupted["profile"],
        })
    return result


def _load_frames(path: Path) -> tuple[list[dict[str, dict[str, float]]], str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, path.stem
    if isinstance(payload.get("frames"), list):
        return payload["frames"], payload.get("video_or_session_id", path.stem)
    series = payload.get("landmark_time_series")
    if isinstance(series, list):
        return [item["landmarks"] for item in series], payload.get("video_path", path.stem)
    raise ValueError("Input must be a raw frame list or contain frames/landmark_time_series")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline controlled pose-corruption evaluation")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--csv", type=Path, dest="csv_path")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--frame", type=int, required=True, help="Clean knee source frame for the deterministic matrix")
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    args = parser.parse_args()
    frames, session_id = _load_frames(args.input_json)
    evaluation = evaluate_pose_corruptions(frames, args.side, args.visibility_threshold,
                                           knee_evaluation_matrix(args.frame), session_id)
    args.output_json.write_text(json.dumps(evaluation.to_dict(), indent=2), encoding="utf-8")
    if args.csv_path:
        rows = evaluation.compact_rows()
        with args.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
