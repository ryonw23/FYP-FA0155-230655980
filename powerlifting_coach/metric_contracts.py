"""Typed contracts for deterministic, metric-level analysis evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EvidenceStatus(str, Enum):
    """Whether the evidence needed to calculate a metric is available."""

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class LandmarkState(str, Enum):
    """Availability state of a landmark observation used as evidence."""

    OBSERVED = "OBSERVED"
    INTERPOLATED = "INTERPOLATED"
    UNAVAILABLE = "UNAVAILABLE"


class DepthClassification(str, Enum):
    DEPTH_ACHIEVED = "DEPTH_ACHIEVED"
    DEPTH_NOT_ACHIEVED = "DEPTH_NOT_ACHIEVED"
    BORDERLINE = "BORDERLINE"


class EarlyAscentClassification(str, Enum):
    HIP_FIRST_PATTERN_DETECTED = "HIP_FIRST_PATTERN_DETECTED"
    HIP_FIRST_PATTERN_NOT_DETECTED = "HIP_FIRST_PATTERN_NOT_DETECTED"


@dataclass(frozen=True)
class SensitivityEvaluation:
    """Serializable summary of one deterministic one-factor parameter sweep."""

    parameter_name: str
    production_value: float
    evaluated_values: list[float]
    outcomes: list[dict[str, Any]]
    production_classification: str
    alternative_values_evaluated: int
    same_classification_count: int
    different_classification_count: int
    classification_changes: int
    stable_across_evaluated_range: bool
    boundary_crossings: list[dict[str, Any]]
    first_lower_value_where_classification_changes: float | None
    first_upper_value_where_classification_changes: float | None
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricSpecification:
    """Small, declarative description of one deterministic metric."""

    metric_id: str
    lift: str
    required_landmarks: tuple[str, ...]
    required_phase: str
    supported_view: str
    measurement_key: str
    normalisation_method: str | None
    decision_rule: str
    threshold: Any
    threshold_source: dict[str, Any]
    threshold_version: str
    reliability_requirements: dict[str, Any]
    missingness_policy: str
    sensitivity_parameters: dict[str, Any] | None
    evidence_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricEvidence:
    """Serializable measurement, provenance, and decision for one repetition."""

    metric_id: str
    repetition_id: int | None
    phase: str | None
    source_frames: list[int]
    landmarks_used: list[str]
    landmark_quality: dict[str, Any]
    landmark_state: dict[str, LandmarkState]
    landmark_origin: dict[str, str]
    measurement: dict[str, Any] | None
    normalisation: str | None
    decision_rule: str
    threshold: Any
    distance_from_boundary: float | None
    reliability: dict[str, Any]
    sensitivity_result: dict[str, Any] | None
    evidence_status: EvidenceStatus
    classification: str | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
