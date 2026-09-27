"""Generate local-model wording from deterministic lift findings."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .coaching_rule_library import build_actionable_coaching_brief
from .model_assets import LANGUAGE_MODEL_PATH, valid_gguf


@dataclass(frozen=True)
class LocalModelConfig:
    model_path: str = os.getenv("POWERLIFTING_LLM_PATH", str(LANGUAGE_MODEL_PATH))
    context_size: int = int(os.getenv("POWERLIFTING_LLM_CONTEXT", "4096"))
    max_tokens: int = int(os.getenv("POWERLIFTING_LLM_MAX_TOKENS", "700"))

    @property
    def model(self) -> str:
        return Path(self.model_path).name


MAX_LOCAL_MODEL_ATTEMPTS = 2

SYSTEM_MESSAGE = """You are a cautious powerlifting coaching-language assistant.

The supplied source-backed coaching brief is authoritative.
Use only the supplied coaching themes.
Setup-to-bottom torso-angle change is descriptive only. Never turn it into a fault, priority, correction, cue, or claim that the ascent transition needs improvement.
When no corrective theme is supplied, do not claim that every technique check passed or that the lift was perfect.

For each theme:
- describe the observed movement in plain language;
- explain why it is useful to focus on;
- do not generate a new coaching cue, rank, source, caveat, diagnosis, or biomechanical finding.

Only improve these fields: overall_assessment, what_was_observed, why_focus_on_this.
Use the supplied theme_id exactly as written. Do not generate or reword visible titles.
Do not decide whether a technique issue occurred, source IDs, citation references, cue rank, approved next-rep cue, caveats, or whether depth is positive or corrective.

Avoid vague words such as monitor, be aware, maintain, ensure, or focus on your form.

Do not mention weak quads, weak core, fatigue, injury, pain, danger, unsafe, mobility restriction, or competition-valid depth.
Do not claim neutral spine, back or lumbar rounding, hips shooting up, optimal bar path, a correct elbow angle, excessive elbow flare, or a successful, failed, or competition-valid lockout.
Do not echo internal labels such as “hip_first_ascent” or “torso_lean_change”.
Return only valid JSON matching the required schema."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_assessment": {"type": "string"},
        "cue_explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme_id": {"type": "string"},
                    "what_was_observed": {"type": "string"},
                    "why_focus_on_this": {"type": "string"},
                },
                "required": [
                    "theme_id",
                    "what_was_observed",
                    "why_focus_on_this",
                ],
            },
        },
    },
    "required": ["overall_assessment", "cue_explanations"],
}

REVISION_REQUIRED_CODES = {
    "generic_phrase",
    "missing_specific_explanation",
    "missing_why_focus",
    "generic_title",
    "vague_wording",
    "missing_theme",
    "incorrect_rank",
    "missing_source_ids",
    "missing_next_rep_cue",
    "too_many_cues",
    "duplicate_cue",
}
HIGH_RISK_REVISION_CODES = {
    "unsupported_causal_claim",
    "medical_or_injury_claim",
    "fatigue_or_weakness_claim",
    "unsafe_or_dangerous_claim",
    "invented_biomechanics",
    "depth_positive_as_correction",
    "forbidden_internal_identifier",
    "unsupported_torso_correction",
    "unsupported_all_clear_claim",
    "unsupported_no_theme_coaching",
}
IMMEDIATE_FALLBACK_CODES = {
    "invalid_json",
    "missing_required_structure",
    "unparseable_response",
    "local_model_unavailable",
    "local_model_timeout",
    "local_model_error",
}

FORBIDDEN_GENERIC_PHRASES = (
    "monitor torso lean", "be aware of", "maintain hip upward progress",
    "ensure sufficient depth", "focus on your form",
)
UNSUPPORTED_CLAIMS = (
    "injury", "pain", "medical", "diagnos", "fatigue", "fatigued", "weak", "weakness",
    "dangerous", "unsafe", "failed", "bad form", "good morning", "collapse",
    "mobility limitation", "competition valid", "competition-valid", "red light",
    "poor mobility", "neutral spine", "back rounding", "lumbar rounding", "hips shot up",
    "optimal bar path", "correct elbow angle", "too much elbow flare", "successful lockout",
    "failed lockout", "competition-valid lockout",
)
INTERNAL_LABELS = (
    "hip_first_ascent", "torso_lean_change", "prolonged_ascent_or_sticking_region",
    "bench_elbow_flexion_2d", "bench_wrist_trajectory_2d", "bench_finish_geometry_similarity",
    "deadlift_start_geometry_2d", "deadlift_phase_extension_2d", "deadlift_finish_geometry_2d",
    "deadlift_early_coordination_2d",
)

FORBIDDEN_KEYS = {
    "frames",
    "landmarks",
    "raw_landmarks",
    "pose_landmarks",
    "pose_keypoints",
    "keypoints",
    "smoothed_hip_y",
    "video_path",
    "annotated_video_path",
    "annotated_path",
    "debug_json",
    "segmentation_debug",
}


def _safe_reason(reason: str | None) -> str | None:
    allowed = {
        "analysis_invalid",
        "rep_count_not_one",
        "missing_rep",
        "local_model_unavailable",
        "model_missing",
        "invalid_llm_response",
        "timeout",
        "local_model_request_failed",
    }
    return reason if reason in allowed else None


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items() if k not in FORBIDDEN_KEYS}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _print_fallback(
    reason_code: str, detail: str | None = None, status: int | None = None
) -> None:
    line = f"[local-model] fallback_reason={reason_code}"
    if status is not None:
        line += f" status={status}"
    if detail:
        safe_detail = " ".join(str(detail).split())[:120]
        line += f" detail={safe_detail}"
    print(line)


def _metadata(
    config: LocalModelConfig,
    mode: str,
    reason_code: str | None = None,
    reason_detail: str | None = None,
    completion_status: int | None = None,
    *,
    attempt_count: int = 0,
    revision_attempted: bool = False,
    initial_quality: dict | None = None,
    final_quality: dict | None = None,
    initial_parsed_response: dict | None = None,
    revised_parsed_response: dict | None = None,
) -> dict:
    initial_quality = initial_quality or _quality_result(True)
    final_quality = final_quality or _quality_result(True)
    return {
        "generation_mode": mode,
        "local_model_attempt_count": attempt_count,
        "revision_attempted": revision_attempted,
        "initial_quality_failures": initial_quality.get("failure_codes", []),
        "initial_quality_failure_details": initial_quality.get("details", []),
        "final_quality_failures": final_quality.get("failure_codes", []),
        "final_quality_failure_details": final_quality.get("details", []),
        "initial_parsed_response": initial_parsed_response,
        "revised_parsed_response": revised_parsed_response,
        "fallback_reason_code": reason_code,
        "fallback_reason_detail": reason_detail,
        "local_model_status": completion_status,
        "local_model": config.model,
    }


def _status_for_reason(reason_code: str) -> str:
    if reason_code == "local_model_unavailable":
        return "Using deterministic feedback: local language model unavailable"
    if reason_code == "local_model_missing":
        return (
            "Using deterministic feedback: local language model is missing or invalid; "
            "run python -m powerlifting_coach.setup"
        )
    if reason_code in {
        "local_model_response_quality_rejected",
        "local_model_unsafe_after_revision",
    }:
        return (
            "Using deterministic feedback: local response could not be safely refined"
        )
    if reason_code in {
        "local_model_invalid_json",
        "local_model_schema_validation_failed",
        "local_model_error",
        "local_model_timeout",
    }:
        return "Using deterministic feedback: local response did not meet the required format"
    return "Using deterministic feedback: local response did not meet the required format"


_MODEL_INSTANCE: Any = None
_MODEL_INSTANCE_PATH: str | None = None


def _load_local_model(config: LocalModelConfig) -> Any:
    """Lazily load an already-installed model without making network requests."""

    global _MODEL_INSTANCE, _MODEL_INSTANCE_PATH
    path = Path(config.model_path)
    if not valid_gguf(path):
        raise FileNotFoundError(
            f"Local language model is missing or invalid at {path}. "
            "Run python -m powerlifting_coach.setup before launching the app."
        )
    if _MODEL_INSTANCE is None or _MODEL_INSTANCE_PATH != str(path.resolve()):
        from llama_cpp import Llama

        _MODEL_INSTANCE = Llama(
            model_path=str(path),
            n_ctx=config.context_size,
            n_gpu_layers=-1,
            verbose=False,
        )
        _MODEL_INSTANCE_PATH = str(path.resolve())
    return _MODEL_INSTANCE


def check_local_model_status(config: LocalModelConfig) -> dict:
    path = Path(config.model_path)
    if not valid_gguf(path):
        return {
            "reachable": False,
            "configured_model_installed": False,
            "error": "model_missing",
            "detail": (
                f"Local language model is missing or invalid at {path}. "
                "Run python -m powerlifting_coach.setup before launching the app."
            ),
        }
    try:
        _load_local_model(config)
        return {"reachable": True, "configured_model_installed": True, "error": None}
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {
            "reachable": False,
            "configured_model_installed": True,
            "error": "unavailable",
            "detail": str(exc),
        }


def _valid_rep(analysis: dict) -> dict | None:
    rep = analysis.get("rep")
    if isinstance(rep, dict) and rep:
        return rep
    reps = analysis.get("reps")
    if isinstance(reps, list) and reps and isinstance(reps[0], dict):
        return reps[0]
    if analysis.get("lift") in {"bench_press", "deadlift"}:
        return analysis
    return None


def _finding_names(rep: dict) -> set[str]:
    return {str(f.get("name")) for f in rep.get("findings", []) if isinstance(f, dict)}


def _finding_confidence(rep: dict, names: set[str], default: str = "medium") -> str:
    levels = {"low": 0, "medium": 1, "high": 2}
    found = [
        str(f.get("confidence", default))
        for f in rep.get("findings", [])
        if isinstance(f, dict) and f.get("name") in names
    ]
    found = [x for x in found if x in levels]
    return min(found, key=lambda x: levels[x]) if found else default


def _depth_status(analysis: dict, rep: dict) -> str:
    for f in rep.get("findings", []):
        if isinstance(f, dict) and f.get("name") == "depth":
            return str(f.get("status", ""))
    return str(rep.get("depth_assessment", analysis.get("depth_assessment", "")))


def build_feedback_payload(analysis: dict) -> dict:
    rep = _valid_rep(analysis) or {}
    metric_results = rep.get("metric_results", analysis.get("metric_results", []))
    if analysis.get("lift") == "bench_press":
        metric_results = [
            metric
            for metric in metric_results
            if not isinstance(metric, dict)
            or metric.get("metric_id") != "motionless_bottom_2d_proxy"
        ]
    payload = {
        "quality": _sanitize(
            {
                "analysis_valid": analysis.get("quality", {}).get("analysis_valid"),
                "confidence": analysis.get("quality", {}).get(
                    "confidence", analysis.get("confidence")
                ),
            }
        ),
        "source_backed_coaching_brief": build_actionable_coaching_brief(analysis),
        "actionable_coaching_brief": build_actionable_coaching_brief(analysis),
        "warnings": [
            w
            for w in (analysis.get("warnings", []) + rep.get("warnings", []))
            if isinstance(w, str)
        ],
        "metric_availability": [
            {
                "metric_id": metric.get("metric_id"),
                "evidence_status": metric.get("evidence_status"),
                "classification": metric.get("classification"),
                "warnings": metric.get("warnings", []),
            }
            for metric in metric_results
            if isinstance(metric, dict)
        ],
        "descriptive_movement_angles": _sanitize(
            analysis.get("measurements", {}).get("movement_angle_summary", {})
        ),
    }
    assert not _contains_key(payload, "frames")
    return payload


def build_template_feedback(analysis: dict, reason: str | None = None) -> dict:
    reason = _safe_reason(reason)
    quality = analysis.get("quality", {}) if isinstance(analysis, dict) else {}
    confidence = quality.get("confidence", analysis.get("confidence", "unknown"))
    rep = _valid_rep(analysis) or {}
    metric_results = rep.get("metric_results", analysis.get("metric_results", []))
    if analysis.get("lift") == "bench_press":
        metric_results = [
            metric
            for metric in metric_results
            if not isinstance(metric, dict)
            or metric.get("metric_id") != "motionless_bottom_2d_proxy"
        ]
    warnings = []
    for warning in analysis.get("warnings", []) if isinstance(analysis, dict) else []:
        if isinstance(warning, str):
            warnings.append(warning)
    for warning in rep.get("warnings", []):
        if isinstance(warning, str) and warning not in warnings:
            warnings.append(warning)
    for metric in metric_results:
        if not isinstance(metric, dict) or not (
            metric.get("evidence_status") == "INSUFFICIENT"
            or metric.get("classification") == "BORDERLINE"
        ):
            continue
        label = str(metric.get("metric_id", "Metric")).replace("_", " ")
        limitation = (
            f"{label} was withheld because its required evidence was unavailable."
            if metric.get("evidence_status") == "INSUFFICIENT"
            else f"{label} was borderline and should not be treated as a confident finding."
        )
        if limitation not in warnings:
            warnings.append(limitation)

    invalid = quality.get("analysis_valid") is False or reason in {
        "analysis_invalid",
        "rep_count_not_one",
        "missing_rep",
    }
    if invalid:
        return {
            "overall_assessment": "Tracking or rep detection was insufficient for reliable technique coaching. Feedback was not generated from unreliable analysis.",
            "priority_cues": [],
            "positive_observations": [],
            "neutral_context": [],
            "limitations": warnings or ["Analysis was not valid."],
        }

    brief = build_actionable_coaching_brief(analysis)
    limitations = list(dict.fromkeys(warnings + brief.get("limitations", [])))
    if confidence in {"medium", "low"} and not any(
        "2D video-based estimate" in x for x in limitations
    ):
        limitations.append(
            "This is a 2D video-based estimate and not a competition judging decision."
        )
    overall = brief["overall_summary"]
    return {
        "overall_assessment": overall,
        "priority_cues": [
            {
                "rank": t["rank"],
                "title": t["title"],
                "what_was_observed": t["what_was_observed"].replace(
                    "The hips", "Your hips"
                ),
                "why_focus_on_this": t["why_focus_on_this"],
                "next_rep_cue": t["next_rep_cue"],
                "practice_task": t.get("practice_task", ""),
                "success_check": t.get("success_check", ""),
                "confidence": t["confidence"],
            }
            | {
                "source_ids": t.get("source_ids", []),
                "evidence_and_limits": {
                    "detected_pattern": t.get("what_was_observed"),
                    "interpretation": t.get("why_focus_on_this"),
                    "references": t.get("reference_notes", []),
                    "important_limits": t.get("caveats", []),
                },
            }
            for t in brief.get("priority_themes", [])
        ],
        "positive_observations": [
            p["text"] for p in brief.get("positive_observations", [])
        ],
        "neutral_context": brief.get("neutral_context", []),
        "limitations": limitations,
    }


def _sentence_count(text: str) -> int:
    return sum(
        1
        for part in text.replace("!", ".").replace("?", ".").split(".")
        if part.strip()
    )


def _quality_result(
    accepted: bool, codes: list[str] | None = None, details: list[str] | None = None
) -> dict:
    codes = list(dict.fromkeys(codes or []))
    return {
        "accepted": accepted,
        "failure_codes": codes,
        "details": details or [],
        "severity": "accepted" if accepted else "revision_required",
    }


def _classify_bad_phrase(text: str) -> tuple[str, str] | None:
    lower = text.lower()
    torso_corrections = (
        "hold your torso position",
        "torso lean needs",
        "torso angle needs",
        "correct your torso",
        "improve your torso",
        "stay tall as i stand",
        "stay tall as you stand",
        "ascent transition needs improvement",
        "improve the ascent transition",
    )
    if any(phrase in lower for phrase in torso_corrections) or (
        "torso" in lower and ("lean" in lower or "forward" in lower)
    ):
        return (
            "unsupported_torso_correction",
            "Turns descriptive torso-angle change into corrective coaching.",
        )
    if any(
        phrase in lower
        for phrase in ("all technique checks passed", "every technique check passed", "perfect lift", "perfect technique")
    ):
        return (
            "unsupported_all_clear_claim",
            "Claims broader success than the implemented checks establish.",
        )
    for phrase in FORBIDDEN_GENERIC_PHRASES:
        if phrase in lower:
            return "generic_phrase", f"Uses vague wording: '{phrase}'."
    for word in ("monitor", "be aware", "maintain", "ensure", "focus on your form"):
        if word in lower:
            return "vague_wording", f"Uses vague wording: '{word}'."
    if any(term in lower for term in ("weak quad", "weak core", "weakness", "weak ")):
        return "fatigue_or_weakness_claim", "Introduces an unsupported weakness claim."
    if any(term in lower for term in ("fatigue", "fatigued")):
        return "fatigue_or_weakness_claim", "Introduces an unsupported fatigue claim."
    if any(term in lower for term in ("injury", "pain", "medical", "diagnos")):
        return "medical_or_injury_claim", "Introduces a medical, pain, or injury claim."
    if any(term in lower for term in ("danger", "dangerous", "unsafe", "injury risk")):
        return (
            "unsafe_or_dangerous_claim",
            "Introduces an unsupported safety or danger claim.",
        )
    if any(
        term in lower
        for term in (
            "mobility restriction",
            "poor mobility",
            "mobility is limiting",
            "mobility limitation",
        )
    ):
        return (
            "unsupported_causal_claim",
            "Introduces an unsupported mobility or cause claim.",
        )
    if any(
        term in lower
        for term in (
            "knees collapsed",
            "knee collapse",
            "good morning",
            "bad form",
            "neutral spine",
            "back rounding",
            "lumbar rounding",
            "hips shot up",
            "optimal bar path",
            "correct elbow angle",
            "too much elbow flare",
            "successful lockout",
            "failed lockout",
            "competition-valid lockout",
        )
    ):
        return "invented_biomechanics", "Introduces an unsupported technique fault."
    if any(
        term in lower
        for term in ("competition valid", "competition-valid", "red light")
    ):
        return (
            "depth_positive_as_correction",
            "Frames depth as a competition judging outcome.",
        )
    for label in INTERNAL_LABELS:
        if label in text:
            return (
                "forbidden_internal_identifier",
                "Mentions an internal rule label or raw technical identifier.",
            )
    return None


def normalise_text(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def _is_missing_or_generic_explanation(text: Any) -> bool:
    if not isinstance(text, str) or not text.strip():
        return True
    return normalise_text(text).rstrip(".!?") in {
        "this is important",
        "this matters",
        "this is worth focusing on",
        "the movement was observed",
        "there was an issue",
    }


def _validate_feedback(data: Any, payload: dict) -> tuple[dict | None, dict]:
    if not isinstance(data, dict):
        return None, _quality_result(
            False, ["missing_required_structure"], ["Response is not a JSON object."]
        )
    brief = payload.get("actionable_coaching_brief", {})
    themes = brief.get("priority_themes", []) if isinstance(brief, dict) else []
    explanations = data.get("cue_explanations")
    # Accept legacy priority_cues for older tests, but only read prose from it.
    if explanations is None and isinstance(data.get("priority_cues"), list):
        explanations = [
            {
                "theme_id": c.get("theme_id"),
                "theme_title": c.get("theme_title", c.get("title")),
                "what_was_observed": c.get("what_was_observed"),
                "why_focus_on_this": c.get("why_focus_on_this"),
            }
            for c in data.get("priority_cues", [])
            if isinstance(c, dict)
        ]
    codes: list[str] = []
    details: list[str] = []
    overall = data.get("overall_assessment")
    if not isinstance(overall, str) or not overall.strip():
        codes.append("missing_specific_explanation")
        details.append("Overall assessment is missing or empty.")
    elif _sentence_count(overall) > 3:
        codes.append("missing_specific_explanation")
        details.append(
            "Overall assessment is too long to be concise coaching feedback."
        )
    else:
        bad = _classify_bad_phrase(overall)
        if bad and bad[0] != "vague_wording":
            codes.append(bad[0])
            details.append(f"Overall assessment {bad[1]}")
        elif not themes and normalise_text(overall) != normalise_text(
            str(brief.get("overall_summary", ""))
        ):
            codes.append("unsupported_no_theme_coaching")
            details.append(
                "Overall assessment adds coaching when no corrective theme was supplied."
            )
    if not isinstance(explanations, list):
        codes.append("missing_required_structure")
        details.append("cue_explanations must be an array.")
        return None, _quality_result(False, codes, details)
    if len(explanations) > len(themes):
        codes.append("too_many_cues")
        details.append("Response includes more cue explanations than supplied themes.")
    if len(explanations) < len(themes):
        codes.append("missing_theme")
        details.append(
            "Response does not include an explanation for every supplied coaching theme."
        )
    repair_notes: list[str] = []
    themes_by_id = {
        t["theme_id"]: t
        for t in themes
        if isinstance(t, dict) and isinstance(t.get("theme_id"), str)
    }
    themes_by_title = {
        normalise_text(t["title"]): t
        for t in themes
        if isinstance(t, dict) and isinstance(t.get("title"), str)
    }
    seen: set[str] = set()
    prose_by_id: dict[str, dict] = {}
    for i, item in enumerate(explanations):
        if not isinstance(item, dict):
            codes.append("missing_required_structure")
            details.append(f"Cue explanation {i + 1} is not an object.")
            continue
        theme = None
        identifier = item.get("theme_id")
        legacy_title = item.get("theme_title")
        if isinstance(identifier, str) and identifier in themes_by_id:
            theme = themes_by_id[identifier]
        elif isinstance(legacy_title, str) and legacy_title in themes_by_id:
            theme = themes_by_id[legacy_title]
            repair_notes.append("legacy_theme_title_internal_id_repaired")
        elif (
            isinstance(legacy_title, str)
            and normalise_text(legacy_title) in themes_by_title
        ):
            theme = themes_by_title[normalise_text(legacy_title)]
            repair_notes.append("legacy_theme_title_visible_title_repaired")
        elif len(themes) == 1 and len(explanations) == 1:
            theme = themes[0]
            repair_notes.append("single_theme_association_repaired")
        if theme is None:
            codes.append("missing_theme")
            details.append(
                f"Cue explanation {i + 1} does not match a supplied coaching theme."
            )
            continue
        theme_id = theme["theme_id"]
        title = theme["title"]
        if theme_id in seen:
            codes.append("duplicate_cue")
            details.append(f"Theme '{title}' appears more than once.")
        seen.add(theme_id)
        what = item.get("what_was_observed")
        why = item.get("why_focus_on_this")
        if _is_missing_or_generic_explanation(what):
            codes.append("missing_specific_explanation")
            details.append(
                f"Cue '{title}' does not describe the observed movement in plain language."
            )
        if _is_missing_or_generic_explanation(why):
            codes.append("missing_why_focus")
            details.append(
                f"Cue '{title}' does not explain why the movement is worth focusing on."
            )
        for field_name, text in (
            ("what_was_observed", what),
            ("why_focus_on_this", why),
        ):
            if isinstance(text, str):
                bad = _classify_bad_phrase(text)
                if bad:
                    codes.append(bad[0])
                    details.append(f"Cue '{title}' {field_name} {bad[1]}")
        prose_by_id[theme_id] = {
            "what_was_observed": str(what).strip(),
            "why_focus_on_this": str(why).strip(),
        }
    if codes:
        return None, _quality_result(False, codes, details)
    return _merge_llm_prose_with_brief(
        str(overall).strip(), prose_by_id, brief
    ), _quality_result(True, details=repair_notes)


def _merge_llm_prose_with_brief(
    overall: str, prose_by_id: dict[str, dict], brief: dict
) -> dict:
    limitations = list(brief.get("limitations", []))
    if not any("2D video-based estimate" in x for x in limitations):
        limitations.append(
            "This is a 2D video-based estimate and not a competition judging decision."
        )
    # A short free-form summary cannot be reliably matched to an individual
    # theme. Keep the deterministic summary so accepted prose cannot add a
    # movement finding that was not supplied by the analysis.
    overall = brief["overall_summary"]
    return {
        "overall_assessment": overall,
        "priority_cues": [
            {
                "rank": t["rank"],
                "title": t["title"],
                "what_was_observed": (
                    t["what_was_observed"]
                    if str(t.get("theme_id", "")).startswith(("DL-", "BP-"))
                    else prose_by_id.get(t.get("theme_id"), {}).get(
                        "what_was_observed", t["what_was_observed"]
                    )
                ),
                "why_focus_on_this": prose_by_id.get(t.get("theme_id"), {}).get(
                    "why_focus_on_this", t["why_focus_on_this"]
                ),
                "next_rep_cue": t["next_rep_cue"],
                "practice_task": t.get("practice_task", ""),
                "success_check": t.get("success_check", ""),
                "confidence": t["confidence"],
                "source_ids": t.get("source_ids", []),
                "evidence_and_limits": {
                    "detected_pattern": t.get("what_was_observed"),
                    "interpretation": t.get("why_focus_on_this"),
                    "references": t.get("reference_notes", []),
                    "important_limits": t.get("caveats", []),
                },
            }
            for t in brief.get("priority_themes", [])
        ],
        "positive_observations": [
            p["text"] for p in brief.get("positive_observations", [])
        ],
        "neutral_context": brief.get("neutral_context", []),
        "limitations": limitations[:8],
    }


def build_revision_prompt(
    coaching_brief: dict,
    previous_response: dict,
    quality_result: dict,
) -> str:
    themes = coaching_brief.get("priority_themes", [])
    theme_ids = [
        theme["theme_id"]
        for theme in themes
        if isinstance(theme, dict) and isinstance(theme.get("theme_id"), str)
    ]
    problems = (
        "\n".join(
            f"- {code}: {detail}"
            for code, detail in zip(
                quality_result.get("failure_codes", []),
                quality_result.get("details", []),
            )
        )
        or "- coaching_quality_validation_failed: The response did not meet the quality requirements."
    )
    return (
        "Your previous coaching response was not accepted because it did not meet the project's coaching-quality requirements.\n\n"
        "Rewrite the complete response using only the supplied coaching brief.\n\n"
        f"Supplied source-backed coaching brief:\n{json.dumps(coaching_brief, sort_keys=True)}\n\n"
        f"Previous parsed JSON response:\n{json.dumps(previous_response, sort_keys=True)}\n\n"
        f"Problems to correct:\n{problems}\n\n"
        "Requirements:\n"
        f"- cue_explanations must contain exactly {len(theme_ids)} item(s), one for each priority_themes entry, using only these theme_id values: {json.dumps(theme_ids)}.\n"
        "- positive_observations and neutral_context are not coaching themes: do not turn them into cue_explanations. If the allowed theme_id list is empty, return cue_explanations as [].\n"
        "- Explain the movement in plain language that a lifter can recognise.\n"
        "- Explain why the movement is worth focusing on.\n"
        "- The overall_assessment may only paraphrase the supplied overall_summary; do not add another movement finding.\n"
        "- Do not use vague wording such as “monitor”, “be aware”, “maintain”, “ensure”, or “focus on your form”.\n"
        "- Do not introduce any diagnosis, fatigue claim, weakness claim, injury claim, safety claim, or unsupported technique fault.\n"
        "- Do not mention internal rule labels, source IDs, or raw technical identifiers.\n"
        "- Do not generate a next-rep cue; it is provided separately by the application.\n"
        "- Return only valid JSON matching the required schema."
    )


def _chat_request(
    payload: dict, config: LocalModelConfig, revision_prompt: str | None = None
) -> dict:
    coaching_brief = payload.get(
        "source_backed_coaching_brief", payload.get("actionable_coaching_brief", {})
    )
    themes = coaching_brief.get("priority_themes", [])
    theme_ids = [
        theme["theme_id"]
        for theme in themes
        if isinstance(theme, dict) and isinstance(theme.get("theme_id"), str)
    ]
    user = revision_prompt or json.dumps(
        {
            "instructions": (
                "Return only JSON matching the supplied schema. Use only the supplied coaching themes. "
                f"Return exactly {len(theme_ids)} cue_explanations, one for each priority_themes entry, and use only these theme_id values: {json.dumps(theme_ids)}. "
                "Positive_observations and neutral_context are not coaching themes and must not become cue_explanations. If the theme_id list is empty, return cue_explanations as an empty array. "
                "For each theme, describe the observed movement in plain language and explain why it is useful to focus on. "
                "Use the supplied theme_id exactly as written; do not substitute the visible title for the ID or omit the ID. "
                "Do not generate a new coaching cue, rank, source, caveat, diagnosis, or biomechanical finding. "
                "Setup-to-bottom torso-angle change is descriptive only and must not become corrective advice or an ascent-transition claim. "
                "If no coaching theme is supplied, do not imply that every technique check passed or that the lift was perfect. "
                "Avoid vague words such as monitor, be aware, maintain, ensure, and focus on your form. "
                "Do not mention weak quads, weak core, fatigue, injury, pain, danger, unsafe, mobility restriction, or competition-valid depth."
            ),
            "expected_response_schema": RESPONSE_SCHEMA,
            "source_backed_coaching_brief": coaching_brief,
            "analysis_confidence": payload.get("quality", {}).get("confidence"),
        },
        sort_keys=True,
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user},
        ],
    }


def _run_chat_completion(chat_body: dict, config: LocalModelConfig) -> tuple[int, dict]:
    model = _load_local_model(config)
    result = model.create_chat_completion(
        messages=chat_body["messages"],
        temperature=0,
        max_tokens=config.max_tokens,
        response_format={"type": "json_object", "schema": RESPONSE_SCHEMA},
    )
    content = result["choices"][0]["message"]["content"]
    return 200, {"message": {"content": content}}


def test_local_model(config: LocalModelConfig) -> dict:
    try:
        model = _load_local_model(config)
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": "Reply with exactly: health check passed"}],
            temperature=0,
            max_tokens=12,
        )
        content = response["choices"][0]["message"]["content"].strip()
        return {
            "ok": "health check passed" in content.lower(),
            "configured_model": config.model,
            "model_path": config.model_path,
            "response_content": content,
        }
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {
            "ok": False,
            "configured_model": config.model,
            "model_path": config.model_path,
            "error": "unreachable",
            "error_detail": str(exc),
        }
    except (KeyError, TypeError):
        return {
            "ok": False,
            "configured_model": config.model,
            "model_path": config.model_path,
            "error": "invalid_response",
        }


def _validate_response_schema(data: Any) -> str | None:
    if not isinstance(data, dict):
        return "top_level_not_object"
    if not isinstance(data.get("overall_assessment"), str):
        return "overall_assessment_not_string"
    explanations = data.get("cue_explanations")
    if explanations is None and isinstance(data.get("priority_cues"), list):
        explanations = data.get("priority_cues")
    if not isinstance(explanations, list):
        return "missing_priority_cues"
    for item in explanations:
        if not isinstance(item, dict):
            return "cue_explanation_not_object"
        if "cue_explanations" in data:
            for key in ("what_was_observed", "why_focus_on_this"):
                if key not in item:
                    return "cue_explanation_missing_" + key
        else:
            for key in ("title", "what_was_observed", "why_focus_on_this"):
                if key not in item:
                    return "priority_cue_missing_" + key
    return None


def generate_local_feedback(
    analysis: dict, config: LocalModelConfig | None = None
) -> dict:
    config = config or LocalModelConfig()

    def fallback(
        reason_code: str,
        template_reason: str,
        detail: str | None = None,
        completion_status: int | None = None,
        status_text: str | None = None,
        *,
        attempt_count: int = 0,
        revision_attempted: bool = False,
        initial_quality: dict | None = None,
        final_quality: dict | None = None,
        initial_parsed_response: dict | None = None,
        revised_parsed_response: dict | None = None,
    ) -> dict:
        _print_fallback(
            reason_code,
            detail=detail,
            status=completion_status if reason_code == "local_model_error" else None,
        )
        feedback = build_template_feedback(analysis, template_reason)
        return {
            "status": status_text or _status_for_reason(reason_code),
            "source": "fallback",
            "feedback": feedback,
        } | _metadata(
            config,
            "fallback",
            reason_code,
            detail,
            completion_status,
            attempt_count=attempt_count,
            revision_attempted=revision_attempted,
            initial_quality=initial_quality,
            final_quality=final_quality,
            initial_parsed_response=initial_parsed_response,
            revised_parsed_response=revised_parsed_response,
        )

    quality = analysis.get("quality", {}) if isinstance(analysis, dict) else {}
    descriptive_lift = analysis.get("lift") in {"bench_press", "deadlift"}
    analysis_valid = (
        analysis.get("analysis_status") in {"valid", "warning"}
        if descriptive_lift
        else quality.get("analysis_valid") is True
    )
    if not analysis_valid:
        feedback = build_template_feedback(analysis, "analysis_invalid")
        return {
            "status": "Using deterministic feedback: analysis was not valid",
            "source": "fallback",
            "feedback": feedback,
        } | _metadata(config, "fallback", "analysis_invalid")
    if not descriptive_lift and analysis.get("reps_detected") != 1:
        feedback = build_template_feedback(analysis, "rep_count_not_one")
        return {
            "status": "Using deterministic feedback: analysis was not valid",
            "source": "fallback",
            "feedback": feedback,
        } | _metadata(config, "fallback", "rep_count_not_one")
    if _valid_rep(analysis) is None:
        feedback = build_template_feedback(analysis, "missing_rep")
        return {
            "status": "Using deterministic feedback: analysis was not valid",
            "source": "fallback",
            "feedback": feedback,
        } | _metadata(config, "fallback", "missing_rep")

    status = check_local_model_status(config)
    if not status.get("configured_model_installed"):
        return fallback(
            "local_model_missing",
            "model_missing",
            status.get("detail"),
        )
    if not status.get("reachable"):
        code = (
            "local_model_timeout"
            if status.get("error") == "timeout"
            else "local_model_unavailable"
        )
        return fallback(
            code,
            "timeout" if code == "local_model_timeout" else "local_model_unavailable",
            status.get("detail"),
        )
    payload = build_feedback_payload(analysis)
    completion_status = None
    initial_quality = _quality_result(True)
    final_quality = _quality_result(True)
    initial_parsed = None
    revised_parsed = None
    try:
        for attempt in range(1, MAX_LOCAL_MODEL_ATTEMPTS + 1):
            revision_prompt = None
            if attempt == 2:
                revision_prompt = build_revision_prompt(
                    payload.get(
                        "source_backed_coaching_brief",
                        payload.get("actionable_coaching_brief", {}),
                    ),
                    initial_parsed or {},
                    initial_quality,
                )
            completion_status, response_json = _run_chat_completion(
                _chat_request(payload, config, revision_prompt), config
            )
            generated_content = response_json["message"]["content"]
            parsed = json.loads(generated_content)
            if attempt == 1:
                initial_parsed = parsed
            else:
                revised_parsed = parsed
            schema_error = _validate_response_schema(parsed)
            if schema_error:
                return fallback(
                    "local_model_schema_validation_failed",
                    "invalid_llm_response",
                    schema_error,
                    completion_status,
                    attempt_count=attempt,
                    revision_attempted=attempt > 1,
                    initial_quality=initial_quality,
                    final_quality=final_quality,
                    initial_parsed_response=initial_parsed,
                    revised_parsed_response=revised_parsed,
                )
            validated, quality_result = _validate_feedback(parsed, payload)
            if attempt == 1:
                initial_quality = quality_result
            final_quality = quality_result
            if validated is not None and quality_result.get("accepted"):
                locally_repaired = bool(quality_result.get("details"))
                return {
                    "status": (
                        "Generated with the local language model"
                        if attempt == 1 and not locally_repaired
                        else "Generated with the local language model, with deterministic coaching safeguards"
                    ),
                    "source": "local_model",
                    "feedback": validated,
                } | _metadata(
                    config,
                    "local_model",
                    None,
                    None,
                    completion_status,
                    attempt_count=attempt,
                    revision_attempted=attempt > 1,
                    initial_quality=(
                        initial_quality if attempt > 1 else _quality_result(True)
                    ),
                    final_quality=quality_result,
                    initial_parsed_response=initial_parsed,
                    revised_parsed_response=revised_parsed,
                )
            if attempt >= MAX_LOCAL_MODEL_ATTEMPTS:
                return fallback(
                    "local_model_response_quality_rejected",
                    "invalid_llm_response",
                    "coaching_quality_validation_failed",
                    completion_status,
                    attempt_count=attempt,
                    revision_attempted=True,
                    initial_quality=initial_quality,
                    final_quality=final_quality,
                    initial_parsed_response=initial_parsed,
                    revised_parsed_response=revised_parsed,
                )
        return fallback("local_model_response_quality_rejected", "invalid_llm_response")
    except TimeoutError:
        return fallback("local_model_timeout", "timeout")
    except json.JSONDecodeError as exc:
        return fallback(
            "local_model_invalid_json",
            "invalid_llm_response",
            exc.msg,
            completion_status,
            attempt_count=1,
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        return fallback("local_model_unavailable", "local_model_unavailable")
    except (KeyError, TypeError):
        return fallback(
            "local_model_schema_validation_failed",
            "invalid_llm_response",
            "missing_message_content",
            completion_status,
            attempt_count=1,
        )
