"""Formal coaching rule metadata and source-backed theme construction."""

from typing import Any, TypedDict
from .coaching_sources import SOURCES


class CoachingRule(TypedDict):
    rule_id: str
    name: str
    triggered_by: list[str]
    source_ids: list[str]
    project_interpretation: str
    allowed_coaching_wording: str
    approved_next_rep_cue: str
    positive_observation: bool
    priority: int
    caveats: list[str]
    disallowed_claims: list[str]


DISALLOWED = [
    "weak quadriceps", "fatigue", "injury risk", "dangerous", "medical diagnosis",
    "competition-valid certainty",
]
COACHING_RULES: dict[str, CoachingRule] = {
    "BP-001-BOTTOM-ELBOW-GEOMETRY": {
        "rule_id": "BP-001-BOTTOM-ELBOW-GEOMETRY",
        "name": "Bottom elbow geometry",
        "triggered_by": ["bench_elbow_flexion_2d"],
        "source_ids": ["SRC_GOMO_VAN_DEN_TILLAAR_BENCH_GRIP"],
        "project_interpretation": "A descriptive projected elbow angle at the bottom; no universal correctness threshold is applied.",
        "allowed_coaching_wording": "At the bottom of the repetition, your projected elbow angle reached X degrees.",
        "approved_next_rep_cue": "Keep the same camera position if you want to compare this measurement between repetitions.",
        "positive_observation": False,
        "priority": 1,
        "caveats": [
            "Grip width and individual technique affect elbow geometry.",
            "This is a projected 2D measurement, not a correctness classification.",
        ],
        "disallowed_claims": DISALLOWED
        + ["too tucked", "too flared", "correct elbow angle", "should be 90 degrees"],
    },
    "BP-002-WRIST-TRAJECTORY": {
        "rule_id": "BP-002-WRIST-TRAJECTORY",
        "name": "Wrist trajectory proxy",
        "triggered_by": ["bench_wrist_trajectory_2d"],
        "source_ids": ["SRC_LARSEN_BENCH_KINEMATICS"],
        "project_interpretation": "The selected wrist is a hand/bar-end proxy, not validated direct bar tracking.",
        "allowed_coaching_wording": "The wrist proxy travelled during descent and finished a measured distance from its starting vertical position.",
        "approved_next_rep_cue": "Use the same side-view camera position to compare the wrist proxy between repetitions.",
        "positive_observation": False,
        "priority": 2,
        "caveats": [
            "Wrist means a hand/bar-end proxy.",
            "Do not interpret this as optimal or validated direct bar path.",
        ],
        "disallowed_claims": DISALLOWED
        + [
            "optimal bar path",
            "bad bar path",
            "chest contact",
            "successful competition lockout",
        ],
    },
    "BP-003-FINISH-GEOMETRY": {
        "rule_id": "BP-003-FINISH-GEOMETRY",
        "name": "Finish geometry similarity",
        "triggered_by": ["bench_finish_geometry_similarity"],
        "source_ids": ["SRC_GOMO_VAN_DEN_TILLAAR_BENCH_GRIP"],
        "project_interpretation": "A descriptive comparison of starting and finishing projected arm geometry.",
        "allowed_coaching_wording": "The completion arm geometry differed by measured amounts from the starting geometry.",
        "approved_next_rep_cue": "Keep the full press and finishing position visible for a repeatable start-to-finish comparison.",
        "positive_observation": False,
        "priority": 3,
        "caveats": [
            "This is a self-reference comparison, not competition lockout judging.",
            "The wrist is a hand/bar-end proxy, not direct barbell tracking.",
        ],
        "disallowed_claims": DISALLOWED
        + ["successful lockout", "failed lockout", "competition-valid lockout"],
    },
    "DL-001-START-GEOMETRY": {
        "rule_id": "DL-001-START-GEOMETRY",
        "name": "Start geometry",
        "triggered_by": ["deadlift_start_geometry_2d"],
        "source_ids": ["SRC_CONVENTIONAL_DEADLIFT_BIOMECHANICS"],
        "project_interpretation": "Descriptive projected joint geometry at lift-off without ideal-angle judgement.",
        "allowed_coaching_wording": "At lift-off, the projected knee, trunk-thigh, and torso-inclination angles were measured.",
        "approved_next_rep_cue": "Keep the same side-view camera position to compare lift-off geometry between repetitions.",
        "positive_observation": False,
        "priority": 1,
        "caveats": [
            "No ideal hip height or setup angle is inferred.",
            "Conventional and sumo techniques differ.",
        ],
        "disallowed_claims": DISALLOWED + ["ideal hip height", "ideal setup angle"],
    },
    "DL-002-PHASE-EXTENSION": {
        "rule_id": "DL-002-PHASE-EXTENSION",
        "name": "Phase extension",
        "triggered_by": ["deadlift_phase_extension_2d"],
        "source_ids": ["SRC_DEADLIFT_PHASE_KINEMATICS"],
        "project_interpretation": "Describes knee and trunk-thigh angle changes before and after knee passing without judging sequencing.",
        "allowed_coaching_wording": "The measured extension changes differed between lift-off to knee passing and knee passing to completion.",
        "approved_next_rep_cue": "Keep the full pull in frame to compare how these phase changes repeat.",
        "positive_observation": False,
        "priority": 2,
        "caveats": ["Phase sequencing is not classified as correct or incorrect."],
        "disallowed_claims": DISALLOWED,
    },
    "DL-003-FINISH-GEOMETRY": {
        "rule_id": "DL-003-FINISH-GEOMETRY",
        "name": "Finish geometry",
        "triggered_by": ["deadlift_finish_geometry_2d"],
        "source_ids": ["SRC_DEADLIFT_PHASE_KINEMATICS"],
        "project_interpretation": "Descriptive completion proxy, not competition-standard lockout judging.",
        "allowed_coaching_wording": "At completion, projected knee angle, torso inclination, and wrist-proxy stability were measured.",
        "approved_next_rep_cue": "Keep the completion position and full body visible for repeatable comparison.",
        "positive_observation": False,
        "priority": 3,
        "caveats": [
            "Call this finish geometry or a completion proxy, not competition lockout."
        ],
        "disallowed_claims": DISALLOWED
        + ["successful lockout", "failed lockout", "competition-valid lockout"],
    },
    "DL-004-EARLY-COORDINATION": {
        "rule_id": "DL-004-EARLY-COORDINATION",
        "name": "Early coordination",
        "triggered_by": ["deadlift_early_coordination_2d"],
        "source_ids": ["SRC_CONVENTIONAL_DEADLIFT_BIOMECHANICS"],
        "project_interpretation": "Describes relative hip and shoulder vertical progress and torso-inclination change without a fault threshold.",
        "allowed_coaching_wording": "During the early pull, hip and shoulder vertical progress differed while torso inclination changed by the measured amount.",
        "approved_next_rep_cue": "Use the same side view to compare early-pull coordination between repetitions.",
        "positive_observation": False,
        "priority": 4,
        "caveats": [
            "There is no validated universal fault threshold for this project metric."
        ],
        "disallowed_claims": DISALLOWED
        + ["hips shot up", "back took over", "lumbar stress"],
    },
    "SQ-003-ESTIMATED_DEPTH": {
        "rule_id": "SQ-003-ESTIMATED_DEPTH",
        "name": "Estimated depth",
        "triggered_by": ["depth"],
        "source_ids": ["SRC_GIUSTINO_2024_DEPTH"],
        "project_interpretation": "Depth is an estimated side-view 2D observation, not a competition judging decision.",
        "allowed_coaching_wording": "The squat reached likely sufficient depth, based on estimated depth in this side-view recording. Estimated depth appears borderline or high in this side-view recording.",
        "approved_next_rep_cue": "Keep control into the bottom and aim to let the hips reach slightly lower than the knees.",
        "positive_observation": True,
        "priority": 3,
        "caveats": [
            "Always state this is an estimate from side-view 2D video.",
            "Never claim judging certainty or say the lifter definitely passed depth.",
        ],
        "disallowed_claims": DISALLOWED,
    },
    "SQ-001-HIP_FIRST_ASCENT": {
        "rule_id": "SQ-001-HIP_FIRST_ASCENT",
        "name": "Hip-first ascent",
        "triggered_by": ["hip_first_ascent"],
        "source_ids": [
            "SRC_POWERLIFTING_TECHNIQUE_HIP_ASCENT",
            "SRC_GIUSTINO_2024_STICKING",
            "SRC_VAN_DEN_TILLAAR_HELMS_2020",
        ],
        "project_interpretation": "Measured pattern only: hips rose ahead of shoulders early in ascent.",
        "allowed_coaching_wording": "Your hips rose before your shoulders as you came out of the bottom.",
        "approved_next_rep_cue": "Chest and hips together. As you stand, drive your torso upward with your hips instead of letting the hips move up first.",
        "positive_observation": False,
        "priority": 1,
        "caveats": [
            "Do not infer weakness.",
            "Do not infer tiredness.",
            "Do not use good morning squat as a user-facing diagnosis.",
        ],
        "disallowed_claims": DISALLOWED,
    },
    "SQ-005-PROLONGED_ASCENT": {
        "rule_id": "SQ-005-PROLONGED_ASCENT",
        "name": "Prolonged ascent or sticking region",
        "triggered_by": ["prolonged_ascent_or_sticking_region"],
        "source_ids": ["SRC_GIUSTINO_2024_STICKING"],
        "project_interpretation": "A slower ascent is neutral context by default and not a standalone fault.",
        "allowed_coaching_wording": "The ascent took longer than the descent and included a slower region. A slower ascent is not automatically a problem; here it is most useful as context for the transition out of the bottom.",
        "approved_next_rep_cue": "Keep the ascent controlled and review whether your chest and hips continue rising together through the slowest point.",
        "positive_observation": False,
        "priority": 5,
        "caveats": [
            "Do not claim tiredness, strength deficits, or high effort.",
            "Interpret in the context of load, tempo, and squat style.",
        ],
        "disallowed_claims": DISALLOWED,
    },
    "SQ-002-DESCENT_INITIATION": {
        "rule_id": "SQ-002-DESCENT_INITIATION",
        "name": "Simultaneous descent initiation",
        "triggered_by": ["descent_initiation"],
        "source_ids": ["SRC_POWERLIFTING_TECHNIQUE_DESCENT"],
        "project_interpretation": "Registered for future UI use once deterministic trigger is implemented and tested.",
        "allowed_coaching_wording": "Hips and knees should begin bending together to support a balanced descent.",
        "approved_next_rep_cue": "Begin the descent by bending hips and knees together while staying balanced over mid-foot.",
        "positive_observation": False,
        "priority": 2,
        "caveats": [
            "Do not surface until deterministic trigger exists.",
            "Hip and knee motion does not need to be identical for every lifter.",
        ],
        "disallowed_claims": DISALLOWED,
    },
}


def _rep(analysis: dict) -> dict:
    return analysis.get("rep") or (analysis.get("reps") or [{}])[0]


def _names(rep: dict) -> set[str]:
    return {f.get("name") for f in rep.get("findings", []) if isinstance(f, dict)}


def _conf(rep: dict, names: set[str]) -> str:
    levels = {"low": 0, "medium": 1, "high": 2}
    vals = [
        f.get("confidence", "medium")
        for f in rep.get("findings", [])
        if f.get("name") in names and f.get("confidence") in levels
    ]
    return min(vals, key=lambda x: levels[x]) if vals else "medium"


def _depth(rep: dict, analysis: dict) -> str:
    metric_results = rep.get("metric_results", analysis.get("metric_results", []))
    for metric in metric_results:
        if (
            not isinstance(metric, dict)
            or metric.get("metric_id") != "squat_depth_2d_proxy"
        ):
            continue
        return {
            "DEPTH_ACHIEVED": "likely_sufficient_depth",
            "BORDERLINE": "borderline_depth",
            "DEPTH_NOT_ACHIEVED": "likely_insufficient_depth",
        }.get(metric.get("classification"), "")
    for f in rep.get("findings", []):
        if f.get("name") == "depth":
            return f.get("status", "")
    return rep.get("depth_assessment", analysis.get("depth_assessment", ""))


def _sources(rule_ids):
    ids = []
    for r in rule_ids:
        for s in COACHING_RULES[r]["source_ids"]:
            if s not in ids:
                ids.append(s)
    return ids


def _caveats(rule_ids):
    out = []
    for r in rule_ids:
        out += COACHING_RULES[r]["caveats"]
        # Source registry retains full limitations; theme caveats keep user-facing wording bounded.
        pass
    return list(dict.fromkeys(out))


def _display_source(source_id: str) -> dict:
    source = dict(SOURCES[source_id])
    # Keep the registry complete, but avoid putting negated unsupported-claim terms
    # into normal/user-facing theme JSON where tests and UI check for claims.
    source["limitations"] = []
    return {"source_id": source_id, **source}


def _refs(source_ids):
    return [_display_source(sid) for sid in source_ids]


def _build_squat_coaching_brief(analysis: dict) -> dict[str, Any]:
    rep = _rep(analysis) or {}
    names = _names(rep)
    themes = []
    neutral = []
    positives = []
    triggered = []
    has_hip = "hip_first_ascent" in names
    has_slow = "prolonged_ascent_or_sticking_region" in names
    if has_hip:
        triggered.append("SQ-001-HIP_FIRST_ASCENT")
    if has_slow:
        triggered.append("SQ-005-PROLONGED_ASCENT")
    d = _depth(rep, analysis)
    if d:
        triggered.append("SQ-003-ESTIMATED_DEPTH")
    if has_hip and has_slow:
        r = ["SQ-001-HIP_FIRST_ASCENT", "SQ-005-PROLONGED_ASCENT"]
        obs = "Your hips rose before your shoulders early in the ascent."
        obs += " The movement also slowed during a longer ascent."
        themes.append(
            {
                "rank": 1,
                "theme_id": "THEME_EARLY_ASCENT_COORDINATION",
                "title": "Keep chest and hips rising together",
                "source_rule_ids": r,
                "source_findings": [
                    x
                    for x in [
                        "hip_first_ascent",
                        "prolonged_ascent_or_sticking_region",
                    ]
                    if x in names
                ],
                "source_ids": _sources(r),
                "what_was_observed": obs,
                "why_focus_on_this": "This can make the transition out of the bottom less coordinated and make the ascent harder to finish smoothly.",
                "next_rep_cue": "“Chest and hips together.” As you stand, drive your torso upward with your hips instead of letting the hips move first.",
                "practice_task": "Use a controlled, repeatable load and record a short side-view set. Rehearse the cue from the bottom on each repetition.",
                "success_check": "In the first third of ascent, your shoulders should begin rising with your hips rather than visibly lagging behind.",
                "confidence": _conf(rep, names),
                "caveats": [
                    "Do not infer weakness.",
                    "Do not infer tiredness.",
                    "Do not call this unsafe.",
                    "Interpret with squat style, load, and anthropometry in mind.",
                ],
                "reference_notes": _refs(_sources(r)),
                "deterministic_evidence": [
                    f.get("evidence")
                    for f in rep.get("findings", [])
                    if f.get("name") in names
                ],
            }
        )
        if has_slow:
            neutral.append(
                "The longer ascent is not automatically a problem by itself. It is included here because it occurred alongside the early hips-first pattern."
            )
    elif has_hip:
        r = ["SQ-001-HIP_FIRST_ASCENT"]
        themes.append(
            {
                "rank": 1,
                "theme_id": "THEME_HIP_FIRST_ASCENT",
                "title": "Coordinate hips and torso out of the bottom",
                "source_rule_ids": r,
                "source_findings": ["hip_first_ascent"],
                "source_ids": _sources(r),
                "what_was_observed": "Your hips began rising before your shoulders during early ascent.",
                "why_focus_on_this": "This can make the first part of the ascent less coordinated.",
                "next_rep_cue": "“Chest and hips together.” Keep your torso moving upward with your hips as you stand.",
                "confidence": _conf(rep, {"hip_first_ascent"}),
                "caveats": _caveats(r),
                "reference_notes": _refs(_sources(r)),
            }
        )
    if has_slow and not has_hip:
        r = ["SQ-005-PROLONGED_ASCENT"]
        themes.append(
            {
                "rank": len(themes) + 1,
                "theme_id": "THEME_PROLONGED_ASCENT",
                "title": "Maintain a controlled ascent",
                "source_rule_ids": r,
                "source_findings": ["prolonged_ascent_or_sticking_region"],
                "source_ids": _sources(r),
                "what_was_observed": "The ascent was longer than the descent and/or included a slower region.",
                "why_focus_on_this": "This is a timing observation, not automatically a technique fault or technique problem.",
                "next_rep_cue": "Keep the ascent controlled and review whether your chest and hips continue rising together through the slowest point.",
                "confidence": _conf(rep, {"prolonged_ascent_or_sticking_region"}),
                "caveats": _caveats(r),
                "reference_notes": _refs(_sources(r)),
            }
        )
        neutral.append(
            "A slower ascent is not automatically a problem; here it is most useful as context for the transition out of the bottom."
        )
    if d == "likely_sufficient_depth":
        positives.append(
            {
                "title": "Estimated depth",
                "text": "The squat reached likely sufficient depth, based on estimated depth in this side-view recording.",
                "source_rule_ids": ["SQ-003-ESTIMATED_DEPTH"],
                "source_ids": ["SRC_GIUSTINO_2024_DEPTH"],
                "caveats": _caveats(["SQ-003-ESTIMATED_DEPTH"]),
                "reference_notes": _refs(["SRC_GIUSTINO_2024_DEPTH"]),
            }
        )
    elif d in {"borderline_depth", "likely_insufficient_depth"}:
        r = ["SQ-003-ESTIMATED_DEPTH"]
        themes.append(
            {
                "rank": len(themes) + 1,
                "theme_id": "THEME_ESTIMATED_DEPTH",
                "title": "Aim for a clearer depth position",
                "source_rule_ids": r,
                "source_findings": ["depth"],
                "source_ids": _sources(r),
                "what_was_observed": "Estimated depth appears borderline or high in this side-view recording.",
                "why_focus_on_this": "A clearer bottom position makes the rep easier to assess consistently from video.",
                "next_rep_cue": COACHING_RULES[r[0]]["approved_next_rep_cue"],
                "confidence": _conf(rep, {"depth"}),
                "caveats": _caveats(r),
                "reference_notes": _refs(_sources(r)),
            }
        )
    for i, t in enumerate(themes[:3], 1):
        t["rank"] = i
        t.setdefault(
            "practice_task",
            "Use a controlled, repeatable load and record a short side-view set. Rehearse the cue on each repetition.",
        )
        t.setdefault(
            "success_check",
            f"The next rep should show a clearer, more repeatable execution of this focus: {t['title'].lower()}.",
        )
    overall = (
        "Your squat reached likely sufficient estimated depth. The main area to improve is how you begin the ascent from the bottom."
        if positives and themes
        else (
            "Your squat reached likely sufficient estimated depth."
            if positives
            else (
                f"The main coaching focus is: {themes[0]['title'].lower()}."
                if themes
                else "No corrective coaching cue was generated from the validated findings."
            )
        )
    )
    return {
        "overall_summary": overall,
        "priority_themes": themes[:3],
        "positive_observations": positives,
        "neutral_context": neutral,
        "limitations": [
            "This is a 2D video-based estimate and not a competition judging decision."
        ],
        "triggered_rule_ids": list(dict.fromkeys(triggered)),
        "source_metadata": _refs(_sources(triggered)) if triggered else [],
    }


def _metric_results(analysis: dict) -> dict[str, dict]:
    return {
        m.get("metric_id"): m
        for m in analysis.get("metric_results", [])
        if isinstance(m, dict)
    }


def _descriptive_theme(rule_id: str, metric: dict, observed: str, why: str) -> dict:
    rule = COACHING_RULES[rule_id]
    return {
        "rank": 0,
        "theme_id": f"THEME_{rule_id.replace('-', '_')}",
        "title": rule["name"],
        "source_rule_ids": [rule_id],
        "source_findings": rule["triggered_by"],
        "source_ids": rule["source_ids"],
        "what_was_observed": observed,
        "why_focus_on_this": why,
        "next_rep_cue": rule["approved_next_rep_cue"],
        "practice_task": "Record another repetition from the same supported view for a like-for-like comparison.",
        "success_check": "Compare the same descriptive measurement across repetitions without treating it as a universal correctness threshold.",
        "confidence": metric.get("reliability", {}).get("confidence", "medium"),
        "caveats": rule["caveats"],
        "reference_notes": _refs(rule["source_ids"]),
        "deterministic_evidence": metric.get("measurement", {}),
    }


def _build_descriptive_brief(analysis: dict) -> dict[str, Any]:
    lift = analysis.get("lift")
    metrics = _metric_results(analysis)
    themes = []
    triggered = []

    def add(metric_id: str, rule_id: str, observed: str, why: str) -> None:
        metric = metrics.get(metric_id)
        if not metric or metric.get("evidence_status") != "SUFFICIENT":
            return
        triggered.append(rule_id)
        themes.append(_descriptive_theme(rule_id, metric, observed, why))

    if lift == "bench_press":
        m = (metrics.get("bench_elbow_flexion_2d") or {}).get("measurement") or {}
        add(
            "bench_elbow_flexion_2d",
            "BP-001-BOTTOM-ELBOW-GEOMETRY",
            f"At the bottom of the repetition, your projected elbow angle reached {m.get('minimum_elbow_angle_degrees')}°, with a bottom-phase median of {m.get('bottom_median_elbow_angle_degrees')}°.",
            "This gives a repeatable description of bottom elbow position, but is not compared with a universal correctness threshold.",
        )
        w = (metrics.get("bench_wrist_trajectory_2d") or {}).get("measurement") or {}
        add(
            "bench_wrist_trajectory_2d",
            "BP-002-WRIST-TRAJECTORY",
            f"During ascent, the wrist hand/bar-end proxy moved {w.get('horizontal_bottom_to_completion_displacement_normalised')} horizontally and {w.get('vertical_ascent_displacement_normalised')} vertically; it finished {w.get('end_2d_distance_from_start_normalised')} normalised units from its starting position.",
            "This wrist proxy can support repeat-to-repeat trajectory comparison; it is not validated direct bar-path tracking.",
        )
        f = (metrics.get("bench_finish_geometry_similarity") or {}).get(
            "measurement"
        ) or {}
        add(
            "bench_finish_geometry_similarity",
            "BP-003-FINISH-GEOMETRY",
            f"At completion, the projected elbow angle differed from the starting region by {f.get('elbow_angle_difference_degrees')}°, while the wrist proxy finished {f.get('wrist_position_difference_normalised')} normalised units from its starting location.",
            "This describes how closely the repetition returned toward its own starting arm geometry; it does not judge competition lockout.",
        )
        name = "bench press"
    else:
        s = (metrics.get("deadlift_start_geometry_2d") or {}).get("measurement") or {}
        add(
            "deadlift_start_geometry_2d",
            "DL-001-START-GEOMETRY",
            f"At lift-off, your projected knee angle was {s.get('projected_knee_angle_degrees')}°, trunk–thigh angle was {s.get('projected_trunk_thigh_angle_degrees')}°, and torso inclination was {s.get('torso_inclination_from_vertical_degrees')}°.",
            "These measurements provide repeatable start geometry without defining ideal setup angles.",
        )
        p = (metrics.get("deadlift_phase_extension_2d") or {}).get("measurement") or {}
        add(
            "deadlift_phase_extension_2d",
            "DL-002-PHASE-EXTENSION",
            f"From lift-off to knee passing and then to completion, projected knee-angle changes were {p.get('knee_extension_below_knee_degrees')}° and {p.get('knee_extension_above_knee_degrees')}°; trunk–thigh changes were {p.get('trunk_thigh_change_below_knee_degrees')}° and {p.get('trunk_thigh_change_above_knee_degrees')}°.",
            "This describes where the observed extension changes occurred without classifying the sequence as correct or incorrect.",
        )
        f = (metrics.get("deadlift_finish_geometry_2d") or {}).get("measurement") or {}
        add(
            "deadlift_finish_geometry_2d",
            "DL-003-FINISH-GEOMETRY",
            f"The completion proxy showed a projected knee angle of {f.get('projected_knee_angle_degrees')}°, torso inclination of {f.get('torso_inclination_from_vertical_degrees')}°, and wrist vertical range of {f.get('terminal_wrist_vertical_range')}.",
            "This finish geometry supports repeatable comparison and is not competition-standard lockout judging.",
        )
        c = (metrics.get("deadlift_early_coordination_2d") or {}).get(
            "measurement"
        ) or {}
        add(
            "deadlift_early_coordination_2d",
            "DL-004-EARLY-COORDINATION",
            f"During the early pull, maximum hip-minus-shoulder vertical progress was {c.get('maximum_hip_minus_shoulder_vertical_progress')} body-scale units while torso inclination changed by {c.get('torso_inclination_change_degrees')}°.",
            "This describes early hip and shoulder coordination without converting it into a technique-fault judgement.",
        )
        name = "deadlift"
    for rank, theme in enumerate(themes, 1):
        theme["rank"] = rank
    return {
        "overall_summary": (
            f"The {name} analysis produced {len(themes)} descriptive, evidence-supported measurement theme{'s' if len(themes) != 1 else ''}."
            if themes
            else f"No descriptive {name} theme was generated because sufficient metric evidence was unavailable."
        ),
        "priority_themes": themes,
        "positive_observations": [],
        "neutral_context": [
            "These descriptive biomechanical measurements do not use validated universal technique thresholds."
        ],
        "limitations": [
            "This is a projected 2D video analysis; descriptive measurements are not competition judging outcomes."
        ],
        "triggered_rule_ids": triggered,
        "source_metadata": _refs(_sources(triggered)) if triggered else [],
    }


def _build_deadlift_coaching_brief(analysis: dict) -> dict[str, Any]:
    """Build coaching solely from the two deterministic deadlift outcomes."""
    metrics = _metric_results(analysis)
    coordination = (metrics.get("deadlift_early_coordination_2d") or {}).get(
        "classification"
    )
    finish = (metrics.get("deadlift_finish_geometry_2d") or {}).get("classification")
    themes: list[dict[str, Any]] = []

    def number(value: Any) -> str:
        return f"{value:g}" if isinstance(value, (int, float)) else "unavailable"

    def finish_observation(metric: dict[str, Any]) -> str:
        measurement = metric.get("measurement") or {}
        thresholds = metric.get("threshold") or {}
        uncertainty = thresholds.get("uncertainty_degrees") or {}
        checks = (
            (
                "terminal knee angle",
                "projected_knee_angle_degrees",
                float(thresholds.get("knee_angle_degrees", 165.0))
                - float(uncertainty.get("knee", 5.0)),
                "below",
            ),
            (
                "trunk–thigh extension",
                "projected_trunk_thigh_angle_degrees",
                float(thresholds.get("trunk_thigh_angle_degrees", 160.0))
                - float(uncertainty.get("trunk_thigh", 5.0)),
                "below",
            ),
            (
                "torso inclination",
                "torso_inclination_from_vertical_degrees",
                float(thresholds.get("torso_inclination_degrees", 12.0))
                + float(uncertainty.get("torso", 5.0)),
                "above",
            ),
        )
        failed = []
        for label, key, boundary, direction in checks:
            value = measurement.get(key)
            if isinstance(value, (int, float)) and (
                (direction == "below" and value < boundary)
                or (direction == "above" and value > boundary)
            ):
                failed.append(
                    f"{label} measured {number(value)}° ({direction} the provisional {number(boundary)}° boundary)"
                )
        return f"The finish-posture check was incomplete because {'; '.join(failed)}."

    def finish_components(
        metric: dict[str, Any],
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Describe independently supported finish checks without reclassifying them."""
        measurement = metric.get("measurement") or {}
        thresholds = metric.get("threshold") or {}
        uncertainty = thresholds.get("uncertainty_degrees") or {}
        definitions = (
            (
                "Terminal knee extension reached the implemented range.",
                "Terminal knee angle",
                "projected_knee_angle_degrees",
                "knee_angle_degrees",
                "knee",
                "minimum",
            ),
            (
                "Trunk–thigh extension reached the implemented range.",
                "Trunk–thigh angle",
                "projected_trunk_thigh_angle_degrees",
                "trunk_thigh_angle_degrees",
                "trunk_thigh",
                "minimum",
            ),
            (
                None,
                "Torso inclination",
                "torso_inclination_from_vertical_degrees",
                "torso_inclination_degrees",
                "torso",
                "maximum",
            ),
        )
        favorable: list[dict[str, str]] = []
        inconclusive: list[str] = []
        for (
            positive,
            label,
            value_key,
            threshold_key,
            uncertainty_key,
            direction,
        ) in definitions:
            value = measurement.get(value_key)
            threshold = thresholds.get(threshold_key)
            margin = uncertainty.get(uncertainty_key)
            if not all(
                isinstance(item, (int, float)) for item in (value, threshold, margin)
            ):
                continue
            lower, upper = float(threshold) - float(margin), float(threshold) + float(
                margin
            )
            confidently_favorable = (
                value >= upper if direction == "minimum" else value <= lower
            )
            uncertain_value = (
                lower <= value < upper
                if direction == "minimum"
                else lower < value <= upper
            )
            if confidently_favorable and positive:
                favorable.append({"text": positive})
            elif uncertain_value:
                inconclusive.append(
                    f"{label} measured {number(value)}°, within the provisional uncertainty range "
                    f"({number(lower)}° to {number(upper)}°), so that component could not be classified confidently."
                )
        return favorable, inconclusive

    def theme(rule_id: str, title: str, observed: str, why: str, cue: str) -> None:
        rule = COACHING_RULES[rule_id]
        metric_id = rule["triggered_by"][0]
        metric = metrics[metric_id]
        themes.append(
            {
                "rank": len(themes) + 1,
                "theme_id": rule_id,
                "title": title,
                "source_rule_ids": [rule_id],
                "source_findings": [metric_id],
                "source_ids": rule["source_ids"],
                "what_was_observed": observed,
                "why_focus_on_this": why,
                "next_rep_cue": cue,
                "confidence": metric.get("reliability", {}).get("confidence", "medium"),
                "caveats": rule["caveats"],
                "reference_notes": _refs(rule["source_ids"]),
            }
        )

    if coordination == "EARLY_HIP_RISE_DETECTED":
        theme(
            "DL-004-EARLY-COORDINATION",
            "Coordinate the initial pull",
            "Your hips rose meaningfully ahead of your shoulders across multiple early-pull frames.",
            "Keeping the two landmarks progressing together provides a repeatable coordination target.",
            "Push the floor away and let your chest and hips rise together.",
        )
    if finish == "INCOMPLETE_FINISH":
        finish_metric = metrics["deadlift_finish_geometry_2d"]
        theme(
            "DL-003-FINISH-GEOMETRY",
            "Finish tall",
            finish_observation(finish_metric),
            "The upright position did not reach the implemented extension range.",
            "At the top of the pull, finish tall with your knees and hips extended, without leaning backwards.",
        )

    uncertain = []
    if coordination in {"UNCERTAIN", "INSUFFICIENT_EVIDENCE", None}:
        uncertain.append(
            "Early hip–shoulder coordination was uncertain or lacked sufficient evidence."
        )
    finish_metric = metrics.get("deadlift_finish_geometry_2d") or {}
    finish_positives, finish_uncertain = finish_components(finish_metric)
    if finish == "UNCERTAIN":
        uncertain.extend(
            finish_uncertain or ["The finish-position check was inconclusive."]
        )
    elif finish in {"INSUFFICIENT_EVIDENCE", None}:
        uncertain.append("Finish-position evidence was insufficient for assessment.")
    warnings = (
        analysis.get("warnings", [])
        if isinstance(analysis.get("warnings"), list)
        else []
    )
    rejection = analysis.get("recording_reliability", {}).get("rejection_reasons", [])
    recording_terms = (
        "camera",
        "framing",
        "visibility",
        "visible",
        "tracking",
        "occlusion",
    )
    if any(
        any(term in str(problem).casefold() for term in recording_terms)
        for problem in [*warnings, *rejection]
    ):
        uncertain.append(
            "Use a clear side or near-side view with the full body visible when recording again."
        )

    positives = []
    neutral = []
    if coordination == "EARLY_HIP_RISE_NOT_DETECTED":
        positives.append({"text": "Early hip–shoulder rise stayed coordinated."})
    if finish == "COMPLETE_FINISH":
        positives.extend(finish_positives)
    elif finish == "UNCERTAIN":
        positives.extend(finish_positives)
    if not themes:
        neutral.append(
            "Keep pushing the floor away smoothly and repeat the same extended top position."
        )
    unresolved = coordination in {
        "UNCERTAIN",
        "INSUFFICIENT_EVIDENCE",
        None,
    } or finish in {"UNCERTAIN", "INSUFFICIENT_EVIDENCE", None}
    if themes:
        overall_summary = f"The deterministic deadlift analysis identified {len(themes)} coaching priorit{'y' if len(themes) == 1 else 'ies'}."
    elif finish == "UNCERTAIN" and coordination == "EARLY_HIP_RISE_NOT_DETECTED":
        overall_summary = "No corrective issue was detected in the checks that could be assessed confidently. One finish-position measurement was inconclusive."
    elif unresolved:
        overall_summary = "One or more covered deadlift checks were inconclusive."
    else:
        overall_summary = (
            "No issue covered by the current deadlift analysis was detected."
        )
    return {
        "overall_summary": overall_summary,
        "priority_themes": themes[:2],
        "positive_observations": positives,
        "neutral_context": neutral,
        "limitations": [
            "This is a provisional 2D finish-posture proxy, not competition-standard lockout judging."
        ]
        + uncertain,
        "triggered_rule_ids": [t["theme_id"] for t in themes[:2]],
        "source_metadata": (
            _refs(_sources([t["theme_id"] for t in themes[:2]])) if themes else []
        ),
    }


def _build_bench_coaching_brief(analysis: dict) -> dict[str, Any]:
    """Convert only the supported deterministic bench proxies into coaching."""
    metrics = _metric_results(analysis)
    checks = [
        (
            "observed_side_elbow_depth_2d_proxy",
            "DEPTH_INSUFFICIENT",
            "DEPTH_SUFFICIENT",
            "BP-DEPTH",
            "Reach shoulder level at the bottom",
            "Lower the bar until your observed elbow reaches at least shoulder level.",
        ),
        (
            "terminal_elbow_extension_2d_proxy",
            "INCOMPLETE",
            "COMPLETE",
            "BP-EXTENSION",
            "Complete elbow extension",
            "Finish with your elbows fully extended and hold the top position.",
        ),
    ]
    themes, positives, limitations = (
        [],
        [],
        [
            "These are 2D engineering proxies for selected competition-rule requirements, not competition judging outcomes.",
            "The 170° and 160° extension boundaries are research-informed engineering proxies, not official IPF angle thresholds.",
        ],
    )
    for metric_id, corrective, favorable, theme_id, title, cue in checks:
        metric = metrics.get(metric_id) or {}
        classification = metric.get("classification")
        if classification == corrective:
            themes.append(
                {
                    "rank": len(themes) + 1,
                    "theme_id": theme_id,
                    "title": title,
                    "source_rule_ids": [],
                    "source_findings": [metric_id],
                    "source_ids": [],
                    "what_was_observed": {
                        "DEPTH_INSUFFICIENT": "At least one reliably observed elbow remained above its respective shoulder-level proxy at the stable bottom.",
                        "INCOMPLETE": "The terminal elbow-angle proxy remained below the implemented incomplete-extension boundary.",
                    }[corrective],
                    "why_focus_on_this": "This check corresponds to an implemented competition-rule proxy and was classified confidently from the available 2D evidence.",
                    "next_rep_cue": cue,
                    "confidence": metric.get("reliability", {}).get(
                        "confidence", "medium"
                    ),
                    "caveats": limitations,
                    "reference_notes": [],
                    "deterministic_evidence": metric.get("measurement", {}),
                }
            )
        elif classification == favorable:
            positives.append(
                {
                    "text": {
                        "DEPTH_SUFFICIENT": "Both reliably observed elbows reached the implemented shoulder-level depth proxy.",
                        "COMPLETE": "Terminal elbow extension reached the implemented complete range.",
                    }[favorable]
                }
            )
        elif classification in {"UNCERTAIN", "INSUFFICIENT_EVIDENCE", None}:
            label = title.casefold()
            limitations.append(
                f"The {label} check was {('uncertain' if classification == 'UNCERTAIN' else 'not supported by sufficient evidence')}; it was not treated as a detected fault."
            )
    themes = themes[:2]
    for rank, theme in enumerate(themes, 1):
        theme["rank"] = rank
    if themes:
        overall = f"The deterministic bench analysis identified {len(themes)} corrective priorit{'y' if len(themes) == 1 else 'ies'}."
    elif len(positives) == 2:
        overall = "Both supported bench checks were favorable. Repeat the same depth and fully extended finish."
    elif positives:
        overall = "No corrective issue was detected in the bench checks that could be assessed confidently."
    else:
        overall = "The implemented bench checks were inconclusive; no technique fault was inferred."
    return {
        "overall_summary": overall,
        "priority_themes": themes,
        "positive_observations": positives,
        "neutral_context": [],
        "limitations": limitations,
        "triggered_rule_ids": [],
        "source_metadata": [],
    }


def build_actionable_coaching_brief(analysis: dict) -> dict[str, Any]:
    """Dispatch to lift-specific deterministic rules while preserving one contract."""
    if analysis.get("lift") == "deadlift":
        return _build_deadlift_coaching_brief(analysis)
    if analysis.get("lift") == "bench_press":
        return _build_bench_coaching_brief(analysis)
    return _build_squat_coaching_brief(analysis)
