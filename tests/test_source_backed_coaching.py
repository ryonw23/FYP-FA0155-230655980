import json
from powerlifting_coach.coaching_sources import SOURCES
from powerlifting_coach.coaching_rule_library import COACHING_RULES, build_actionable_coaching_brief
from powerlifting_coach.llm_feedback import build_feedback_payload, build_template_feedback, generate_local_feedback, LocalModelConfig
from powerlifting_coach import app
from tests.test_llm_feedback import valid_analysis, make_local_model


def test_rules_have_valid_sources_and_websites_not_peer_reviewed():
    for rule in COACHING_RULES.values():
        assert rule["source_ids"]
        assert all(sid in SOURCES for sid in rule["source_ids"])
    assert all(s["source_type"] != "peer_reviewed_article" for s in SOURCES.values() if s["source_type"] == "coaching_website")


def test_incomplete_references_do_not_fabricate_formal_details():
    for source in SOURCES.values():
        if not source["citation_complete"]:
            note = source["bibliography_note"].lower()
            assert "add" in note or "later" in note
            assert "doi:" not in source["short_reference"].lower()


def test_descriptive_torso_change_does_not_modify_supported_theme():
    analysis = valid_analysis()
    analysis["rep"]["findings"].insert(1, {"name": "torso_lean_change", "confidence": "medium", "evidence": "Torso leaned forward more at bottom."})
    brief = build_actionable_coaching_brief(analysis)
    assert len(brief["priority_themes"]) == 1
    assert brief["priority_themes"][0]["title"] == "Keep chest and hips rising together"
    assert "torso_lean_change" not in brief["priority_themes"][0]["source_findings"]
    assert all(sid in SOURCES for t in brief["priority_themes"] for sid in t["source_ids"])


def test_torso_only_has_no_corrective_theme_or_fallback_cue():
    analysis = valid_analysis()
    analysis["rep"]["findings"] = [f for f in analysis["rep"]["findings"] if f["name"] in {"torso_lean_change", "depth"}]
    analysis["rep"]["findings"].insert(0, {"name": "torso_lean_change", "confidence": "medium", "evidence": "Torso leaned forward more at bottom."})
    analysis["rep"]["phase_angles"]["setup"] = {"torso_lean_degrees": {"median": 5.7}}
    analysis["rep"]["phase_angles"]["bottom"]["torso_lean_degrees"] = {"median": 21.9}
    analysis["rep"]["derived_metrics"]["bottom_torso_lean_change_degrees"] = 16.2

    brief = build_actionable_coaching_brief(analysis)
    fallback = build_template_feedback(analysis)

    assert brief["priority_themes"] == []
    assert fallback["priority_cues"] == []
    assert "every technique check passed" not in fallback["overall_assessment"].lower()


def test_no_forbidden_claims_in_rule_output():
    text = json.dumps(build_actionable_coaching_brief(valid_analysis())).lower()
    for phrase in ["weak quadriceps", "injury risk", "dangerous", "medical diagnosis", "competition valid"]:
        assert phrase not in text


def test_bad_llm_phrases_and_unknown_source_fallback(monkeypatch):
    bad = {"overall_assessment":"Your squat reached likely sufficient depth.","priority_cues":[{"rank":1,"title":"Keep chest and hips rising together","what_was_observed":"Your hips rose first.","why_focus_on_this":"This shows weak quads.","next_rep_cue":"Chest and hips together.","confidence":"medium","source_ids":["SRC_UNKNOWN"]}],"positive_observations":[],"neutral_context":[],"limitations":["This is a 2D video-based estimate and not a competition judging decision."]}
    monkeypatch.setattr("powerlifting_coach.llm_feedback._load_local_model", make_local_model(chat_content=bad))
    monkeypatch.setattr(
        "powerlifting_coach.llm_feedback.check_local_model_status",
        lambda config: {"reachable": True, "configured_model_installed": True},
    )
    assert generate_local_feedback(valid_analysis(), LocalModelConfig(model_path="unused.gguf"))["source"] == "fallback"


def test_fallback_has_evidence_limits_and_payload_sanitized():
    fb = build_template_feedback(valid_analysis())
    assert fb["priority_cues"][0]["evidence_and_limits"]["references"]
    assert "frames" not in json.dumps(build_feedback_payload(valid_analysis()))
    assert "landmarks" not in json.dumps(build_feedback_payload(valid_analysis()))


def test_ui_shows_evidence_not_internal_ids():
    md = app._feedback_markdown(build_template_feedback(valid_analysis()))
    assert "Evidence and limits" in md
    assert "Peer-reviewed article" in md or "Coaching website" in md
    assert "hip_first_ascent" not in md
