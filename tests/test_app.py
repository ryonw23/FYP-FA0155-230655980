import csv
from pathlib import Path
from types import SimpleNamespace

from powerlifting_coach import app
from powerlifting_coach.model_assets import download_file


def _prepare_callback_test(monkeypatch, tmp_path, *, analysis):
    video_path = tmp_path / "upload.mp4"
    model_path = tmp_path / "pose.task"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"model")
    monkeypatch.chdir(tmp_path)

    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("Analyse must not access the network")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_download)

    def fake_extract(**kwargs):
        Path(kwargs["log_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["log_path"]).write_text("{}", encoding="utf-8")
        if kwargs["annotated_video_path"] is not None:
            Path(kwargs["annotated_video_path"]).write_bytes(b"annotated")
        return SimpleNamespace(
            frames=[], metadata={"fps": 30.0, "pose_counts_by_frame": []}
        )

    def fake_render(_input, output, *_args):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"annotated")

    def return_analysis(*_args, **_kwargs):
        return analysis

    def return_deterministic_feedback(_analysis):
        return {
            "feedback": {"overall_assessment": "Safe deterministic fallback."},
            "source": "fallback",
            "status": "Fallback retained",
        }

    monkeypatch.setattr(app, "extract_pose", fake_extract)
    monkeypatch.setattr(app, "render_pose_overlay", fake_render)
    monkeypatch.setattr(app, "analyse_squat_landmarks", return_analysis)
    monkeypatch.setattr(app, "analyse_bench_press_landmarks", return_analysis)
    monkeypatch.setattr(app, "generate_local_feedback", return_deterministic_feedback)
    return video_path, model_path


def test_model_download_reuses_a_valid_cached_asset(monkeypatch, tmp_path):
    cached_model = tmp_path / "pose.task"
    cached_model.write_bytes(b"cached model")

    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("valid cached assets must not be downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_download)

    result = download_file(
        "https://example.invalid/pose.task",
        cached_model,
        "cached pose model",
        lambda path: path.read_bytes() == b"cached model",
    )

    assert result == cached_model


def test_squat_fallback_does_not_return_working_directory_for_optional_file(
    monkeypatch, tmp_path
):
    video_path, model_path = _prepare_callback_test(monkeypatch, tmp_path, analysis={})

    outputs = app.process_uploaded_video(
        str(video_path), model_path=str(model_path), lift="Squat"
    )

    assert outputs[32] is None
    assert "Safe deterministic fallback." in outputs[4]
    for index in (0, 1, 2, 3, 12, 24):
        assert Path(outputs[index]).is_file()


def test_bench_success_returns_regular_file_artifacts(monkeypatch, tmp_path):
    video_path, model_path = _prepare_callback_test(monkeypatch, tmp_path, analysis={})

    outputs = app.process_uploaded_video(
        str(video_path), model_path=str(model_path), lift="Bench Press"
    )

    for index in (0, 1, 2, 3, 12, 24, 32):
        assert Path(outputs[index]).is_file()
    assert outputs[25] == {}
    assert outputs[26]["selected_side"] is None
    assert outputs[26]["trajectory"]["raw"] == []


def test_each_request_uses_distinct_artifact_directory(monkeypatch, tmp_path):
    video_path, model_path = _prepare_callback_test(monkeypatch, tmp_path, analysis={})

    first = app.process_uploaded_video(
        str(video_path), model_path=str(model_path), lift="Squat"
    )
    second = app.process_uploaded_video(
        str(video_path), model_path=str(model_path), lift="Squat"
    )

    first_directory = Path(first[12]).parent
    second_directory = Path(second[12]).parent
    assert first_directory != second_directory
    assert first_directory.parent == second_directory.parent
    assert Path(first[1]).is_file()
    assert Path(second[1]).is_file()


def test_squat_creates_request_directory_before_landmark_export(monkeypatch, tmp_path):
    video_path, model_path = _prepare_callback_test(monkeypatch, tmp_path, analysis={})
    observed = {}

    def exporting_extractor(**kwargs):
        log_path = Path(kwargs["log_path"])
        observed["directory_existed"] = log_path.parent.is_dir()
        log_path.write_text('{"landmark_time_series": []}', encoding="utf-8")
        return SimpleNamespace(
            frames=[], metadata={"fps": 30.0, "pose_counts_by_frame": []}
        )

    monkeypatch.setattr(app, "extract_pose", exporting_extractor)

    outputs = app.process_uploaded_video(
        str(video_path), model_path=str(model_path), lift="Squat"
    )

    assert observed["directory_existed"] is True
    assert Path(outputs[24]).is_file()
    assert Path(outputs[24]).parent == Path(outputs[12]).parent


def test_phase_table_maps_common_result_phase_frames_with_named_headers():
    analysis = {
        "phase_frames": {
            "descent": {"start_frame": 12, "end_frame": 20},
            "bottom": {"start_frame": 21, "end_frame": 22},
        },
        "processing_information": {"fps": 25.0},
    }

    table = app._bench_phase_timing_table(analysis)

    assert table["headers"] == app.PHASE_TABLE_HEADERS
    assert table["data"] == [
        ["descent", 12, 20, 0.48, 0.8, 0.36],
        ["bottom", 21, 22, 0.84, 0.88, 0.08],
    ]


def test_empty_phase_and_angle_tables_keep_named_headers():
    assert app._phase_timing_table({}) == {
        "headers": app.PHASE_TABLE_HEADERS,
        "data": [],
    }
    assert app._phase_angle_table({}) == {
        "headers": app.PHASE_ANGLE_HEADERS,
        "data": [],
    }


def test_session_request_guard_rejects_duplicates_and_recovers_after_release():
    request = SimpleNamespace(session_hash="session-a")
    revision = app._mark_selection_changed(request)

    token = app._claim_request(request, revision)

    assert token
    assert app._claim_request(request, revision) is None
    assert app._request_is_current(request, token, revision)

    newer_revision = app._mark_selection_changed(request)
    assert not app._request_is_current(request, token, revision)

    app._release_request(request, token)
    replacement = app._claim_request(request, newer_revision)
    assert replacement
    app._release_request(request, replacement)
    

def test_analysis_event_does_not_make_input_controls_pending_outputs():
    demo = app.build_demo()
    components = {component["id"]: component for component in demo.config["components"]}
    analysis_event = next(
        dependency
        for dependency in demo.config["dependencies"]
        if any(
            event == "click"
            and components[component_id].get("props", {}).get("elem_id")
            == "analyse-button"
            for component_id, event in dependency["targets"]
        )
    )
    control_ids = {
        component_id
        for component_id, component in components.items()
        if component.get("props", {}).get("elem_id")
        in {"analyse-button", "video-upload-button"}
        or component.get("props", {}).get("label")
        in {"Lift", "Declared camera position", "Pose Landmarker model"}
    }

    assert control_ids.isdisjoint(analysis_event["outputs"])
    assert "concurrency_limit" not in analysis_event


def test_setting_invalidation_uses_user_input_events():
    demo = app.build_demo()
    components = {component["id"]: component for component in demo.config["components"]}
    setting_events = {
        components[component_id].get("props", {}).get("label"): event
        for dependency in demo.config["dependencies"]
        for component_id, event in dependency["targets"]
        if components[component_id].get("props", {}).get("label")
        in {"Lift", "Declared camera position", "Pose Landmarker model"}
    }

    assert setting_events == {
        "Lift": "input",
        "Declared camera position": "input",
        "Pose Landmarker model": "input",
    }


def test_build_demo_uses_light_theme_for_dark_mode_tokens():
    theme = app.LIGHT_THEME

    assert theme.body_background_fill == "#fbfbfa"
    assert theme.body_background_fill_dark == "#fbfbfa"
    assert theme.body_text_color == "#20242a"
    assert theme.body_text_color_dark == "#20242a"
    assert theme.block_background_fill_dark == "#ffffff"
    assert theme.input_background_fill_dark == "#ffffff"


def test_custom_css_forces_light_color_scheme():
    assert "color-scheme: light !important;" in app.CUSTOM_CSS
    assert ".gradio-container input" in app.CUSTOM_CSS
    assert "background: #ffffff !important;" in app.CUSTOM_CSS


def test_build_demo_raises_csv_field_limit_for_large_flagged_payloads():
    original_limit = csv.field_size_limit()
    csv.field_size_limit(1024)

    try:
        demo = app.build_demo()

        assert demo is not None
        assert csv.field_size_limit() > 131072
    finally:
        csv.field_size_limit(original_limit)


def test_feedback_markdown_does_not_expose_internal_finding_ids():
    markdown = app._feedback_markdown(
        {
            "overall_assessment": "Your squat reached likely sufficient depth. Your hips rose before your shoulders early in the ascent.",
            "priority_cues": [
                {
                    "rank": 1,
                    "title": "Keep chest and hips rising together",
                    "what_was_observed": "Your hips rose ahead of your shoulders.",
                    "why_focus_on_this": "This can make the ascent less coordinated.",
                    "next_rep_cue": "Chest and hips together. Drive your torso upward as you stand.",
                    "practice_task": "Use a controlled, repeatable load and record a short side-view set.",
                    "success_check": "Your shoulders should begin rising with your hips.",
                    "confidence": "medium",
                }
            ],
            "positive_observations": ["The squat reached likely sufficient depth."],
            "neutral_context": [
                "The long ascent is not automatically a problem on its own."
            ],
            "limitations": [
                "This is a 2D video-based estimate and not a competition judging decision."
            ],
        }
    )

    assert "hip_first_ascent" not in markdown
    assert "torso_lean_change" not in markdown
    assert "prolonged_ascent_or_sticking_region" not in markdown
    assert "What happened" in markdown
    assert "Next rep cue" in markdown
    assert "Technique practice" in markdown
    assert "What a better rep should look like" in markdown


def test_local_model_debug_summary_uses_current_runtime_fields():
    config = app.LocalModelConfig()

    assert app._llm_debug_summary(
        {
            "generation_mode": "fallback",
            "fallback_reason_code": "local_model_unavailable",
            "fallback_reason_detail": None,
            "local_model_status": None,
        },
        config,
    ) == {
        "Configured model": config.model,
        "Model path": config.model_path,
        "Generation mode": "fallback",
        "Fallback reason code": "local_model_unavailable",
        "Fallback reason detail": None,
        "Local model status": None,
    }


def test_llm_status_wording_for_failure_modes():
    from powerlifting_coach.llm_feedback import _status_for_reason

    config = app.LocalModelConfig(model_path="models/llama.gguf")
    assert (
        _status_for_reason("local_model_unavailable", config)
        == "Using deterministic feedback: local language model unavailable"
    )
    assert (
        _status_for_reason("local_model_missing", config)
        == "Using deterministic feedback: local language model is missing or invalid; run python -m powerlifting_coach.setup"
    )
    assert (
        _status_for_reason("local_model_schema_validation_failed", config)
        == "Using deterministic feedback: local response did not meet the required format"
    )
    assert (
        _status_for_reason("local_model_response_quality_rejected", config)
        == "Using deterministic feedback: local response could not be safely refined"
    )


def test_feedback_presentation_distinguishes_rejected_and_unavailable_fallbacks():
    feedback = {
        "overall_assessment": "The bench analysis produced one supported finding.",
        "priority_cues": [],
        "limitations": [],
    }

    rejected = app.render_coaching_feedback_html(
        feedback, "fallback", "local_model_response_quality_rejected"
    )
    unavailable = app.render_coaching_feedback_html(
        feedback, "fallback", "local_model_timeout"
    )

    assert "Generated feedback was rejected by validation" in rejected
    assert "unavailable" not in rejected
    assert "Generated feedback was unavailable" in unavailable
    assert "timed out" in unavailable
    assert rejected.count("The bench analysis produced one supported finding.") == 1
    assert unavailable.count("The bench analysis produced one supported finding.") == 1


def test_feedback_evidence_omits_empty_content_and_preserves_populated_content():
    feedback = {
        "overall_assessment": "Supported bench findings are shown below.",
        "priority_cues": [
            {
                "rank": 1,
                "title": "Supported cue",
                "what_was_observed": "",
                "why_focus_on_this": "",
                "next_rep_cue": "Repeat the same supported view.",
                "evidence_and_limits": {
                    "detected_pattern": "The terminal elbow position was observed.",
                    "interpretation": "Compare this evidence across repetitions.",
                    "references": [
                        {
                            "short_reference": "Project source",
                            "page_or_section": "Bench press",
                            "source_type": "project_interpretation",
                        }
                    ],
                    "important_limits": ["Do not infer weakness."],
                },
            },
            {
                "rank": 2,
                "title": "Empty evidence",
                "what_was_observed": "",
                "why_focus_on_this": "",
                "next_rep_cue": "Keep the setup repeatable.",
                "evidence_and_limits": {
                    "detected_pattern": "",
                    "interpretation": None,
                    "references": [],
                    "important_limits": [""],
                },
            },
        ],
        "limitations": [
            "bench_wrist_trajectory_2d was withheld because its required evidence was unavailable."
        ],
    }

    rendered = app.render_coaching_feedback_html(feedback)

    assert rendered.count("Evidence and limits") == 1
    assert "The terminal elbow position was observed." in rendered
    assert "Project source" in rendered
    assert "This video does not establish weakness." in rendered
    assert "Wrist trajectory was withheld" in rendered
    assert "bench_wrist_trajectory_2d" not in rendered
    assert "<li></li>" not in rendered
