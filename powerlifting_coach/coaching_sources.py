"""Structured source registry for bounded coaching interpretations."""

from typing import Literal, TypedDict

SourceType = Literal["peer_reviewed_article", "coaching_website", "project_rule"]


class SourceRecord(TypedDict):
    source_id: str
    short_reference: str
    source_type: SourceType
    year: int | None
    page_or_section: str
    claim_ids: list[str]
    supported_summary: str
    limitations: list[str]
    citation_complete: bool
    bibliography_note: str


SOURCES: dict[str, SourceRecord] = {
    "SRC_GOMO_VAN_DEN_TILLAAR_BENCH_GRIP": {
        "source_id": "SRC_GOMO_VAN_DEN_TILLAAR_BENCH_GRIP",
        "short_reference": "Gomo & van den Tillaar",
        "source_type": "peer_reviewed_article",
        "year": None,
        "page_or_section": "Bench-press grip width and elbow geometry",
        "claim_ids": ["008"],
        "supported_summary": "Bench-press elbow geometry varies with grip width, so a projected elbow angle is descriptive rather than a universal correctness threshold.",
        "limitations": [
            "The project measurement is a selected-side 2D projection.",
            "Do not prescribe a universal elbow angle.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add the project's verified full citation later; do not infer missing details.",
    },
    "SRC_LARSEN_BENCH_KINEMATICS": {
        "source_id": "SRC_LARSEN_BENCH_KINEMATICS",
        "short_reference": "Larsen et al.",
        "source_type": "peer_reviewed_article",
        "year": None,
        "page_or_section": "Bench-press kinematics and bar trajectory",
        "claim_ids": ["009"],
        "supported_summary": "Bench-press kinematics and trajectory can be described across a repetition, while this project's wrist landmark remains only a hand/bar-end proxy.",
        "limitations": [
            "Wrist tracking is not validated direct barbell tracking.",
            "Do not infer an optimal bar path or competition outcome.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add the project's verified full citation later; do not infer missing details.",
    },
    "SRC_CONVENTIONAL_DEADLIFT_BIOMECHANICS": {
        "source_id": "SRC_CONVENTIONAL_DEADLIFT_BIOMECHANICS",
        "short_reference": "Conventional deadlift biomechanics research",
        "source_type": "peer_reviewed_article",
        "year": None,
        "page_or_section": "Start and early-pull joint geometry",
        "claim_ids": ["010"],
        "supported_summary": "Phase-specific projected joint geometry can describe a conventional deadlift start and early pull.",
        "limitations": [
            "No universal ideal setup or early-coordination threshold is established here.",
            "Conventional and sumo deadlift geometry differs.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add the exact already-established project reference before academic submission; no bibliography details are inferred here.",
    },
    "SRC_DEADLIFT_PHASE_KINEMATICS": {
        "source_id": "SRC_DEADLIFT_PHASE_KINEMATICS",
        "short_reference": "Deadlift phase-kinematics research",
        "source_type": "peer_reviewed_article",
        "year": None,
        "page_or_section": "Lift-off, knee passing, and completion",
        "claim_ids": ["011"],
        "supported_summary": "Lift-off, knee-passing, and completion boundaries support descriptive comparison of joint-angle changes during a deadlift.",
        "limitations": [
            "The project's boundaries use a wrist hand/bar-end proxy.",
            "Completion geometry is not competition-standard lockout judging.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add the exact already-established project reference before academic submission; no bibliography details are inferred here.",
    },
    "SRC_GIUSTINO_2024_DEPTH": {
        "source_id": "SRC_GIUSTINO_2024_DEPTH",
        "short_reference": "Giustino et al. (2024)",
        "source_type": "peer_reviewed_article",
        "year": 2024,
        "page_or_section": "Introduction",
        "claim_ids": ["001"],
        "supported_summary": "Squat depth is primarily related to knee range of motion; a full squat requires the thigh at the hip to pass below the knee.",
        "limitations": [
            "This project estimates depth from a side-view 2D video.",
            "Landmark position is not identical to official competition judging.",
            "Do not claim competition-valid depth with certainty.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add full article title, journal, DOI, and full citation from the original paper.",
    },
    "SRC_POWERLIFTING_TECHNIQUE_HIP_ASCENT": {
        "source_id": "SRC_POWERLIFTING_TECHNIQUE_HIP_ASCENT",
        "short_reference": "PowerliftingTechnique.com",
        "source_type": "coaching_website",
        "year": None,
        "page_or_section": "What To Do If The Squat Bar Path Isn't Following The Correct Position",
        "claim_ids": ["002"],
        "supported_summary": "A coaching interpretation of hips rising faster than shoulders during ascent and the resulting increase in forward torso position.",
        "limitations": [
            "This is a coaching website, not peer-reviewed evidence.",
            "Do not infer quadriceps weakness from a single video.",
            "Do not diagnose a 'good morning' squat from one metric alone.",
            "Interpret in context of load, bar position, anthropometry, and experience.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add source URL, publication date if available, and access date later.",
    },
    "SRC_SATO_TORSO_LEAN": {
        "source_id": "SRC_SATO_TORSO_LEAN",
        "short_reference": "Sato et al. (n.d.)",
        "source_type": "peer_reviewed_article",
        "year": None,
        "page_or_section": "Discussion",
        "claim_ids": ["003"],
        "supported_summary": "Forward trunk flexion is associated with hip kinematics, including hip flexion and posterior hip displacement.",
        "limitations": [
            "Torso lean is context-dependent.",
            "Do not assume all forward lean is a fault.",
            "Anthropometry and squat style can change expected torso angle.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add publication year, article title, journal, and DOI from the original paper.",
    },
    "SRC_VAN_DEN_TILLAAR_HELMS_2020": {
        "source_id": "SRC_VAN_DEN_TILLAAR_HELMS_2020",
        "short_reference": "van den Tillaar & Helms (2020)",
        "source_type": "peer_reviewed_article",
        "year": 2020,
        "page_or_section": "Discussion",
        "claim_ids": ["004"],
        "supported_summary": "Increased forward lean can occur near maximal-capacity repetitions and can influence ascent velocity, particularly in the reported context.",
        "limitations": [
            "Do not infer fatigue from one repetition.",
            "Do not generalise high-bar findings directly to low-bar powerlifting squats.",
            "Load and set context are required before making fatigue claims.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add full article title, journal, DOI, and complete citation from the original paper.",
    },
    "SRC_GIUSTINO_2024_STICKING": {
        "source_id": "SRC_GIUSTINO_2024_STICKING",
        "short_reference": "Giustino et al. (2024)",
        "source_type": "peer_reviewed_article",
        "year": 2024,
        "page_or_section": "Discussion",
        "claim_ids": ["005"],
        "supported_summary": "Ascent speed may slow near high loads; small increases in forward trunk tilt and hip flexion may occur around a sticking region.",
        "limitations": [
            "A slow ascent is not automatically poor technique.",
            "Do not infer fatigue, weakness, or excessive effort from video alone.",
            "Interpret in the context of load, tempo, and squat style.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add full article title, journal, DOI, and complete citation from the original paper.",
    },
    "SRC_POWERLIFTING_TECHNIQUE_DESCENT": {
        "source_id": "SRC_POWERLIFTING_TECHNIQUE_DESCENT",
        "short_reference": "PowerliftingTechnique.com",
        "source_type": "coaching_website",
        "year": None,
        "page_or_section": "Tips For Keeping The Squat Bar Path In An Optimal Position",
        "claim_ids": ["006"],
        "supported_summary": "Coaching guidance that hips and knees should begin bending together to support a balanced descent.",
        "limitations": [
            "This is a coaching website, not peer-reviewed evidence.",
            "Do not require identical hip and knee movement for every lifter.",
            "Anthropometry and squat style affect movement strategy.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add source URL, publication date if available, and access date later.",
    },
    "SRC_POWERLIFTING_TECHNIQUE_ANTHROPOMETRY": {
        "source_id": "SRC_POWERLIFTING_TECHNIQUE_ANTHROPOMETRY",
        "short_reference": "PowerliftingTechnique.com",
        "source_type": "coaching_website",
        "year": None,
        "page_or_section": "What To Do If The Squat Bar Path Isn't Following The Correct Position",
        "claim_ids": ["007"],
        "supported_summary": "Body proportions influence natural torso lean and bar path; a perfectly vertical bar path may not be practical for every lifter.",
        "limitations": [
            "This is a coaching website, not peer-reviewed evidence.",
            "Do not use this source as evidence for MediaPipe or 2D-video validity.",
            "Do not classify a lifter as incorrect solely because they lean forward.",
        ],
        "citation_complete": False,
        "bibliography_note": "Add source URL, publication date if available, and access date later.",
    },
}

SOURCE_TYPE_LABELS = {
    "peer_reviewed_article": "Peer-reviewed article",
    "coaching_website": "Coaching website",
    "project_rule": "Project interpretation rule",
}
