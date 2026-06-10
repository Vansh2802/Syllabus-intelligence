"""OpenAI-backed curriculum analysis for a single chapter."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any


ALLOWED_SKILL_DOMAINS = [
    "Quantitative Reasoning",
    "Scientific Reasoning",
    "Computational Thinking",
    "Analytical Thinking",
    "Communication",
    "Research Skills",
    "Design Thinking",
    "Problem Solving",
]

ALLOWED_DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]

_OPENAI_QUOTA_EXHAUSTED = False

DEFAULT_SUBJECT_SKILL_DOMAINS = {
    "mathematics": "Quantitative Reasoning",
    "math": "Quantitative Reasoning",
    "science": "Scientific Reasoning",
    "physics": "Scientific Reasoning",
    "chemistry": "Scientific Reasoning",
    "biology": "Scientific Reasoning",
    "computer science": "Computational Thinking",
    "cs": "Computational Thinking",
    "informatics practices": "Computational Thinking",
    "social science": "Research Skills",
    "history": "Research Skills",
    "geography": "Analytical Thinking",
    "economics": "Analytical Thinking",
    "english": "Communication",
    "hindi": "Communication",
    "sanskrit": "Communication",
}

SPECIAL_CORE_CONCEPTS = {
    "chemistry": {
        "surface chemistry": "Adsorption and Colloids",
        "chemical equilibrium": "Reaction Equilibrium",
        "solutions": "Solution Properties",
        "electrochemistry": "Redox and Electrochemical Cells",
        "chemical kinetics": "Reaction Rates",
        "qualitative analysis": "Ionic Identification",
        "preparation of inorganic": "Inorganic Synthesis",
        "organic chemistry": "Organic Reaction Mechanisms",
        "coordination compounds": "Coordination Chemistry",
    },
    "physics": {
        "laws of motion": "Force Dynamics",
        "motion": "Kinematics",
        "physical world": "Physical Quantities and Units",
        "units and measurements": "Physical Quantities and Units",
        "motion in a straight line": "Kinematics in One Dimension",
        "motion in a plane": "Two-Dimensional Motion",
        "work, energy and power": "Work-Energy Principles",
        "gravitation": "Gravitational Forces",
        "thermodynamics": "Heat and Energy Transfer",
        "waves": "Wave Motion and Properties",
        "oscillations": "Simple Harmonic Motion",
        "electrostatics": "Electric Fields and Potential",
        "current electricity": "Electric Circuits",
        "magnetism": "Magnetic Fields and Forces",
        "optics": "Light and Optical Systems",
        "dual nature of radiation and matter": "Wave-Particle Duality",
        "atoms": "Atomic Structure",
        "nuclei": "Nuclear Physics",
        "semiconductor": "Semiconductor Devices",
        "sound": "Wave Propagation",
    },
    "mathematics": {
        "number systems": "Real Number Systems",
        "number system": "Real Number Systems",
        "real numbers": "Number Systems and Irrationals",
        "algebra": "Algebraic Expressions and Equations",
        "geometry": "Axioms, Theorems, and Congruence",
        "trigonometry": "Trigonometric Ratios and Identities",
        "mensuration": "Area, Surface Area, and Volume",
        "calculus": "Differential and Integral Calculus",
        "probability": "Probability and Random Events",
        "linear programming": "Optimization and Constraints",
        "vectors": "Vector Algebra and Geometry",
        "relations and functions": "Relations and Functional Mapping",
        "sets and functions": "Set Theory and Functions",
        "polynomials": "Polynomial Expressions and Roots",
        "coordinate geometry": "Coordinate Plane Geometry",
        "geometry": "Axioms, Theorems, and Congruence",
        "mensuration": "Area, Surface Area, and Volume",
        "statistics and probability": "Data Interpretation and Uncertainty",
        "quadratic equations": "Quadratic Equation Solving",
        "trigonometry": "Trigonometric Ratios and Identities",
        "linear equations in two variables": "Linear Relations",
        "sequence and series": "Patterns and Progressions",
    },
    "computer science": {
        "sql": "Database Querying",
        "database": "Database Design and Querying",
        "programming": "Algorithmic Programming",
        "python": "Python Programming",
    },
    "science": {
        "cell": "Cell Structure and Function",
        "tissues": "Plant and Animal Tissues",
        "reproduction": "Reproduction and Continuity of Life",
        "motion": "Motion and Force",
        "sound": "Wave Motion and Acoustics",
        "matter": "Matter and Its Properties",
        "number systems": "Real Number Systems",
        "real numbers": "Number Systems and Irrationals",
        "polynomials": "Polynomial Expressions and Roots",
        "coordinate geometry": "Coordinate Plane Geometry",
        "geometry": "Axioms, Theorems, and Congruence",
        "mensuration": "Area, Surface Area, and Volume",
        "statistics and probability": "Data Interpretation and Uncertainty",
        "quadratic equations": "Quadratic Equation Solving",
        "trigonometry": "Trigonometric Ratios and Identities",
        "linear equations in two variables": "Linear Relations",
        "algebra": "Algebraic Expressions and Equations",
        "heights and distances": "Trigonometric Applications",
        "sets and functions": "Sets, Relations, and Functions",
        "calculus": "Limits, Derivatives, and Integration",
        "vectors": "Vector Algebra and Geometry",
        "probability": "Probability and Random Events",
        "linear programming": "Optimization and Constraints",
    },
    "social science": {
        "nationalism": "Nationalism and Modern States",
        "resources": "Resource Distribution and Management",
        "economics": "Economic Systems and Development",
        "history": "Historical Inquiry and Change",
        "geography": "Spatial Analysis and Environment",
    },
}


@dataclass(slots=True)
class CurriculumAnalysis:
    """Structured output for one chapter."""

    core_concept: str
    skill_domain: str
    difficulty: str
    summary: str


def _clean_text(text: str, limit: int = 5000) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _normalize_subject(subject: str) -> str:
    lowered = subject.lower().strip()
    ordered_keys = sorted(DEFAULT_SUBJECT_SKILL_DOMAINS, key=len, reverse=True)
    for key in ordered_keys:
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered):
            return key
    return lowered


def _chapter_key(chapter_name: str) -> str:
    lowered = re.sub(r"[^a-z0-9 ]+", " ", chapter_name.lower())
    lowered = re.sub(r"\s+", " ", lowered).strip()
    lowered = re.sub(r"^(chapter|unit|lesson|topic|section)\s*(?:\d+[a-z]?|[ivxlcdm]+)?\s*", "", lowered)
    return lowered


def _infer_core_concept(subject: str, chapter_name: str, chapter_text: str) -> str:
    chapter_key = _chapter_key(chapter_name)
    combined = f"{chapter_name} {chapter_text}".lower()
    subject_key = _normalize_subject(subject)

    mappings = SPECIAL_CORE_CONCEPTS.get(subject_key, {})
    for pattern, concept in sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True):
        if pattern in chapter_key:
            return concept

    for key in SPECIAL_CORE_CONCEPTS:
        mappings = SPECIAL_CORE_CONCEPTS.get(key, {})
        for pattern, concept in sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True):
            if pattern in chapter_key:
                return concept

    keyword_concepts = [
        ("database", "Database Querying"),
        ("query", "Database Querying"),
        ("programming", "Programming Concepts"),
        ("linear programming", "Optimization and Constraints"),
        ("algorithm", "Algorithmic Thinking"),
        ("equation", "Equation Solving"),
        ("probability", "Probability and Uncertainty"),
        ("statistics", "Data Analysis"),
        ("proof", "Logical Proofs"),
        ("analysis", "Analytical Reasoning"),
        ("observation", "Observation and Inquiry"),
        ("research", "Research Methods"),
        ("communication", "Communication Skills"),
        ("history", "Historical Inquiry"),
        ("economics", "Economic Reasoning"),
        ("geography", "Spatial Analysis"),
        ("motion", "Motion and Force"),
        ("energy", "Energy Transfer"),
        ("cells", "Cell Structure and Function"),
        ("tissues", "Tissue Organization"),
    ]
    for keyword, concept in keyword_concepts:
        if keyword in chapter_key:
            return concept if concept.lower() != chapter_key else f"{concept} Concepts"

    topic_hints = [
        ("adsorption", "Adsorption and Colloids"),
        ("colloid", "Adsorption and Colloids"),
        ("equilibrium", "Reaction Equilibrium"),
        ("electromagnet", "Electromagnetic Phenomena"),
        ("electrostatic", "Electric Fields and Potential"),
        ("trigonometr", "Trigonometric Ratios"),
        ("polynomial", "Polynomial Expressions"),
        ("coordinate", "Coordinate Plane Geometry"),
        ("statistics", "Data Analysis"),
        ("probability", "Probability and Uncertainty"),
        ("reproduction", "Reproduction and Life Cycles"),
        ("tissue", "Tissue Organization"),
        ("nationalism", "Nationalism and Identity"),
        ("federalism", "Federal Governance"),
        ("globalisation", "Global Economic Integration"),
        ("democracy", "Democratic Institutions"),
        ("resources", "Resource Management"),
    ]
    for keyword, concept in topic_hints:
        if keyword in combined and concept.lower() != chapter_key:
            return concept

    words = [word for word in re.findall(r"[A-Za-z]{4,}", chapter_text[:600]) if word.lower() not in chapter_key.split()]
    if words:
        return f"{words[0].title()} Concepts"

    return "Curriculum Concepts"


def _infer_skill_domain(subject: str, chapter_name: str, chapter_text: str) -> str:
    normalized_subject = _normalize_subject(subject)
    combined = f"{normalized_subject} {chapter_name} {chapter_text}".lower()
    if "social science" in combined:
        return "Research Skills" if any(token in combined for token in ["source", "survey", "case study", "chronology", "history", "map", "resources", "economics", "geography"]) else "Analytical Thinking"
    if "computer science" in combined:
        return "Computational Thinking"
    for key, domain in DEFAULT_SUBJECT_SKILL_DOMAINS.items():
        if key in {"science", "social science", "computer science"}:
            continue
        if key in combined:
            return domain

    if any(token in combined for token in ["code", "program", "algorithm", "data structure", "software", "sql", "database"]):
        return "Computational Thinking"
    if any(token in combined for token in ["experiment", "observation", "chem", "physics", "bio", "motion", "reaction"]):
        return "Scientific Reasoning"
    if any(token in combined for token in ["proof", "equation", "graph", "statistics", "probability", "numerical"]):
        return "Quantitative Reasoning"
    if any(token in combined for token in ["analyze", "analysis", "interpret", "compare", "evaluate"]):
        return "Analytical Thinking"
    if any(token in combined for token in ["write", "speaking", "speech", "essay", "letter", "grammar"]):
        return "Communication"
    if any(token in combined for token in ["research", "source", "case study", "investigation", "survey"]):
        return "Research Skills"
    if any(token in combined for token in ["project", "design", "prototype", "create", "innovation"]):
        return "Design Thinking"
    return "Problem Solving"


def _infer_difficulty(chapter_name: str, chapter_text: str) -> str:
    combined = f"{chapter_name} {chapter_text}".lower()
    advanced_markers = ["calculus", "electromagnetic", "coordination", "kinetics", "thermodynamics", "probability", "integration", "derivative"]
    beginner_markers = ["introduction", "basics", "fundamentals", "overview", "simple", "basic"]

    if any(marker in combined for marker in advanced_markers) or len(chapter_text.split()) > 1200:
        return "Advanced"
    if any(marker in combined for marker in beginner_markers) or len(chapter_text.split()) < 350:
        return "Beginner"
    return "Intermediate"


def _infer_summary(core_concept: str, subject: str, difficulty: str) -> str:
    summary = f"Explores {core_concept} through {subject or 'curriculum'} study."
    words = summary.split()
    return " ".join(words[:15]).rstrip(".") + "."


def _validate_analysis(candidate: dict[str, Any], subject: str, chapter_name: str, chapter_text: str) -> CurriculumAnalysis | None:
    core_concept = str(candidate.get("core_concept", candidate.get("Core_Concept", ""))).strip()
    skill_domain = str(candidate.get("skill_domain", candidate.get("Skill_Domain", ""))).strip()
    difficulty = str(candidate.get("difficulty", candidate.get("Difficulty", ""))).strip()
    summary = str(candidate.get("summary", candidate.get("Summary", ""))).strip()

    if not core_concept:
        return None
    if skill_domain not in ALLOWED_SKILL_DOMAINS:
        return None
    if difficulty not in ALLOWED_DIFFICULTIES:
        return None
    if not summary:
        return None

    cleaned_chapter = _chapter_key(chapter_name)
    if core_concept.lower() == cleaned_chapter or core_concept.lower() == chapter_name.strip().lower():
        return None

    summary = " ".join(summary.split())
    if len(summary.split()) > 15:
        summary = " ".join(summary.split()[:15]).rstrip(".") + "."

    return CurriculumAnalysis(core_concept=core_concept, skill_domain=skill_domain, difficulty=difficulty, summary=summary)


def _fallback_analysis(subject: str, chapter_name: str, chapter_text: str) -> CurriculumAnalysis:
    core_concept = _infer_core_concept(subject, chapter_name, chapter_text)
    chapter_key = _chapter_key(chapter_name)
    if core_concept.lower() == chapter_key or core_concept.lower() == chapter_name.lower().strip():
        core_concept = _infer_core_concept(subject, chapter_name, chapter_text + " topics concepts principles applications")
    skill_domain = _infer_skill_domain(subject, chapter_name, chapter_text)
    difficulty = _infer_difficulty(chapter_name, chapter_text)
    summary = _infer_summary(core_concept, subject, difficulty)
    return CurriculumAnalysis(core_concept=core_concept, skill_domain=skill_domain, difficulty=difficulty, summary=summary)


def _parse_json_response(payload: str) -> dict[str, Any] | None:
    text = payload.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _openai_client() -> Any:
    """Create an OpenAI client when credentials are present."""

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except Exception as exc:  # pragma: no cover - import/runtime environment dependent
        logging.warning("OpenAI client unavailable: %s", exc)
        return None


def _build_analysis_prompt(chapter_name: str, chapter_text: str, subject: str, class_label: str) -> str:
    return (
        "You are a CBSE curriculum expert.\n"
        "Analyze the chapter.\n"
        "Return ONLY valid JSON.\n\n"
        "{\n"
        '"core_concept": "",\n'
        '"skill_domain": "",\n'
        '"difficulty": "",\n'
        '"summary": ""\n'
        "}\n\n"
        "Rules:\n"
        "Core concept must NOT repeat the chapter title.\n"
        "Infer the actual educational concept.\n\n"
        "Examples:\n"
        "Chapter: Surface Chemistry\n"
        "Core Concept: Adsorption and Colloids\n\n"
        "Chapter: Laws of Motion\n"
        "Core Concept: Force Dynamics\n\n"
        "Chapter: SQL\n"
        "Core Concept: Database Querying\n\n"
        "Use only:\n"
        "Quantitative Reasoning\n"
        "Scientific Reasoning\n"
        "Computational Thinking\n"
        "Analytical Thinking\n"
        "Communication\n"
        "Research Skills\n"
        "Design Thinking\n"
        "Problem Solving\n\n"
        "Difficulty:\n"
        "Beginner\n"
        "Intermediate\n"
        "Advanced\n\n"
        "Summary:\n"
        "Maximum 15 words.\n\n"
        "Return JSON only.\n\n"
        f"Class: {class_label or 'Unknown'}\n"
        f"Subject: {subject or 'Unknown'}\n"
        f"Chapter: {chapter_name}\n"
        f"Description: {_clean_text(chapter_text)}"
    )


def analyze_chapter(chapter_name: str, chapter_text: str, subject: str, class_label: str) -> CurriculumAnalysis:
    """Analyze a chapter into one core concept, one skill domain, one difficulty, and one summary."""

    global _OPENAI_QUOTA_EXHAUSTED

    if _OPENAI_QUOTA_EXHAUSTED:
        return _fallback_analysis(subject, chapter_name, chapter_text)

    prompt = _build_analysis_prompt(chapter_name, chapter_text, subject, class_label)
    client = _openai_client()
    if client is None:
        return _fallback_analysis(subject, chapter_name, chapter_text)

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a CBSE curriculum expert. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = _parse_json_response(content)
        if parsed is not None:
            validated = _validate_analysis(parsed, subject, chapter_name, chapter_text)
            if validated is not None:
                return validated
        logging.warning("Invalid OpenAI response for chapter '%s'; using fallback analysis.", chapter_name)
    except Exception as exc:  # pragma: no cover - depends on API/network conditions
        error_text = str(exc).lower()
        if "insufficient_quota" in error_text:
            _OPENAI_QUOTA_EXHAUSTED = True
            logging.warning("OpenAI quota exhausted; using fallback analysis for remaining chapters.")
        else:
            logging.warning("OpenAI analysis failed for chapter '%s': %s", chapter_name, exc)

    return _fallback_analysis(subject, chapter_name, chapter_text)