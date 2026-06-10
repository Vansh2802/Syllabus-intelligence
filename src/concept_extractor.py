"""Concept generation helpers with optional Gemini support."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Iterable, List


SUBJECT_HINTS = {
    "mathematics": [
        "Number Systems",
        "Algebraic Thinking",
        "Geometry",
        "Mensuration",
        "Statistics",
        "Probability",
        "Ratio and Proportion",
        "Coordinate Geometry",
        "Trigonometry",
        "Real Numbers",
    ],
    "science": [
        "Scientific Method",
        "Observation",
        "Measurement",
        "Cause and Effect",
        "Experimentation",
        "Classification",
        "Energy and Force",
        "Matter and Materials",
        "Life Processes",
        "Earth Systems",
    ],
    "physics": [
        "Motion",
        "Force",
        "Energy",
        "Wave Behavior",
        "Electricity",
        "Magnetism",
        "Optics",
    ],
    "chemistry": [
        "Atomic Structure",
        "Chemical Reactions",
        "Periodic Classification",
        "Solutions",
        "Acids and Bases",
        "Metals and Non-metals",
    ],
    "biology": [
        "Cell Structure",
        "Nutrition",
        "Respiration",
        "Reproduction",
        "Genetics",
        "Ecology",
    ],
    "computer science": [
        "Algorithms",
        "Data Representation",
        "Programming Concepts",
        "Problem Decomposition",
        "Logic Building",
        "Computational Thinking",
    ],
    "social science": [
        "Historical Understanding",
        "Geographic Literacy",
        "Civic Concepts",
        "Economic Concepts",
        "Map Skills",
        "Source Analysis",
    ],
    "english": [
        "Reading Comprehension",
        "Vocabulary",
        "Grammar",
        "Writing Skills",
        "Literary Appreciation",
        "Communication",
    ],
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "chapter",
    "unit",
    "topic",
    "lesson",
    "class",
    "syllabus",
    "pdf",
    "theory",
    "introduction",
    "curriculum",
    "learn",
    "students",
    "student",
    "concept",
    "concepts",
    "topics",
    "chapter",
}


def _gemini_available() -> bool:
    """Return True when Gemini credentials and package support are present."""

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return False
    return True


def _generate_with_gemini(prompt: str) -> List[str]:
    """Ask Gemini for a compact JSON list and parse the response."""

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return []

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = getattr(response, "text", "") or ""
        parsed = _parse_list_response(text)
        return parsed
    except Exception:
        return []


def _parse_list_response(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    try:
        if cleaned.startswith("["):
            value = json.loads(cleaned)
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
    except Exception:
        pass

    items: List[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().lstrip("-•*").strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\).:\-]\s*", "", line).strip()
        if line:
            items.append(line)
    return items


def _word_tokens(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z\-]+", text) if token.lower() not in STOPWORDS]


def _rule_based_concepts(chapter_name: str, chapter_text: str, subject: str) -> List[str]:
    """Generate 3-5 concepts using subject-aware keyword heuristics."""

    subject_key = subject.lower().strip()
    candidates = list(SUBJECT_HINTS.get(subject_key, []))

    combined_text = f"{chapter_name}\n{chapter_text}".lower()
    scored: list[tuple[int, str]] = []
    for concept in candidates:
        concept_tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z\-]+", concept)]
        score = sum(2 for token in concept_tokens if token in combined_text)
        if score:
            scored.append((score, concept))

    word_counts = Counter(_word_tokens(combined_text))
    for token, _count in word_counts.most_common(12):
        title = token.replace("-", " ").title()
        if len(title) < 4:
            continue
        if title.lower() in {item.lower() for item in candidates}:
            continue
        if title in {item for _, item in scored}:
            continue
        if title.lower() in STOPWORDS:
            continue
        scored.append((1, title))

    scored.sort(key=lambda item: (-item[0], item[1]))
    concepts = [concept for _, concept in scored]

    if len(concepts) < 3:
        concepts.extend(candidates)

    deduped: List[str] = []
    seen = set()
    for concept in concepts:
        normalized = concept.lower().strip()
        if normalized and normalized not in seen:
            deduped.append(concept)
            seen.add(normalized)
        if len(deduped) >= 5:
            break

    if len(deduped) < 3:
        fallback = [chapter_name, "Core Understanding", "Applied Practice"]
        for item in fallback:
            normalized = item.lower().strip()
            if normalized not in seen:
                deduped.append(item)
                seen.add(normalized)
            if len(deduped) >= 3:
                break

    return deduped[:5]


def extract_concepts(chapter_name: str, chapter_text: str, subject: str, class_label: str = "") -> List[str]:
    """Generate 3-5 major concepts for a chapter."""

    prompt = (
        "You are generating curriculum concepts for a CBSE syllabus dataset. "
        "Return only a JSON array of 3 to 5 concise concept names. "
        f"Subject: {subject or 'General'}. Class: {class_label or 'Unknown'}. "
        f"Chapter: {chapter_name}. "
        f"Source text:\n{chapter_text[:5000]}"
    )

    if _gemini_available():
        concepts = _generate_with_gemini(prompt)
        if 3 <= len(concepts) <= 5:
            return concepts[:5]

    return _rule_based_concepts(chapter_name, chapter_text, subject)
