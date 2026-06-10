"""Skill mapping helpers with optional Gemini support."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import List


SKILL_KEYWORDS = {
    "Problem Solving": ["problem", "solve", "solve", "equation", "task", "exercise", "application", "calculat"],
    "Logical Thinking": ["logic", "reason", "deduce", "infer", "pattern", "sequence", "proof"],
    "Analytical Thinking": ["analyze", "analysis", "compare", "interpret", "differentiate", "examine"],
    "Critical Thinking": ["evaluate", "critique", "justify", "explain", "assess", "argument", "evidence"],
    "Quantitative Reasoning": ["number", "measure", "calculate", "quantit", "data", "graph", "formula"],
    "Scientific Reasoning": ["experiment", "hypothesis", "observation", "scientific", "cause", "effect", "investigate"],
    "Computational Thinking": ["algorithm", "step", "code", "decompose", "pattern", "automation", "abstraction"],
    "Communication Skills": ["describe", "present", "explain", "report", "discuss", "communicate", "write"],
    "Observation Skills": ["observe", "measurement", "identify", "inspect", "record", "notice", "compare"],
    "Research Skills": ["research", "study", "collect", "source", "evidence", "inquiry", "investigate"],
}

DEFAULT_SKILLS = [
    "Problem Solving",
    "Logical Thinking",
    "Analytical Thinking",
    "Critical Thinking",
    "Quantitative Reasoning",
    "Scientific Reasoning",
    "Computational Thinking",
    "Communication Skills",
    "Observation Skills",
    "Research Skills",
]

SUBJECT_BOOSTS = {
    "mathematics": ["Quantitative Reasoning", "Problem Solving", "Logical Thinking"],
    "science": ["Scientific Reasoning", "Observation Skills", "Analytical Thinking"],
    "physics": ["Scientific Reasoning", "Quantitative Reasoning", "Problem Solving"],
    "chemistry": ["Scientific Reasoning", "Analytical Thinking", "Observation Skills"],
    "biology": ["Scientific Reasoning", "Observation Skills", "Research Skills"],
    "computer science": ["Computational Thinking", "Logical Thinking", "Problem Solving"],
    "social science": ["Critical Thinking", "Communication Skills", "Research Skills"],
    "english": ["Communication Skills", "Critical Thinking", "Analytical Thinking"],
}


def _gemini_available() -> bool:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return False
    return True


def _generate_with_gemini(prompt: str) -> List[str]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return []

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = getattr(response, "text", "") or ""
        return _parse_list_response(text)
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


def _score_skills(chapter_name: str, chapter_text: str, concepts: List[str], subject: str) -> List[str]:
    combined_text = f"{chapter_name}\n{chapter_text}\n{' '.join(concepts)}".lower()
    scores = Counter()

    for skill, keywords in SKILL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined_text:
                scores[skill] += 2 if len(keyword) > 4 else 1

    subject_key = subject.lower().strip()
    for skill in SUBJECT_BOOSTS.get(subject_key, []):
        scores[skill] += 2

    if not scores:
        scores.update({skill: 1 for skill in DEFAULT_SKILLS})

    ordered = [skill for skill, _ in scores.most_common()]
    if len(ordered) < 3:
        for skill in DEFAULT_SKILLS:
            if skill not in ordered:
                ordered.append(skill)
            if len(ordered) >= 5:
                break

    deduped: List[str] = []
    for skill in ordered:
        if skill not in deduped:
            deduped.append(skill)
        if len(deduped) >= 5:
            break

    return deduped[:5]


def map_skills(chapter_name: str, chapter_text: str, concepts: List[str], subject: str, class_label: str = "") -> List[str]:
    """Generate 3-5 skills relevant to the chapter and its concepts."""

    prompt = (
        "You are generating educational skills for a CBSE curriculum dataset. "
        "Return only a JSON array of 3 to 5 skill names from this list: "
        f"{', '.join(DEFAULT_SKILLS)}. "
        f"Subject: {subject or 'General'}. Class: {class_label or 'Unknown'}. "
        f"Chapter: {chapter_name}. Concepts: {', '.join(concepts)}. "
        f"Source text:\n{chapter_text[:5000]}"
    )

    if _gemini_available():
        skills = _generate_with_gemini(prompt)
        if 3 <= len(skills) <= 5:
            return skills[:5]

    return _score_skills(chapter_name, chapter_text, concepts, subject)
