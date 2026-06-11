"""CSV export helpers for the curriculum intelligence dataset."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from curriculum_analyzer import ALLOWED_DIFFICULTIES, ALLOWED_SKILL_DOMAINS


LEARNING_STAGES = ("Secondary School", "Higher Secondary", "Engineering Year 1", "Engineering Year 2", "Engineering Year 3", "Engineering Year 4")


def derive_learning_stage(class_label: str) -> str:
    """Map a class label to its educational learning stage."""

    cleaned = str(class_label or "").strip()
    if cleaned in {"Engineering Year 1", "Engineering Year 2", "Engineering Year 3", "Engineering Year 4"}:
        return cleaned
    if cleaned.lower() == "engineering":
        return "Engineering Year 1"
    if cleaned.isdigit():
        level = int(cleaned)
        if level <= 10:
            return "Secondary School"
        if level <= 12:
            return "Higher Secondary"
    return ""


@dataclass(slots=True)
class CurriculumRow:
    """Represents a single row in the output CSV."""

    class_label: str
    subject: str
    chapter: str
    core_concept: str
    skill_domain: str
    difficulty: str
    summary: str


CSV_HEADERS = [
    "Class",
    "Subject",
    "Chapter",
    "Core_Concept",
    "Skill_Domain",
    "Difficulty",
    "Learning_Stage",
    "Summary",
]
MAX_EXPORTED_ROWS = 1000
MIN_EXPORTED_ROWS = 60

INVALID_CLASS_VALUES = {"", "0", "none", "nan", "null", "unknown"}

GENERIC_CORE_CONCEPTS = {
    "chapter concepts",
    "curriculum concepts",
    "unit concepts",
    "marks concepts",
    "subject concepts",
    "note concepts",
    "topics concepts",
    "publisher concepts",
    "application concepts",
    "outcomes concepts",
    "portfolio concepts",
    "competencies concepts",
    "economy concepts",
    "total concepts",
    "following concepts",
    "between concepts",
    "tobe concepts",
    "evaluated concepts",
    "with concepts",
    "similar concepts",
    "fundamental concepts",
    "process concepts",
    "historical concepts",
    "applications concepts",
    "algebraic concepts",
    "derivation concepts",
    "ncert concepts",
}

INVALID_CHAPTER_PATTERNS = (
    r"\b(theory|practical|guidelines?|note|evaluation|assessment|instructions?|annexure|appendix|examination|marking scheme|question paper|learning outcomes|project work)\b",
    r"^c[\s\-]?\d",
    r"^c-\d",
    r"\bmark[\s\-]*mcq",
    r"\bpreparation of any\b",
    r"^class\s*[–-]?\s*(ix|x|xi|xii|9|10|11|12)",
    r"[\√∫𝛼𝛽𝒂𝒙]",
    r"^[ivxlcdm]+[\s:.\-–]*$",
    r"^[\-–]+[ivxlcdm\d]+[\s:.\-–]*$",
)


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _is_valid_class(value: str) -> bool:
    cleaned = value.strip()
    if cleaned in {"Engineering Year 1", "Engineering Year 2", "Engineering Year 3", "Engineering Year 4"}:
        return True
    if cleaned.lower() == "engineering":
        return True
    return cleaned.isdigit() and 1 <= int(cleaned) <= 12


def _is_valid_chapter(value: str) -> bool:
    chapter = _normalize_text(value)
    if len(chapter) < 4:
        return False
    if chapter.isdigit():
        return False
    lowered = chapter.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in INVALID_CHAPTER_PATTERNS):
        return False
    if sum(char.isalpha() for char in lowered) < 3:
        return False
    if sum(char.isalpha() for char in lowered) / max(len(lowered), 1) < 0.4:
        return False
    if chapter.count(",") >= 2:
        return False
    if len(chapter.split()) > 9:
        return False
    return True


def _clean_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.reindex(columns=CSV_HEADERS)

    frame = dataframe.copy()
    for column in CSV_HEADERS:
        if column not in frame.columns:
            frame[column] = ""

    for column in CSV_HEADERS:
        frame[column] = frame[column].map(_normalize_text)

    frame = frame[~frame["Class"].str.lower().isin(INVALID_CLASS_VALUES)]
    frame = frame[frame["Class"].map(_is_valid_class)]
    frame = frame[frame["Subject"].str.len().ge(2)]
    frame = frame[frame["Chapter"].map(_is_valid_chapter)]
    frame = frame[frame["Core_Concept"].str.len().ge(3)]
    frame = frame[frame["Summary"].str.len().ge(5)]
    frame = frame[frame["Skill_Domain"].isin(ALLOWED_SKILL_DOMAINS)]
    frame = frame[frame["Difficulty"].isin(ALLOWED_DIFFICULTIES)]
    frame = frame[~frame["Core_Concept"].str.lower().eq(frame["Chapter"].str.lower())]
    frame = frame[~frame["Core_Concept"].str.lower().isin(GENERIC_CORE_CONCEPTS)]
    frame = frame[~frame["Core_Concept"].str.lower().str.match(r"^[a-z]{3,12} concepts$")]

    frame["Learning_Stage"] = frame["Class"].map(derive_learning_stage)
    frame = frame[frame["Learning_Stage"].isin(LEARNING_STAGES)]

    frame = frame.drop_duplicates(subset=["Class", "Subject", "Chapter", "Core_Concept"], keep="first")
    frame["_quality_score"] = frame.apply(_row_quality_score, axis=1)
    frame = frame[frame["_quality_score"] >= 3]
    frame = frame.sort_values(["_quality_score", "Class", "Subject", "Chapter"], ascending=[False, True, True, True], kind="stable")
    frame = frame.head(MAX_EXPORTED_ROWS)
    frame = frame.drop(columns=["_quality_score"])
    frame = frame.sort_values(["Class", "Subject", "Chapter", "Core_Concept"], kind="stable")
    frame = frame.reset_index(drop=True)
    return frame.reindex(columns=CSV_HEADERS)


def _row_quality_score(row: pd.Series) -> int:
    class_val = str(row.get("Class", ""))
    if "Engineering" in class_val:
        return 5

    chapter = str(row.get("Chapter", "")).strip()
    core_concept = str(row.get("Core_Concept", "")).strip()
    summary = str(row.get("Summary", "")).strip()

    lowered = chapter.lower()
    score = 0

    word_count = len(chapter.split())
    if 1 <= word_count <= 6:
        score += 4
    elif word_count <= 8:
        score += 2
    else:
        score -= 3

    if chapter and chapter[0].isupper():
        score += 1
    if re.search(r"[0-9]", chapter):
        score -= 2
    if re.search(r"[^A-Za-z0-9\s'’&(),\-:;/.]", chapter):
        score -= 2

    bad_leads = (
        "action ",
        "present ",
        "find ",
        "study ",
        "understand ",
        "examine ",
        "define ",
        "identify ",
        "compare ",
        "calculate ",
        "prepare ",
        "write ",
        "classify ",
        "discuss ",
        "explain ",
        "compile ",
        "formulate ",
        "analyze ",
        "analyse ",
        "discuss ",
        "discussions ",
    )
    if lowered.startswith(bad_leads):
        score -= 5

    if lowered.endswith((" of", " and", " in", " to", " for", " with", " on", " at", " by", " from", " the", " a", " an", " or")):
        score -= 5

    bad_phrases = (
        "grand total",
        "question paper",
        "internal assessment",
        "subject enrichment",
        "typology of questions",
        "course structure",
        "learning outcomes",
        "learning outcome",
        "subject code",
        "the student will",
        "the teachers",
        "three hours",
        "one paper",
        "map work",
        "map skill",
        "name of the book",
        "weightage",
        "rubrics",
        "interdisciplinary project",
        "demonstrate knowledge",
        "formulate, analyze",
        "portfolio",
        "investigatory project",
        "multiple assessment",
        "suggested teaching",
        "of interdisciplinary",
        "locating and labeling",
        "type of questions",
        "learning process",
        "time schedule",
        "quiz, debate",
        "participation of the student",
        "examining and breaking",
        "compiling information",
        "iron ore mines",
        "oil fields",
        "coal mines",
        "angle of elevation",
        "reaction between sodium",
    )
    if any(phrase in lowered for phrase in bad_phrases):
        score -= 5

    if any(topic in lowered for topic in ("world", "matter", "motion", "force", "resources", "nationalism", "geometry", "probability", "statistics", "chemistry", "physics", "biology", "database", "programming", "living")):
        score += 1

    if core_concept and core_concept.lower() != lowered:
        score += 1
    if 5 <= len(summary.split()) <= 15:
        score += 1

    return score


def clean_curriculum_rows(rows: Iterable[CurriculumRow]) -> pd.DataFrame:
    """Return a cleaned dataframe ready for export or dashboard display."""

    dataframe = pd.DataFrame(
        [
            {
                "Class": row.class_label,
                "Subject": row.subject,
                "Chapter": row.chapter,
                "Core_Concept": row.core_concept,
                "Skill_Domain": row.skill_domain,
                "Difficulty": row.difficulty,
                "Learning_Stage": derive_learning_stage(row.class_label),
                "Summary": row.summary,
            }
            for row in rows
        ],
        columns=CSV_HEADERS,
    )
    return _clean_frame(dataframe)


def ensure_parent_directory(csv_path: Path) -> None:
    """Create the output directory if it does not exist."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)


def export_dataset(rows: Iterable[CurriculumRow], csv_path: Path) -> Path:
    """Write the dataset to disk as a CSV file and return the path used."""

    ensure_parent_directory(csv_path)
    dataframe = clean_curriculum_rows(rows)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", dir=csv_path.parent, encoding="utf-8", newline="") as handle:
        temp_path = Path(handle.name)
        dataframe.to_csv(handle, index=False)

    try:
        os.replace(temp_path, csv_path)
        return csv_path
    except PermissionError:
        fallback_path = csv_path.with_suffix(".tmp.csv")
        dataframe.to_csv(fallback_path, index=False, encoding="utf-8")
        temp_path.unlink(missing_ok=True)
        return fallback_path


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a curriculum dataset from disk."""

    if not csv_path.exists():
        return pd.DataFrame(columns=CSV_HEADERS)
    return _clean_frame(pd.read_csv(csv_path).fillna(""))
