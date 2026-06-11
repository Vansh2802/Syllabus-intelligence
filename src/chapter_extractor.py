"""Chapter, unit, and topic extraction helpers."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from pdf_extractor import PdfDocument


@dataclass(slots=True)
class ChapterSection:
    """Represents a chapter-like section extracted from a syllabus PDF."""

    title: str
    content: str
    section_type: str


IGNORED_TITLES = {
    "contents",
    "table of contents",
    "syllabus",
    "curriculum",
    "index",
    "acknowledgement",
    "acknowledgements",
    "preface",
    "introduction",
    "course structure",
    "question paper design",
    "question paper blueprint",
    "prescribed books",
    "prescribed text books",
    "prescribed textbooks",
    "internal assessment",
    "evaluation scheme",
    "marks",
    "aims & objective",
    "aims and objective",
    "objectives",
    "curricular goals",
    "curricular goals-cg",
    "list of experiments",
    "annexure",
    "annexure iv",
    "annexure v",
    "section",
    "name",
}

IGNORED_TITLE_PHRASES = (
    "theory",
    "theory (1)",
    "general guidelines",
    "course outline",
    "note",
    "notes",
    "practical",
    "practical work",
    "practical examination",
    "marking scheme",
    "evaluation",
    "assessment",
    "instructions",
    "question paper design",
    "learning outcomes",
    "annexure",
    "appendix",
    "examination",
    "project work",
    "course structure",
    "curricular goals",
    "curriculum goals",
    "content",
    "contents",
    "chapter name",
    "unit name",
    "key concepts",
    "learning standards",
    "competencies",
    "prescribed books",
    "class ix",
    "class x",
    "class xi",
    "class xii",
    "grand total",
    "subject code",
    "internal assessment",
    "subject enrichment",
    "question paper",
    "typology of questions",
    "learning standards",
    "curricular goals",
    "map work",
    "map skill",
    "name of the book",
    "subject wise weightage",
    "weightage to competency",
    "rubrics for",
    "suggested teaching",
    "three hours",
    "one paper",
    "interdisciplinary project",
    "research work collaboration",
)

EXCLUDED_KEYWORDS = (
    "marks",
    "periods",
    "course structure",
    "question paper",
    "prescribed",
    "assessment",
    "annexure",
    "internal",
    "objective",
    "goals",
    "syllabus",
    "books",
    "table",
    "list of experiments",
)

OUTCOME_VERBS = (
    "understands",
    "understand",
    "describes",
    "describe",
    "models",
    "model",
    "proves",
    "prove",
    "visualises",
    "visualizes",
    "learns",
    "learn",
    "specifies",
    "specify",
    "carries out",
    "calculate",
    "calculates",
    "uses",
    "use",
    "identifies",
    "identify",
    "implements",
    "implement",
    "examines",
    "examine",
    "explains",
    "explain",
)

INVALID_TITLE_PATTERNS = (
    r"\b(theory|practical|guidelines?|note|evaluation|assessment|instructions?|annexure|appendix|examination|marking scheme|question paper design|learning outcomes|project work)\b",
    r"\b(student|students)\s+will\s+be\s+able\s+to\b",
    r"\b(course structure|curricular goals?|competencies|prescribed books|contents?|table of contents)\b",
    r"\bsubject\s*code\b",
    r"\b(action of|present and|by making|at point of|find the|the student will|grand total|typology of questions|kept in|theme\b|portfolio|learning outcomes|subject enrichment)\b",
    r"^c[\s\-]?\d+(?:\.\d+)?\b",
    r"^c-\d+(?:\.\d+)?\b",
    r"\bmark[\s\-]*mcq",
    r"\bpreparation of any\b",
    r"\bexperiments? based on\b",
    r"\bvisually challenged\b",
    r"^[\(\[]\s*note\b",
    r"^[\d√∫𝛼𝛽𝒂𝒙%]+[\s,√∫𝛼𝛽𝒙]*$",
    r"^[ivxlcdm]+[\s:.\-–]*$",
    r"^[\-–]+[ivxlcdm\d]+[\s:.\-–]*$",
)

HEADING_PATTERN = re.compile(
    r"^(?:(chapter|unit|lesson|topic|section)\s*(?:no\.?\s*)?(\d+[a-zA-Z]?|[ivxlcdm]+)?[\s:.)-]*)?(.+)$",
    flags=re.IGNORECASE,
)
NUMBERED_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
ROMAN_PREFIX_PATTERN = re.compile(r"^[ivxlcdm]+[\s:.-]+(.+)$", flags=re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_title(title: str) -> str:
    cleaned = _normalize_whitespace(title)
    cleaned = re.sub(r"^[•\-*\u2022]+\s*", "", cleaned)
    cleaned = re.sub(r"^[\-–]+\s*", "", cleaned)
    cleaned = re.sub(r"^[\-–]?\s*(?:\d+|[ivxlcdm]+)\s*[:.)\-–]+\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[ivxlcdm]+\s*[:.)\-–]+\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[A-HJ-Z]\.[\s-]*", "", cleaned)
    cleaned = re.sub(r"^\(?[A-HJ-Z]\)?[\s:.-]+", "", cleaned)
    cleaned = re.sub(r"^(chapter|unit|lesson|topic|section)\s*(?:\d+[a-zA-Z]?|[ivxlcdm]+)?\s*[:.)-]*\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\d+(?:\.\d+)*\s*[:.)-]*\s*", "", cleaned)
    cleaned = re.sub(r"^\b(?:theory|practical|note|notes|guidelines?|evaluation|assessment|instructions?)\b\s*[:.)-]*\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\((?:note|notes|guidelines?|excluded|practical|assessment|evaluation|instructions?).*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+no\.\s*of\s+periods\s*:?.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+marks?\s*:?.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+page\s*\d+.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(theory\)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = _collapse_repeated_phrase(cleaned)
    cleaned = cleaned.strip(" -:;.,\t–").strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _collapse_repeated_phrase(text: str) -> str:
    words = text.split()
    if len(words) == 2 and words[0].lower() == words[1].lower():
        return words[0]
    if len(words) < 4:
        return text

    for size in range(1, len(words) // 2 + 1):
        if len(words) % size:
            continue
        chunk = words[:size]
        if chunk * (len(words) // size) == words:
            return " ".join(chunk)

    half = len(words) // 2
    if len(words) % 2 == 0 and words[:half] == words[half:]:
        return " ".join(words[:half])

    return text


def _is_boilerplate_title(title: str) -> bool:
    lowered = title.lower().strip()
    if lowered in IGNORED_TITLES:
        return True
    if any(phrase == lowered for phrase in IGNORED_TITLE_PHRASES):
        return True
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in INVALID_TITLE_PATTERNS):
        return True
    if any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in EXCLUDED_KEYWORDS):
        return True
    if re.match(r"^c[\s\-]?\d", lowered):
        return True
    if any(lowered.startswith(verb + " ") or lowered.startswith(verb + "s ") for verb in OUTCOME_VERBS):
        return True
    admin_starts = (
        "demonstrate ",
        "formulate ",
        "compile ",
        "substantiate ",
        "the teachers ",
        "the schools ",
        "of interdisciplinary ",
        "suggested ",
        "weightage ",
        "rubrics ",
        "name of ",
        "subject wise ",
        "map ",
        "learning outcome",
        "research work ",
        "conquest,",
        "everyday life culture",
    )
    if any(lowered.startswith(prefix) for prefix in admin_starts):
        return True
    if re.search(r"\b(investigates|describes|applies|explains|demonstrates|manipulates|defines|recognises|recognizes|employs|analyses|analyzes|constructs|carries out)\b", lowered):
        return True
    if len(lowered) < 4:
        return True
    if lowered.isdigit():
        return True
    if re.fullmatch(r"[\W_]+", lowered):
        return True
    if sum(char.isalpha() for char in lowered) < 3:
        return True
    if re.match(r"^[a-z]?-?\d+(?:\.\d+)*\b", lowered):
        return True
    if re.match(r"^[a-z]\.[\s\d]", lowered):
        return True
    if re.fullmatch(r"[a-z]", lowered):
        return True
    if re.search(r"[√∫𝛼𝛽𝒂𝒙𝟐±]", title):
        return True
    if sum(char.isalpha() for char in lowered) / max(len(lowered), 1) < 0.4:
        return True
    incomplete_endings = (" and", " or", " of", " in", " to", " for", " with", " on", " at", " by", " from", " the", " a", " an")
    if any(lowered.endswith(ending) for ending in incomplete_endings):
        return True
    if re.search(r"\b(locating and labeling|type of questions|learning process|time schedule|collaboration & communication|quiz, debate|participation of the student|examining and breaking|compiling information)\b", lowered):
        return True
    if re.search(r"[\U0001d400-\U0001d7ff]", title):
        return True
    if title.count(",") >= 2 or len(title.split()) > 9:
        return True
    if len(re.findall(r"\bresources?\b", lowered)) >= 2:
        return True
    if lowered.startswith("lifelines of ") and "economy" not in lowered:
        return True
    return False


def _title_fragment_score(line: str) -> int:
    score = 0
    stripped = _normalize_whitespace(line)
    if not stripped:
        return 0
    if stripped.upper() == stripped and len(stripped.split()) <= 10:
        score += 2
    if len(stripped.split()) <= 10 and re.search(r"[A-Za-z]", stripped):
        score += 1
    if re.search(r"^(?:chapter|unit|lesson|topic|section|unit\s*[ivxlcdm\d]+)\b", stripped, flags=re.IGNORECASE):
        score += 2
    if re.match(r"^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z][A-Z\s&/-]{1,50})$", stripped):
        score += 1
    return score


def _looks_like_heading(line: str) -> tuple[bool, str]:
    stripped = _normalize_whitespace(line)
    if not stripped or len(stripped) > 120:
        return False, ""
    if stripped[0].islower() or stripped.endswith("."):
        return False, ""

    lowered = stripped.lower().rstrip(".")
    if _is_boilerplate_title(lowered):
        return False, ""

    score = 0
    section_match = HEADING_PATTERN.match(stripped)
    if section_match and section_match.group(1):
        score += 3
    if NUMBERED_PATTERN.match(stripped):
        score += 2
    if ROMAN_PREFIX_PATTERN.match(stripped):
        score += 2
    if stripped.upper() == stripped and 2 <= len(stripped.split()) <= 10:
        score += 2
    if re.match(r"^[A-Z][A-Za-z0-9'’&(),\-/]*(?:\s+[A-Za-z0-9'’&(),\-/]+){1,7}$", stripped):
        score += 2
    if len(stripped.split()) <= 12 and re.search(r"[A-Za-z]", stripped):
        score += 1

    if score < 2:
        return False, ""

    candidate = stripped
    if section_match and section_match.group(3):
        candidate = section_match.group(3)
    elif NUMBERED_PATTERN.match(stripped):
        candidate = NUMBERED_PATTERN.match(stripped).group(2)
    elif ROMAN_PREFIX_PATTERN.match(stripped):
        candidate = ROMAN_PREFIX_PATTERN.match(stripped).group(1)

    candidate = _clean_title(candidate)
    if not candidate or _is_boilerplate_title(candidate):
        return False, ""

    if candidate.count(" ") > 10:
        return False, ""
    if not re.match(r"^(?:[A-Z][A-Za-z0-9'’&(),\- ]+|[A-Z]{2,}(?:\s+[A-Z]{2,})*)$", candidate):
        return False, ""

    return True, candidate


def _merge_title_fragments(candidates: List[tuple[int, str]]) -> List[tuple[int, str]]:
    if not candidates:
        return []

    merged: List[tuple[int, str]] = []
    index = 0
    while index < len(candidates):
        start_index, title = candidates[index]
        current_title = title
        current_end = start_index
        next_index = index + 1

        while next_index < len(candidates):
            candidate_index, candidate_title = candidates[next_index]
            gap = candidate_index - current_end
            if gap > 2:
                break
            if len(current_title.split()) >= 8:
                break
            if len(candidate_title.split()) > 3:
                break
            if len(current_title.split()) > 5:
                break

            combined = _clean_title(f"{current_title} {candidate_title}")
            if not combined or combined.lower() == current_title.lower():
                break
            if _is_boilerplate_title(combined):
                break

            current_title = combined
            current_end = candidate_index
            next_index += 1

        merged.append((start_index, current_title))
        index = next_index

    return merged


def _structured_heading_candidates(text: str) -> List[tuple[int, str]]:
    candidates: List[tuple[int, str]] = []
    pattern = re.compile(
        r"(?im)^(?:unit|chapter|topic|lesson)\s*(?:\d+[a-z]?|[ivxlcdm]+)?\s*[:.)-]*\s*([A-Za-z][A-Za-z0-9'’&(),\-/ ]{2,90})$"
    )
    for index, line in enumerate(text.splitlines()):
        match = pattern.match(line.strip())
        if not match:
            continue
        title = _clean_title(match.group(1))
        if title and not _is_boilerplate_title(title):
            candidates.append((index, title))
    return candidates


def _split_by_headings(text: str) -> List[ChapterSection]:
    lines = [line.rstrip() for line in text.splitlines()]
    candidates: List[tuple[int, str]] = _structured_heading_candidates(text)

    for index, line in enumerate(lines):
        is_heading, title = _looks_like_heading(line)
        if is_heading and title:
            candidates.append((index, title))

    candidates.sort(key=lambda item: item[0])
    candidates = _merge_title_fragments(candidates)

    if not candidates:
        fallback = _clean_title(lines[0]) if lines else "General Curriculum"
        return [ChapterSection(title=fallback or "General Curriculum", content=text.strip(), section_type="document")]

    sections: List[ChapterSection] = []
    seen_titles: set[str] = set()

    for position, (start_index, title) in enumerate(candidates):
        normalized_title = title.lower()
        if normalized_title in seen_titles:
            continue

        end_index = candidates[position + 1][0] if position + 1 < len(candidates) else len(lines)
        content_lines = [line.strip() for line in lines[start_index:end_index] if line.strip()]
        content = "\n".join(content_lines).strip()
        if len(content) < 25:
            continue

        lowered_title = title.lower()
        section_type = "chapter"
        if lowered_title.startswith("unit"):
            section_type = "unit"
        elif lowered_title.startswith("topic"):
            section_type = "topic"
        elif lowered_title.startswith("lesson"):
            section_type = "lesson"

        sections.append(ChapterSection(title=title, content=content, section_type=section_type))
        seen_titles.add(normalized_title)

    if not sections:
        return [ChapterSection(title="General Curriculum", content=text.strip(), section_type="document")]

    return sections


def _split_by_toc(document: PdfDocument) -> List[ChapterSection]:
    sections: List[ChapterSection] = []
    toc_entries = [(level, _clean_title(title), page) for level, title, page in document.toc_entries if level <= 6]
    toc_entries = [(level, title, page) for level, title, page in toc_entries if title and not _is_boilerplate_title(title)]
    if not toc_entries:
        return []

    seen_titles: set[str] = set()
    for index, (_, title, page) in enumerate(toc_entries):
        normalized_title = title.lower()
        if normalized_title in seen_titles:
            continue

        start_page = max(page - 1, 0)
        next_page = toc_entries[index + 1][2] if index + 1 < len(toc_entries) else len(document.page_texts) + 1
        end_page = max(next_page - 1, start_page + 1)
        content = "\n".join(document.page_texts[start_page:end_page]).strip()
        if len(content) < 25:
            continue

        sections.append(ChapterSection(title=title, content=content, section_type="unit" if title.lower().startswith("unit") else "chapter"))
        seen_titles.add(normalized_title)

    return sections


def extract_chapter_sections(document: PdfDocument) -> List[ChapterSection]:
    """Extract ordered chapter-like sections from a syllabus PDF."""

    # For school files, prefer headings because CBSE TOCs are often incomplete or cover only practical guidelines.
    is_school = any(term in document.path.name.lower() for term in ["sec", "srsec", "class", "math", "physics", "chemistry", "science", "social"])
    
    if document.toc_entries and not is_school:
        toc_sections = _split_by_toc(document)
        if len(toc_sections) >= 5:
            return toc_sections

    if document.text.strip():
        return _split_by_headings(document.text)

    if document.toc_entries:
        toc_sections = _split_by_toc(document)
        if toc_sections:
            return toc_sections

    return [ChapterSection(title="General Curriculum", content="", section_type="document")]


def normalize_chapter_title(title: str) -> str:
    """Normalize a raw heading into a presentation-ready chapter title."""

    return _clean_title(title)


def is_valid_chapter_title(title: str) -> bool:
    """Return True when a title represents a meaningful curriculum chapter."""

    cleaned = _clean_title(title)
    return bool(cleaned) and not _is_boilerplate_title(cleaned)


def extract_chapters(document: PdfDocument) -> List[str]:
    """Return a simple list of chapter names for downstream consumers."""

    return [section.title for section in extract_chapter_sections(document) if is_valid_chapter_title(section.title)]
