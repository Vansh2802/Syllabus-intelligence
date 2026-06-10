"""PDF loading and metadata extraction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Tuple

import fitz  # PyMuPDF


@dataclass(slots=True)
class PdfDocument:
    """Represents a syllabus PDF and its extracted content."""

    path: Path
    class_label: str
    subject: str
    text: str
    page_texts: list[str]
    toc_entries: list[tuple[int, str, int]]


SUBJECT_ALIASES = {
    "mathematics": "Mathematics",
    "math": "Mathematics",
    "maths": "Mathematics",
    "sciencest": "Science",
    "science": "Science",
    "english": "English",
    "sst": "Social Science",
    "social_science": "Social Science",
    "social science": "Social Science",
    "socialscience": "Social Science",
    "socialsciences": "Social Science",
    "physics": "Physics",
    "phy": "Physics",
    "chemistry": "Chemistry",
    "chem": "Chemistry",
    "biology": "Biology",
    "computer": "Computer Science",
    "computerscience": "Computer Science",
    "computer_science": "Computer Science",
    "cs": "Computer Science",
    "hindi": "Hindi",
    "sanskrit": "Sanskrit",
    "economics": "Economics",
    "business": "Business Studies",
    "accountancy": "Accountancy",
    "accounting": "Accountancy",
    "history": "History",
    "geography": "Geography",
    "civics": "Civics",
    "politicalscience": "Political Science",
    "art": "Art",
}

CLASS_ALIASES = {
    "ix": "9",
    "x": "10",
    "xi": "11",
    "xii": "12",
}

ORDERED_SUBJECT_ALIASES = sorted(SUBJECT_ALIASES.items(), key=lambda item: (-len(item[0]), item[0]))


def _normalize_lookup_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def get_pdf_files(pdf_dir: Path) -> List[Path]:
    """Return all PDF files from the given directory."""

    if not pdf_dir.exists():
        return []
    return sorted(path for path in pdf_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def normalize_subject_token(token: str) -> str:
    """Normalize a filename token into a subject name when possible."""

    cleaned = _normalize_lookup_token(token)
    for alias, mapped in ORDERED_SUBJECT_ALIASES:
        if cleaned == _normalize_lookup_token(alias):
            return mapped
    if token.lower() in SUBJECT_ALIASES:
        return SUBJECT_ALIASES[token.lower()]
    return token.replace("_", " ").title()


def normalize_class_token(token: str) -> str:
    """Normalize a class token into a numeric CBSE class label or Engineering."""

    cleaned = re.sub(r"[^a-z0-9]+", "", token.lower())
    if cleaned in {"engineering", "aicte", "btech", "ug"}:
        return "Engineering"
    if cleaned in CLASS_ALIASES:
        return CLASS_ALIASES[cleaned]
    if cleaned.isdigit() and 1 <= int(cleaned) <= 12:
        return cleaned
    return ""


def infer_engineering_label_from_text(text: str) -> str:
    """Detect AICTE or engineering syllabus PDFs from document text."""

    sample = text[:6000].lower()
    if re.search(r"\b(aicte|all india council for technical education|engineering curriculum|b\.?\s*tech)\b", sample):
        return "Engineering"
    return ""


def infer_class_label_from_text(text: str) -> str:
    """Extract a class label from the PDF body when the filename is incomplete."""

    patterns = [
        r"class\s*[–\-]?\s*(ix|x|xi|xii|9|10|11|12)\b",
        r"class(?:es)?\s*[-–:]?\s*(ix|x|xi|xii|9|10|11|12)\b",
        r"grade\s*[-–:]?\s*(ix|x|xi|xii|9|10|11|12)\b",
        r"std\s*[-–:]?\s*(ix|x|xi|xii|9|10|11|12)\b",
        r"\bclass\s*[–\-]?\s*(ix|x|xi|xii)\s*\(",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_class_token(match.group(1))
    return ""


def infer_subject_from_text(text: str) -> str:
    """Extract a subject name from the PDF body when the filename is incomplete."""

    sample = text[:4000].lower()
    priority_aliases = (
        "social science",
        "social studies",
        "social_science",
        "socialscience",
        "computer science",
        "informatics practices",
        "business studies",
        "political science",
    )
    for alias in priority_aliases:
        alias_text = alias.replace("_", " ")
        if re.search(rf"(?<![a-z0-9]){re.escape(alias_text)}(?![a-z0-9])", sample):
            return SUBJECT_ALIASES.get(alias.replace(" ", "_"), alias_text.title())

    for alias, mapped in ORDERED_SUBJECT_ALIASES:
        alias_text = alias.replace("_", " ").lower()
        if re.search(rf"(?<![a-z0-9]){re.escape(alias_text)}(?![a-z0-9])", sample):
            return mapped
    return ""


def parse_filename_metadata(filename: str) -> Tuple[str, str]:
    """Extract class and subject metadata from a filename when possible."""

    stem = Path(filename).stem
    if re.search(r"\b(aicte|engineering|btech|b\.tech|ug_syllabus)\b", stem, flags=re.IGNORECASE):
        _, subject = _parse_school_filename_metadata(stem)
        return "Engineering", subject

    return _parse_school_filename_metadata(stem)


def _parse_school_filename_metadata(stem: str) -> Tuple[str, str]:
    class_label = ""
    class_patterns = [
        r"(?:class|grade|std)\s*[_\- ]*(\d{1,2}|ix|x|xi|xii)\b",
        r"(?:class|grade|std)(\d{1,2})\b",
        r"(?:secp1|secp\d|part1|p1)(ix|xi|xii|x|9|10|11|12)\b",
        r"(?:srsec|sr_sec|senior)(ix|xi|xii|11|12)\b",
        r"\b(\d{1,2})(?:st|nd|rd|th)?[_\-](?:math|maths|science|physics|phy|chemistry|chem|biology|bio|english|hindi|sst|social|cs|computer)\b",
        r"(?:^|[_\-])(ix|xi|xii|x)(?:[_\-]|$)",
        r"\b(9|10|11|12)\b",
    ]
    for pattern in class_patterns:
        class_match = re.search(pattern, stem, flags=re.IGNORECASE)
        if class_match:
            class_label = normalize_class_token(class_match.group(1))
            if class_label:
                break

    subject = ""
    lowered = _normalize_lookup_token(stem)
    for token, mapped in ORDERED_SUBJECT_ALIASES:
        if _normalize_lookup_token(token) in lowered:
            subject = mapped
            break

    if not subject:
        tokens = [token for token in re.split(r"[_\- ]+", stem) if token]
        if len(tokens) >= 2:
            subject = normalize_subject_token(tokens[-1])
        elif tokens:
            subject = normalize_subject_token(tokens[0])

    return class_label, subject


def _extract_toc_entries(document: fitz.Document) -> list[tuple[int, str, int]]:
    """Return cleaned TOC entries from a PDF document."""

    toc_entries: list[tuple[int, str, int]] = []
    for entry in document.get_toc(simple=True):
        if len(entry) < 3:
            continue
        level, title, page = entry[:3]
        title_text = re.sub(r"\s+", " ", str(title)).strip()
        if not title_text or page <= 0:
            continue
        toc_entries.append((int(level), title_text, int(page)))
    return toc_entries


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, list[str], list[tuple[int, str, int]]]:
    """Extract complete text, page text, and TOC data from a single PDF."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        with fitz.open(pdf_path) as document:
            page_texts = [page.get_text("text").strip() for page in document]
            toc_entries = _extract_toc_entries(document)
    except Exception as exc:  # pragma: no cover - depends on corrupt file contents
        raise RuntimeError(f"Failed to read PDF {pdf_path.name}: {exc}") from exc

    text = "\n".join(page for page in page_texts if page).strip()
    return text, page_texts, toc_entries


def load_pdf_documents(pdf_dir: Path) -> List[PdfDocument]:
    """Load every PDF from the folder and extract its text."""

    documents: List[PdfDocument] = []
    for pdf_path in get_pdf_files(pdf_dir):
        text, page_texts, toc_entries = extract_text_from_pdf(pdf_path)
        class_label, subject = parse_filename_metadata(pdf_path.name)
        if not class_label:
            class_label = infer_engineering_label_from_text(text) or infer_class_label_from_text(text)
        if not subject:
            subject = infer_subject_from_text(text)
        documents.append(
            PdfDocument(
                path=pdf_path,
                class_label=class_label,
                subject=subject,
                text=text,
                page_texts=page_texts,
                toc_entries=toc_entries,
            )
        )
    return documents


def ensure_pdf_directory(pdf_dir: Path) -> None:
    """Create the PDF directory if it does not exist."""

    pdf_dir.mkdir(parents=True, exist_ok=True)
