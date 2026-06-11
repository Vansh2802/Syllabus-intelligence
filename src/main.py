"""End-to-end pipeline for building a curriculum intelligence dataset."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from chapter_extractor import extract_chapter_sections, is_valid_chapter_title, normalize_chapter_title
from curriculum_analyzer import analyze_chapter
from csv_exporter import CurriculumRow, export_dataset
from pdf_extractor import PdfDocument, ensure_pdf_directory, get_pdf_files, infer_class_label_from_text, load_pdf_documents

ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT_DIR / "pdfs"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = ROOT_DIR / "logs"
CSV_PATH = OUTPUT_DIR / "curriculum_dataset.csv"
LOG_PATH = LOG_DIR / "project.log"


def configure_logging() -> None:
    """Configure file and console logging for the pipeline."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def build_rows_for_document(document: PdfDocument) -> List[CurriculumRow]:
    """Transform a PDF into dataset rows."""

    if not document.text.strip():
        logging.warning("Skipping empty PDF: %s", document.path.name)
        return []

    sections = extract_chapter_sections(document)
    rows: List[CurriculumRow] = []

    class_label = document.class_label or infer_class_label_from_text(document.text)
    current_class = class_label

    for section in sections:
        # Check if this section indicates a class change
        combined = f"{section.title}\n{section.content[:1000]}".lower()
        if "class xii" in combined or "class-xii" in combined or "class 12" in combined or "class-12" in combined or "classxii" in combined:
            current_class = "12"
        elif "class xi" in combined or "class-xi" in combined or "class 11" in combined or "class-11" in combined or "classxi" in combined:
            current_class = "11"
        elif "class x" in combined or "class-x" in combined or "class 10" in combined or "class-10" in combined or "classx" in combined:
            current_class = "10"
        elif "class ix" in combined or "class-ix" in combined or "class 9" in combined or "class-9" in combined or "classix" in combined:
            current_class = "9"

        chapter_title = normalize_chapter_title(section.title)
        if not is_valid_chapter_title(chapter_title):
            logging.debug("Skipping invalid chapter: %s", section.title)
            continue
        analysis = analyze_chapter(chapter_title, section.content, document.subject, current_class)
        rows.append(
            CurriculumRow(
                class_label=current_class,
                subject=document.subject,
                chapter=chapter_title,
                core_concept=analysis.core_concept,
                skill_domain=analysis.skill_domain,
                difficulty=analysis.difficulty,
                summary=analysis.summary,
            )
        )

    return rows


def main() -> None:
    """Run the complete syllabus-to-skills generation workflow."""

    load_dotenv(ROOT_DIR / ".env")
    configure_logging()
    logging.info("Starting education pathway intelligence pipeline")

    ensure_pdf_directory(PDF_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = get_pdf_files(PDF_DIR)
    if not pdf_files:
        logging.warning("No PDF files found in %s", PDF_DIR)
        actual_path = export_dataset([], CSV_PATH)
        logging.info("Created empty dataset at %s", actual_path)
        return

    rows: List[CurriculumRow] = []
    try:
        documents = load_pdf_documents(PDF_DIR)
    except Exception as exc:
        logging.exception("Failed to load PDFs: %s", exc)
        actual_path = export_dataset([], CSV_PATH)
        logging.info("Created empty dataset at %s", actual_path)
        return

    for document in documents:
        try:
            rows.extend(build_rows_for_document(document))
        except Exception as exc:
            logging.exception("Failed to process %s: %s", document.path.name, exc)

    actual_path = export_dataset(rows, CSV_PATH)
    logging.info("Generated %d dataset rows", len(rows))
    logging.info("CSV saved to %s", actual_path)


if __name__ == "__main__":
    main()
