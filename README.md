# Curriculum Intelligence Platform

This project converts CBSE Class 9-12 syllabus PDFs into a presentation-ready curriculum intelligence dataset.

## What it does

- Reads every PDF from `pdfs/`
- Extracts text and table-of-contents data with PyMuPDF
- Detects class and subject metadata from filenames when possible
- Extracts chapter and unit sections
- Uses OpenAI to produce one core concept, one skill domain, one difficulty level, and one summary per chapter
- Falls back to deterministic rules when OpenAI is unavailable or returns invalid output
- Exports `output/curriculum_dataset.csv`
- Shows the dataset in a Streamlit dashboard
- Writes logs to `logs/project.log`

## Installation

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Put your syllabus PDFs inside `pdfs/`.
2. Use filenames like `class9_math.pdf`, `class11_physics.pdf`, or `class12_computer_science.pdf`.
3. Add your OpenAI API key to `.env`:

```env
OPEN_AI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

4. If no OpenAI key is available, the pipeline still runs with local fallback rules.

## Running

Generate the dataset:

```bash
python src/main.py
```

Launch the dashboard:

```bash
streamlit run src/dashboard.py
```

The dataset is written to:

- `output/curriculum_dataset.csv`

Logs are written to:

- `logs/project.log`

## Error Handling

- Missing PDFs are handled by creating an empty dataset with headers.
- Corrupted PDFs are logged and skipped.
- Empty PDFs are logged and skipped.
- Invalid OpenAI responses and rate-limit failures fall back to deterministic analysis.

## Notes

- Each chapter maps to exactly one primary skill domain.
- Only these skill domains are used: Quantitative Reasoning, Scientific Reasoning, Computational Thinking, Analytical Thinking, Communication, Research Skills, Design Thinking, Problem Solving.
