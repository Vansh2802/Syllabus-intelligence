"""Analytics layer for the Curriculum Intelligence platform."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd

from csv_exporter import CSV_HEADERS, derive_learning_stage
from curriculum_analyzer import ALLOWED_DIFFICULTIES, ALLOWED_SKILL_DOMAINS


JOURNEY_LEVELS: List[Tuple[str, str]] = [
    ("9", "Class 9"),
    ("10", "Class 10"),
    ("11", "Class 11"),
    ("12", "Class 12"),
    ("Engineering Year 1", "Engineering Year 1"),
    ("Engineering Year 2", "Engineering Year 2"),
    ("Engineering Year 3", "Engineering Year 3"),
    ("Engineering Year 4", "Engineering Year 4"),
]

STAGE_ORDER = [
    "Secondary School",
    "Higher Secondary",
    "Engineering Year 1",
    "Engineering Year 2",
    "Engineering Year 3",
    "Engineering Year 4",
]

FOCUS_SUBJECTS = [
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "Computer Science",
    "English",
]

READINESS_BRIDGES: List[Dict[str, object]] = [
    {
        "school": "Algebra & Equations",
        "engineering": "Engineering Mathematics",
        "keywords": ("algebra", "equation", "polynomial", "quadratic", "linear relation", "matrix", "calculus", "trigonomet"),
    },
    {
        "school": "Programming Basics",
        "engineering": "Data Structures & Algorithms",
        "keywords": ("program", "computational", "algorithm", "coding", "software", "database", "python", "java"),
    },
    {
        "school": "Physics & Motion",
        "engineering": "Engineering Mechanics",
        "keywords": ("motion", "force", "kinematic", "mechanic", "dynamics", "energy", "momentum", "electromagnet"),
    },
    {
        "school": "Chemical Principles",
        "engineering": "Engineering Chemistry",
        "keywords": ("chemical", "reaction", "mole", "bond", "equilibrium", "electrochem", "organic"),
    },
    {
        "school": "Statistics & Probability",
        "engineering": "Engineering Analytics",
        "keywords": ("statistic", "probability", "data interpret", "uncertainty", "distribution"),
    },
    {
        "school": "Geometry & Mensuration",
        "engineering": "Applied Geometry & Design",
        "keywords": ("geometry", "mensuration", "coordinate", "surface area", "volume", "vector"),
    },
    {
        "school": "Scientific Inquiry",
        "engineering": "Research & Innovation",
        "keywords": ("scientific", "experiment", "research", "hypothesis", "observation", "laboratory"),
    },
    {
        "school": "Communication Skills",
        "engineering": "Technical Communication",
        "keywords": ("communication", "writing", "presentation", "language", "report"),
    },
]

CHART_COLORS = {
    "Secondary School": "#2563eb",
    "Higher Secondary": "#7c3aed",
    "Engineering Year 1": "#0d9488",
    "Engineering Year 2": "#0ea5e9",
    "Engineering Year 3": "#f59e0b",
    "Engineering Year 4": "#db2777",
    "Class 9": "#3b82f6",
    "Class 10": "#6366f1",
    "Class 11": "#8b5cf6",
    "Class 12": "#a855f7",
    "Engineering Year 1": "#14b8a6",
    "Engineering Year 2": "#06b6d4",
    "Engineering Year 3": "#d97706",
    "Engineering Year 4": "#e11d48",
}

SKILL_COLORS = {
    "Quantitative Reasoning": "#2563eb",
    "Scientific Reasoning": "#059669",
    "Computational Thinking": "#7c3aed",
    "Analytical Thinking": "#d97706",
    "Communication": "#db2777",
    "Research Skills": "#0891b2",
    "Design Thinking": "#ea580c",
    "Problem Solving": "#dc2626",
}

DIFFICULTY_COLORS = {
    "Beginner": "#22c55e",
    "Intermediate": "#f59e0b",
    "Advanced": "#ef4444",
}


def enrich_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare a curriculum dataframe for analytics visualizations."""

    if frame.empty:
        return pd.DataFrame(columns=CSV_HEADERS)

    enriched = frame.copy()
    for column in CSV_HEADERS:
        if column not in enriched.columns:
            enriched[column] = ""

    enriched["Class"] = enriched["Class"].astype(str).str.strip()
    enriched["Learning_Stage"] = enriched.apply(
        lambda row: row.get("Learning_Stage") or derive_learning_stage(str(row["Class"])),
        axis=1,
    )
    enriched["Journey_Label"] = enriched["Class"].map(dict(JOURNEY_LEVELS)).fillna(enriched["Class"])
    enriched["Curriculum_Unit"] = enriched["Chapter"]
    return enriched


def class_sort_key(value: str) -> int:
    order = {key: index for index, (key, _) in enumerate(JOURNEY_LEVELS)}
    return order.get(str(value), 99)


def journey_labels_present(frame: pd.DataFrame) -> List[str]:
    present = set(frame["Class"].astype(str))
    return [label for key, label in JOURNEY_LEVELS if key in present]


def executive_summary(frame: pd.DataFrame) -> Dict[str, int]:
    return {
        "curriculum_units": int(frame["Curriculum_Unit"].nunique()) if not frame.empty else 0,
        "concepts": int(frame["Core_Concept"].nunique()) if not frame.empty else 0,
        "skill_domains": int(frame["Skill_Domain"].nunique()) if not frame.empty else 0,
        "subjects": int(frame["Subject"].nunique()) if not frame.empty else 0,
        "levels": int(frame["Class"].nunique()) if not frame.empty else 0,
    }


def progression_timeline_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_key, label in JOURNEY_LEVELS:
        subset = frame[frame["Class"].astype(str) == class_key]
        if subset.empty:
            continue
        rows.append(
            {
                "Class_Key": class_key,
                "Journey_Label": label,
                "Concepts": subset["Core_Concept"].nunique(),
                "Subjects": subset["Subject"].nunique(),
                "Units": subset["Curriculum_Unit"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def concept_growth_by_stage(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby("Learning_Stage", as_index=False)["Core_Concept"]
        .nunique()
        .rename(columns={"Core_Concept": "Concept_Count"})
    )
    grouped["Learning_Stage"] = pd.Categorical(grouped["Learning_Stage"], categories=STAGE_ORDER, ordered=True)
    return grouped.sort_values("Learning_Stage")


def skill_domain_by_class(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["Journey_Label", "Skill_Domain"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    label_order = journey_labels_present(frame)
    grouped["Journey_Label"] = pd.Categorical(grouped["Journey_Label"], categories=label_order, ordered=True)
    grouped["Skill_Domain"] = pd.Categorical(grouped["Skill_Domain"], categories=ALLOWED_SKILL_DOMAINS, ordered=True)
    return grouped.sort_values(["Journey_Label", "Skill_Domain"])


def subject_skill_contribution(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["Subject"].isin(FOCUS_SUBJECTS)]
    if subset.empty:
        subset = frame.copy()
    grouped = (
        subset.groupby(["Subject", "Skill_Domain"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    grouped["Subject"] = pd.Categorical(grouped["Subject"], categories=FOCUS_SUBJECTS, ordered=True)
    grouped["Skill_Domain"] = pd.Categorical(grouped["Skill_Domain"], categories=ALLOWED_SKILL_DOMAINS, ordered=True)
    return grouped.sort_values(["Subject", "Skill_Domain"])


def difficulty_by_stage(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["Learning_Stage", "Difficulty"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    grouped["Learning_Stage"] = pd.Categorical(grouped["Learning_Stage"], categories=STAGE_ORDER, ordered=True)
    grouped["Difficulty"] = pd.Categorical(grouped["Difficulty"], categories=ALLOWED_DIFFICULTIES, ordered=True)
    return grouped.sort_values(["Learning_Stage", "Difficulty"])


def top_concepts_by_stage(frame: pd.DataFrame, limit: int = 10) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    for stage in STAGE_ORDER:
        subset = frame[frame["Learning_Stage"] == stage]
        if subset.empty:
            results[stage] = pd.DataFrame(columns=["Core_Concept", "Count"])
            continue
        ranked = (
            subset.groupby("Core_Concept", as_index=False)
            .size()
            .rename(columns={"size": "Count"})
            .sort_values("Count", ascending=False)
            .head(limit)
        )
        results[stage] = ranked
    return results


def _concept_matches_keywords(text: str, keywords: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def engineering_readiness_links(frame: pd.DataFrame) -> Tuple[List[str], List[int], List[int], List[int], List[str]]:
    """Build Sankey nodes and links from school concepts to engineering pathways."""

    school_rows = frame[frame["Learning_Stage"].isin(["Secondary School", "Higher Secondary"])]
    engineering_rows = frame[frame["Learning_Stage"].str.startswith("Engineering")]

    school_text = " ".join(
        school_rows[["Core_Concept", "Chapter", "Subject", "Summary"]].astype(str).agg(" ".join, axis=1).tolist()
    ).lower()
    engineering_text = " ".join(
        engineering_rows[["Core_Concept", "Chapter", "Subject", "Summary"]].astype(str).agg(" ".join, axis=1).tolist()
    ).lower()

    labels: List[str] = []
    sources: List[int] = []
    targets: List[int] = []
    values: List[int] = []
    link_labels: List[str] = []

    for bridge in READINESS_BRIDGES:
        school_label = str(bridge["school"])
        engineering_label = str(bridge["engineering"])
        keywords = tuple(bridge["keywords"])  # type: ignore[arg-type]

        school_hits = sum(
            1
            for _, row in school_rows.iterrows()
            if _concept_matches_keywords(" ".join([row["Core_Concept"], row["Chapter"], row["Subject"]]), keywords)
        )
        engineering_hits = sum(
            1
            for _, row in engineering_rows.iterrows()
            if _concept_matches_keywords(" ".join([row["Core_Concept"], row["Chapter"], row["Subject"]]), keywords)
        )
        corpus_hits = sum(1 for keyword in keywords if keyword in school_text or keyword in engineering_text)
        strength = max(school_hits, engineering_hits, corpus_hits, 1)

        if school_label not in labels:
            labels.append(school_label)
        if engineering_label not in labels:
            labels.append(engineering_label)

        source_index = labels.index(school_label)
        target_index = labels.index(engineering_label)
        sources.append(source_index)
        targets.append(target_index)
        values.append(int(strength))
        link_labels.append(f"{school_label} → {engineering_label}")

    return labels, sources, targets, values, link_labels


def curriculum_coverage_heatmap(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        frame.groupby(["Subject", "Journey_Label"])["Core_Concept"]
        .nunique()
        .reset_index()
        .pivot(index="Subject", columns="Journey_Label", values="Core_Concept")
        .fillna(0)
    )
    label_order = journey_labels_present(frame)
    for label in label_order:
        if label not in pivot.columns:
            pivot[label] = 0
    pivot = pivot[label_order]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    return pivot


def radar_skill_profile(frame: pd.DataFrame, journey_label: str) -> Dict[str, int]:
    subset = frame[frame["Journey_Label"] == journey_label]
    counts = {skill: 0 for skill in ALLOWED_SKILL_DOMAINS}
    if subset.empty:
        return counts
    for skill, count in subset["Skill_Domain"].value_counts().items():
        if skill in counts:
            counts[skill] = int(count)
    return counts


def journey_detail(frame: pd.DataFrame, class_key: str) -> Dict[str, object]:
    subset = frame[frame["Class"].astype(str) == class_key]
    if subset.empty:
        return {
            "subjects": [],
            "concepts": [],
            "skills": [],
            "difficulty": {},
            "units": 0,
        }

    return {
        "subjects": sorted(subset["Subject"].unique().tolist()),
        "concepts": subset["Core_Concept"].value_counts().head(15).index.tolist(),
        "skills": subset["Skill_Domain"].value_counts().to_dict(),
        "difficulty": subset["Difficulty"].value_counts().to_dict(),
        "units": int(subset["Curriculum_Unit"].nunique()),
        "stage": subset["Learning_Stage"].iloc[0],
        "label": subset["Journey_Label"].iloc[0],
    }


def normalize_display_class(value: object) -> str:
    text = str(value).strip()
    if text.lower() == "engineering":
        return "Engineering (AICTE)"
    if text.isdigit():
        return f"Class {text}"
    return text


def generate_skill_evolution_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate the Skill Evolution Dataset: Stage, Skill_Domain, Count"""
    if frame.empty:
        return pd.DataFrame(columns=["Stage", "Skill_Domain", "Count"])
    grouped = frame.groupby(["Learning_Stage", "Skill_Domain"]).size().reset_index(name="Count")
    grouped.rename(columns={"Learning_Stage": "Stage"}, inplace=True)
    return grouped


def generate_subject_complexity_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate the Subject Complexity Dataset: Stage, Subject, Concept_Count, Difficulty_Score"""
    if frame.empty:
        return pd.DataFrame(columns=["Stage", "Subject", "Concept_Count", "Difficulty_Score"])
    temp = frame.copy()
    diff_map = {"Beginner": 1.0, "Intermediate": 2.0, "Advanced": 3.0}
    temp["Diff_Value"] = temp["Difficulty"].map(diff_map).fillna(1.0)
    
    grouped = temp.groupby(["Learning_Stage", "Subject"]).agg(
        Concept_Count=("Core_Concept", "nunique"),
        Difficulty_Score=("Diff_Value", "mean")
    ).reset_index()
    grouped.rename(columns={"Learning_Stage": "Stage"}, inplace=True)
    grouped["Difficulty_Score"] = grouped["Difficulty_Score"].round(2)
    return grouped


def generate_learning_path_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate the Learning Path Dataset: Source_Stage, Source_Concept, Target_Stage, Target_Concept"""
    if frame.empty:
        return pd.DataFrame(columns=["Source_Stage", "Source_Concept", "Target_Stage", "Target_Concept"])
    
    school = frame[frame["Learning_Stage"].isin(["Secondary School", "Higher Secondary"])].copy()
    eng = frame[frame["Learning_Stage"].str.startswith("Engineering")].copy()
    
    links = []
    seen_pairs = set()
    for _, s_row in school.iterrows():
        s_concept = s_row["Core_Concept"]
        s_words = set(s_concept.lower().split()) - {"and", "or", "of", "in", "to", "the", "a", "an", "concepts"}
        if not s_words:
            continue
        for _, e_row in eng.iterrows():
            e_concept = e_row["Core_Concept"]
            e_words = set(e_concept.lower().split()) - {"and", "or", "of", "in", "to", "the", "a", "an", "concepts"}
            if s_words & e_words:
                pair = (s_concept, e_concept)
                if pair not in seen_pairs:
                    links.append({
                        "Source_Stage": s_row["Learning_Stage"],
                        "Source_Concept": s_concept,
                        "Target_Stage": e_row["Learning_Stage"],
                        "Target_Concept": e_concept
                    })
                    seen_pairs.add(pair)
    if not links:
        links.append({
            "Source_Stage": "Secondary School",
            "Source_Concept": "Real Number Systems",
            "Target_Stage": "Engineering Year 1",
            "Target_Concept": "Engineering Mathematics"
        })
        links.append({
            "Source_Stage": "Higher Secondary",
            "Source_Concept": "Electromagnetic Phenomena",
            "Target_Stage": "Engineering Year 2",
            "Target_Concept": "Engineering Electromagnetics"
        })
    return pd.DataFrame(links)


def generate_educational_stage_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate the Educational Stage Dataset: Stage, Subjects, Concepts, Skills, Average_Difficulty"""
    if frame.empty:
        return pd.DataFrame(columns=["Stage", "Subjects", "Concepts", "Skills", "Average_Difficulty"])
    
    temp = frame.copy()
    diff_map = {"Beginner": 1.0, "Intermediate": 2.0, "Advanced": 3.0}
    temp["Diff_Value"] = temp["Difficulty"].map(diff_map).fillna(1.0)
    
    rows = []
    for stage in temp["Learning_Stage"].unique():
        if not stage:
            continue
        subset = temp[temp["Learning_Stage"] == stage]
        subjects_list = ", ".join(sorted(subset["Subject"].unique()))
        concepts_count = subset["Core_Concept"].nunique()
        skills_list = ", ".join(sorted(subset["Skill_Domain"].unique()))
        avg_diff = subset["Diff_Value"].mean()
        
        rows.append({
            "Stage": stage,
            "Subjects": subjects_list,
            "Concepts": concepts_count,
            "Skills": skills_list,
            "Average_Difficulty": round(avg_diff, 2)
        })
    return pd.DataFrame(rows)


def generate_engineering_readiness_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate the Engineering Readiness Dataset: School_Concept, Engineering_Concept, Relationship_Strength"""
    if frame.empty:
        return pd.DataFrame(columns=["School_Concept", "Engineering_Concept", "Relationship_Strength"])
    
    school = frame[frame["Learning_Stage"].isin(["Secondary School", "Higher Secondary"])].copy()
    eng = frame[frame["Learning_Stage"].str.startswith("Engineering")].copy()
    
    readiness = []
    seen_pairs = set()
    for _, s_row in school.iterrows():
        s_concept = s_row["Core_Concept"]
        s_words = set(s_concept.lower().split()) - {"and", "or", "of", "in", "to", "the", "a", "an", "concepts"}
        if not s_words:
            continue
        for _, e_row in eng.iterrows():
            e_concept = e_row["Core_Concept"]
            e_words = set(e_concept.lower().split()) - {"and", "or", "of", "in", "to", "the", "a", "an", "concepts"}
            
            overlap = s_words & e_words
            if overlap:
                pair = (s_concept, e_concept)
                if pair not in seen_pairs:
                    strength = len(overlap) * 2 + 1
                    readiness.append({
                        "School_Concept": s_concept,
                        "Engineering_Concept": e_concept,
                        "Relationship_Strength": strength
                    })
                    seen_pairs.add(pair)
    if not readiness:
        for bridge in READINESS_BRIDGES:
            readiness.append({
                "School_Concept": bridge["school"],
                "Engineering_Concept": bridge["engineering"],
                "Relationship_Strength": 3
            })
    return pd.DataFrame(readiness)
