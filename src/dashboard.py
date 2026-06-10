"""Curriculum Intelligence Analytics Platform dashboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    CHART_COLORS,
    DIFFICULTY_COLORS,
    JOURNEY_LEVELS,
    SKILL_COLORS,
    STAGE_ORDER,
    concept_growth_by_stage,
    curriculum_coverage_heatmap,
    difficulty_by_stage,
    enrich_dataset,
    engineering_readiness_links,
    executive_summary,
    journey_detail,
    journey_labels_present,
    progression_timeline_metrics,
    radar_skill_profile,
    skill_domain_by_class,
    subject_skill_contribution,
    top_concepts_by_stage,
)
from csv_exporter import CSV_HEADERS, load_dataset as load_exported_dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "output" / "curriculum_dataset.csv"

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, Segoe UI, sans-serif", color="#1e293b"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=24, r=24, t=56, b=24),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


st.set_page_config(
    page_title="Curriculum Intelligence Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0e7490 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.22);
    }
    .hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; }
    .hero p { margin: 0.6rem 0 0; opacity: 0.92; font-size: 1.05rem; max-width: 820px; line-height: 1.55; }
    .section-header {
        margin: 2rem 0 0.35rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .section-header h2 { margin: 0; color: #0f172a; font-size: 1.45rem; }
    .section-header p { margin: 0.35rem 0 0; color: #64748b; font-size: 0.95rem; }
    .insight-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .insight-card strong { color: #0f172a; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=CSV_HEADERS)
    return enrich_dataset(load_exported_dataset(CSV_PATH))


def _apply_layout(fig: go.Figure, title: str, height: int = 460) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, x=0, xanchor="left"), height=height)
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    return fig


def _render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Curriculum Intelligence Analytics Platform</h1>
            <p>
                Understand what students learn at each stage — from Class 9 through Class 12 —
                and how school knowledge connects to engineering education.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_executive_summary(frame: pd.DataFrame) -> None:
    _section(
        "Executive Summary",
        "At-a-glance view of curriculum breadth across educational levels.",
    )
    metrics = executive_summary(frame)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Curriculum Units", metrics["curriculum_units"])
    c2.metric("Core Concepts", metrics["concepts"])
    c3.metric("Skill Domains", metrics["skill_domains"])
    c4.metric("Subjects", metrics["subjects"])
    c5.metric("Levels Covered", metrics["levels"])


def _render_learning_progression_timeline(frame: pd.DataFrame) -> None:
    timeline = progression_timeline_metrics(frame)
    if timeline.empty:
        st.info("No progression data available yet.")
        return

    fig = go.Figure()
    y_positions = list(range(len(timeline) - 1, -1, -1))
    colors = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#14b8a6"]

    for index, row in timeline.iterrows():
        y = y_positions[index]
        color = colors[index % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[y],
                mode="markers+text",
                marker=dict(size=28, color=color, line=dict(width=2, color="white")),
                text=[row["Journey_Label"]],
                textposition="middle right",
                textfont=dict(size=14, color="#0f172a"),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Journey_Label']}</b><br>"
                    f"Concepts: {row['Concepts']}<br>"
                    f"Subjects: {row['Subjects']}<br>"
                    f"Units: {row['Units']}<extra></extra>"
                ),
            )
        )
        fig.add_annotation(
            x=0.15,
            y=y,
            text=(
                f"<b>{row['Concepts']}</b> concepts<br>"
                f"<b>{row['Subjects']}</b> subjects<br>"
                f"<b>{row['Units']}</b> units"
            ),
            showarrow=False,
            xanchor="right",
            font=dict(size=12, color="#475569"),
            align="right",
        )
        if index < len(timeline) - 1:
            fig.add_annotation(
                x=0.5,
                y=y - 0.45,
                ax=0.5,
                ay=y_positions[index + 1] + 0.45,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#94a3b8",
                showarrow=True,
            )

    fig.update_xaxes(visible=False, range=[0, 1.2])
    fig.update_yaxes(visible=False, range=[-0.8, len(timeline) - 0.2])
    _apply_layout(fig, "Learning Progression Timeline — What builds at each stage?", height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Educational question: How does curriculum breadth grow from secondary school to engineering?"
    )


def _render_concept_growth(frame: pd.DataFrame) -> None:
    growth = concept_growth_by_stage(frame)
    if growth.empty:
        return
    colors = [CHART_COLORS.get(stage, "#64748b") for stage in growth["Learning_Stage"]]
    fig = go.Figure(
        go.Bar(
            x=growth["Learning_Stage"],
            y=growth["Concept_Count"],
            marker_color=colors,
            text=growth["Concept_Count"],
            textposition="outside",
        )
    )
    _apply_layout(fig, "Concept Growth Across Education Levels", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Educational question: Does curriculum complexity increase at higher stages?")


def _render_skill_domain_evolution(frame: pd.DataFrame) -> None:
    skill_data = skill_domain_by_class(frame)
    if skill_data.empty:
        return
    fig = px.bar(
        skill_data,
        x="Journey_Label",
        y="Count",
        color="Skill_Domain",
        color_discrete_map=SKILL_COLORS,
        barmode="stack",
        category_orders={"Journey_Label": journey_labels_present(frame), "Skill_Domain": list(SKILL_COLORS)},
    )
    _apply_layout(fig, "Skill Domain Evolution Across Classes", height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Educational question: Which capabilities develop at each class level?")


def _render_subject_contribution(frame: pd.DataFrame) -> None:
    contribution = subject_skill_contribution(frame)
    if contribution.empty:
        st.info("No subject-skill data available.")
        return
    fig = px.bar(
        contribution,
        x="Subject",
        y="Count",
        color="Skill_Domain",
        color_discrete_map=SKILL_COLORS,
        barmode="stack",
    )
    _apply_layout(fig, "Subject Contribution to Skill Domains", height=460)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Educational question: Which subjects build which competencies?")


def _render_difficulty_progression(frame: pd.DataFrame) -> None:
    difficulty = difficulty_by_stage(frame)
    if difficulty.empty:
        return
    fig = px.bar(
        difficulty,
        x="Learning_Stage",
        y="Count",
        color="Difficulty",
        color_discrete_map=DIFFICULTY_COLORS,
        barmode="stack",
        category_orders={"Learning_Stage": STAGE_ORDER, "Difficulty": list(DIFFICULTY_COLORS)},
    )
    _apply_layout(fig, "Difficulty Progression — Increasing Academic Rigor", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Educational question: How does academic rigor intensify over time?")


def _render_top_concepts(frame: pd.DataFrame) -> None:
    ranked = top_concepts_by_stage(frame)
    cols = st.columns(3)
    stage_titles = {
        "Secondary School": "Secondary School (Classes 9–10)",
        "Higher Secondary": "Higher Secondary (Classes 11–12)",
        "Engineering": "Engineering (AICTE)",
    }
    for column, stage in zip(cols, STAGE_ORDER):
        data = ranked.get(stage, pd.DataFrame())
        with column:
            if data.empty:
                st.info(f"No concepts for {stage_titles[stage]}.")
                continue
            fig = go.Figure(
                go.Bar(
                    x=data["Count"],
                    y=data["Core_Concept"],
                    orientation="h",
                    marker_color=CHART_COLORS.get(stage, "#64748b"),
                    text=data["Count"],
                    textposition="outside",
                )
            )
            fig.update_yaxes(categoryorder="total ascending")
            _apply_layout(fig, f"Top Concepts — {stage_titles[stage]}", height=420)
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Educational question: What are the dominant knowledge areas at each stage?")


def _render_engineering_readiness(frame: pd.DataFrame) -> None:
    labels, sources, targets, values, link_labels = engineering_readiness_links(frame)
    if not labels:
        st.info("Add engineering syllabus data to visualize readiness pathways.")
        return

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=18,
                line=dict(color="#cbd5e1", width=0.5),
                label=labels,
                color=["#3b82f6" if "Engineering" not in label else "#14b8a6" for label in labels],
            ),
            link=dict(source=sources, target=targets, value=values, label=link_labels, color="rgba(37, 99, 235, 0.28)"),
        )
    )
    _apply_layout(fig, "Engineering Readiness Map — School to Engineering Pathways", height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Educational question: How does school learning prepare students for engineering?")


def _render_coverage_heatmap(frame: pd.DataFrame) -> None:
    heatmap = curriculum_coverage_heatmap(frame)
    if heatmap.empty:
        st.info("No coverage data available.")
        return
    fig = go.Figure(
        go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns.tolist(),
            y=heatmap.index.tolist(),
            colorscale=[[0, "#eff6ff"], [0.5, "#3b82f6"], [1, "#1e3a8a"]],
            text=heatmap.values,
            texttemplate="%{text}",
            hovertemplate="Subject: %{y}<br>Level: %{x}<br>Concepts: %{z}<extra></extra>",
        )
    )
    _apply_layout(fig, "Curriculum Coverage Heatmap — Subject Focus by Class", height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Educational question: Where is curriculum depth concentrated?")


def _render_radar_chart(frame: pd.DataFrame) -> None:
    compare_labels = []
    for key, label in JOURNEY_LEVELS:
        if key in set(frame["Class"].astype(str)):
            compare_labels.append(label)
    if len(compare_labels) < 2:
        compare_labels = journey_labels_present(frame)[:3]

    priority = ["Class 9", "Class 12", "Engineering (AICTE)"]
    selected = [label for label in priority if label in compare_labels]
    if len(selected) < 2:
        selected = compare_labels[:3]

    fig = go.Figure()
    for label in selected:
        profile = radar_skill_profile(frame, label)
        values = list(profile.values())
        categories = list(profile.keys())
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=label,
                opacity=0.65,
                line_color=CHART_COLORS.get(label, "#64748b"),
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Skill Domain Radar — Capability Evolution", x=0, xanchor="left"),
        height=500,
        polar=dict(radialaxis=dict(visible=True, gridcolor="#e2e8f0")),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Educational question: How do student capabilities evolve from early to advanced stages?")


def _render_learning_journey(frame: pd.DataFrame) -> None:
    available = [(key, label) for key, label in JOURNEY_LEVELS if key in set(frame["Class"].astype(str))]
    if not available:
        st.info("No class data available for exploration.")
        return

    class_keys = [key for key, _ in available]
    class_labels = [label for _, label in available]
    selected_label = st.selectbox("Select educational level", class_labels, key="journey_select")
    selected_key = class_keys[class_labels.index(selected_label)]
    detail = journey_detail(frame, selected_key)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Learning Stage", detail.get("stage", "—"))
    m2.metric("Subjects", len(detail["subjects"]))
    m3.metric("Top Concepts", len(detail["concepts"]))
    m4.metric("Curriculum Units", detail["units"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Subjects at this level**")
        for subject in detail["subjects"]:
            st.markdown(f"- {subject}")

        st.markdown("**Top concepts**")
        for concept in detail["concepts"][:10]:
            st.markdown(f"- {concept}")

    with col2:
        if detail["skills"]:
            skill_df = pd.DataFrame(
                {"Skill Domain": list(detail["skills"].keys()), "Count": list(detail["skills"].values())}
            )
            fig = px.bar(skill_df, x="Skill Domain", y="Count", color="Skill Domain", color_discrete_map=SKILL_COLORS)
            _apply_layout(fig, f"Skills — {selected_label}", height=320)
            st.plotly_chart(fig, use_container_width=True)

        if detail["difficulty"]:
            diff_df = pd.DataFrame(
                {"Difficulty": list(detail["difficulty"].keys()), "Count": list(detail["difficulty"].values())}
            )
            fig = px.pie(diff_df, names="Difficulty", values="Count", color="Difficulty", color_discrete_map=DIFFICULTY_COLORS)
            fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="Difficulty Mix", x=0), height=280, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    explorer = frame[frame["Class"].astype(str) == selected_key]
    st.dataframe(
        explorer[
            ["Class", "Subject", "Chapter", "Core_Concept", "Skill_Domain", "Difficulty", "Learning_Stage", "Summary"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    frame = load_dataset()

    st.sidebar.title("Navigation")
    sections = {
        "Overview": "executive",
        "Learning Progression": "progression",
        "Skill Intelligence": "skills",
        "Concept Analytics": "concepts",
        "Engineering Readiness": "engineering",
        "Curriculum Explorer": "explorer",
    }
    choice = st.sidebar.radio("Section", list(sections.keys()))
    st.sidebar.divider()
    st.sidebar.caption(f"Dataset: `{CSV_PATH.name}`")
    st.sidebar.caption(f"Records: {len(frame)}")

    if frame.empty:
        st.warning("No curriculum data found. Run `python src/main.py` after adding syllabus PDFs to `pdfs/`.")
        return

    _render_hero()

    if choice == "Overview":
        _render_executive_summary(frame)
        _section("Learning Journey Snapshot", "Five-level progression from Class 9 to Engineering.")
        _render_learning_progression_timeline(frame)

    elif choice == "Learning Progression":
        _section("Learning Progression", "How knowledge and curriculum breadth evolve across levels.")
        _render_learning_progression_timeline(frame)
        c1, c2 = st.columns(2)
        with c1:
            _render_concept_growth(frame)
        with c2:
            _render_difficulty_progression(frame)

    elif choice == "Skill Intelligence":
        _section("Skill Intelligence", "How competencies develop and which subjects contribute.")
        _render_skill_domain_evolution(frame)
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            _render_subject_contribution(frame)
        with c2:
            _render_radar_chart(frame)

    elif choice == "Concept Analytics":
        _section("Concept Analytics", "Dominant knowledge areas and subject coverage patterns.")
        _render_top_concepts(frame)
        _render_coverage_heatmap(frame)

    elif choice == "Engineering Readiness":
        _section("Engineering Readiness", "Connections between school learning and engineering pathways.")
        _render_engineering_readiness(frame)
        st.markdown(
            """
            <div class="insight-card">
                <strong>How to read this map:</strong> Each pathway shows how foundational school concepts
                (algebra, programming, physics, chemistry) connect to engineering disciplines.
                Stronger links indicate more curriculum evidence supporting that transition.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif choice == "Curriculum Explorer":
        _section("Interactive Learning Journey", "Explore subjects, concepts, skills, and difficulty for any level.")
        _render_learning_journey(frame)


if __name__ == "__main__":
    main()
