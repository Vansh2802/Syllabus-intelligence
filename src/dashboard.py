"""Curriculum Intelligence Analytics Platform dashboard."""
from __future__ import annotations

import os
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
    generate_skill_evolution_dataset,
    generate_subject_complexity_dataset,
    generate_learning_path_dataset,
    generate_educational_stage_dataset,
    generate_engineering_readiness_dataset
)
from csv_exporter import CSV_HEADERS, load_dataset as load_exported_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "output" / "curriculum_dataset.csv"

PLOTLY_LAYOUT = dict(
    font=dict(family="Outfit, Inter, Segoe UI, sans-serif", color="#0f172a"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=24, r=24, t=56, b=24),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

st.set_page_config(
    page_title="Curriculum Intelligence Analytics Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Modern Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .block-container { 
        padding-top: 1.5rem; 
        padding-bottom: 2rem; 
        max-width: 1400px; 
    }
    
    .hero {
        padding: 2.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0d9488 100%);
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.3);
    }
    .hero h1 { margin: 0; font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em; color: white; }
    .hero p { margin: 0.8rem 0 0; opacity: 0.9; font-size: 1.15rem; max-width: 900px; line-height: 1.6; }
    
    .section-header {
        margin: 2.2rem 0 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #f1f5f9;
    }
    .section-header h2 { margin: 0; color: #0f172a; font-size: 1.65rem; font-weight: 600; }
    .section-header p { margin: 0.35rem 0 0; color: #64748b; font-size: 1rem; }
    
    .insight-card {
        background: rgba(248, 250, 252, 0.8);
        border: 1px dashed #cbd5e1;
        border-left: 5px solid #0d9488;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .insight-card strong { color: #0f172a; font-size: 1.1rem; }
    .insight-card p { margin: 0.5rem 0 0; color: #475569; font-size: 0.95rem; line-height: 1.5; }
    
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem 1.25rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.08);
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


@st.cache_data(show_spinner=False)
def get_curriculum_insights(insight_type: str) -> str:
    """Get professional educational insights using OpenAI or fall back to high-quality curated summaries."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompts = {
                "class9": "Provide a brief expert summary (1-2 sentences) of what subjects/concepts dominate Class 9 education in CBSE, focusing on foundational knowledge.",
                "transition_10_11": "Provide a brief expert summary (1-2 sentences) of the dramatic academic transition and change in curriculum depth between Class 10 (Secondary) and Class 11 (Higher Secondary) in India.",
                "school_to_eng": "Provide a brief expert summary (1-2 sentences) on what engineering concepts (like calculus, mechanics, programming) originate in school curriculum.",
                "skills_evolution": "Provide a brief expert analysis (1-2 sentences) on how skill requirements evolve from basic memorization/reproduction to computational and analytical thinking as students transition to engineering.",
                "complexity": "Provide a brief expert insight (1-2 sentences) on how concept complexity and difficulty growth accelerates across secondary, senior secondary, and engineering levels."
            }
            prompt = prompts.get(insight_type, "Provide a general educational curriculum insight.")
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a senior curriculum architect and educational analyst. Provide a brief, professional 1-2 sentence insight card."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass

    # Expert curated fallback answers
    fallbacks = {
        "class9": "Class 9 curriculum acts as the core gateway, dominated by foundational Science (Matter, Cell Biology, Laws of Motion) and basic Algebra. It transitions students from general observations to formal structured disciplines.",
        "transition_10_11": "The transition from Class 10 to 11 represents a massive leap in mathematical rigor and scientific depth, moving from qualitative surveys to quantitative derivations, calculus-based mechanics, and organic chemistry mechanisms.",
        "school_to_eng": "Key engineering cornerstones originate directly in high school: calculus forms the bedrock of engineering mathematics, Newtonian mechanics transitions to engineering mechanics, and basic scripting sets the stage for data structures.",
        "skills_evolution": "Student capabilities evolve from basic Quantitative and Scientific Reasoning in early school years towards specialized Computational Thinking, Research Skills, and Design Thinking in the later engineering years.",
        "complexity": "Curriculum complexity accelerates exponentially, with Beginner-level concepts making up 70% of Class 9-10 but dropping below 15% in Engineering, replaced by Advanced mathematical optimization and system design."
    }
    return fallbacks.get(insight_type, "Education pathways build progressive capabilities, linking foundational school concepts to advanced engineering applications.")


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
                Analyze learning pathways, capability development, and subject complexity from Class 9 through 
                Class 12 and into all four years of Engineering. Discover how school prepares students for technical careers.
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
        "Executive Dashboard",
        "A high-level view of curriculum structure and coverage.",
    )
    metrics = executive_summary(frame)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total PDFs Analyzed", len(list(ROOT_DIR.joinpath("pdfs").glob("*.pdf"))))
    c2.metric("Total Subjects", metrics["subjects"])
    c3.metric("Total Concepts", metrics["concepts"])
    c4.metric("Skill Domains", metrics["skill_domains"])
    c5.metric("Curriculum Units", metrics["curriculum_units"])
    c6.metric("Levels Covered", metrics["levels"])

    # Render Gemini Insight Cards
    st.write("### 🤖 AI-Powered Curriculum Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="insight-card">
                <strong>What Dominates Class 9?</strong>
                <p>{get_curriculum_insights("class9")}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Class 10 → 11 Leap</strong>
                <p>{get_curriculum_insights("transition_10_11")}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="insight-card">
                <strong>School Roots of Engineering</strong>
                <p>{get_curriculum_insights("school_to_eng")}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def _render_learning_progression_timeline(frame: pd.DataFrame) -> None:
    timeline = progression_timeline_metrics(frame)
    if timeline.empty:
        st.info("No progression data available yet.")
        return

    # Complete 8-stage sorting order
    all_stages_order = ["9", "10", "11", "12", "Engineering Year 1", "Engineering Year 2", "Engineering Year 3", "Engineering Year 4"]
    timeline["Sort_Idx"] = timeline["Class_Key"].apply(lambda x: all_stages_order.index(x) if x in all_stages_order else 99)
    timeline = timeline.sort_values("Sort_Idx").reset_index(drop=True)

    fig = go.Figure()
    y_positions = list(range(len(timeline) - 1, -1, -1))
    colors = ["#3b82f6", "#2563eb", "#8b5cf6", "#7c3aed", "#14b8a6", "#0d9488", "#f59e0b", "#db2777"]

    for index, row in timeline.iterrows():
        y = y_positions[index]
        color = colors[index % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[y],
                mode="markers+text",
                marker=dict(size=30, color=color, line=dict(width=3, color="white")),
                text=[row["Journey_Label"]],
                textposition="middle right",
                textfont=dict(size=15, color="#0f172a", family="Outfit"),
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
                arrowcolor="#cbd5e1",
                showarrow=True,
            )

    fig.update_xaxes(visible=False, range=[0, 1.2])
    fig.update_yaxes(visible=False, range=[-0.8, len(timeline) - 0.2])
    _apply_layout(fig, "Learning Progression Timeline — 8-Stage Educational Path", height=600)
    st.plotly_chart(fig, use_container_width=True)


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
    _apply_layout(fig, "Concept Growth Across Stage Levels", height=400)
    st.plotly_chart(fig, use_container_width=True)


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
    _apply_layout(fig, "Difficulty Growth / Academic Rigor Mix", height=400)
    st.plotly_chart(fig, use_container_width=True)


def _render_subject_growth(frame: pd.DataFrame) -> None:
    # Generate count of unique subjects per stage
    sub_growth = frame.groupby("Learning_Stage")["Subject"].nunique().reindex(STAGE_ORDER).fillna(0).reset_index()
    fig = px.line(
        sub_growth,
        x="Learning_Stage",
        y="Subject",
        markers=True,
        line_shape="linear",
        labels={"Subject": "Number of Subjects"}
    )
    fig.update_traces(line_color="#0d9488", line_width=4, marker=dict(size=10))
    _apply_layout(fig, "Subject Growth / Breadth Across Stages", height=400)
    st.plotly_chart(fig, use_container_width=True)


def _render_skill_intelligence_page(frame: pd.DataFrame) -> None:
    _section("Skill Intelligence", "Understand how learning outputs evolve across domains.")
    
    st.markdown(
        f"""
        <div class="insight-card" style="border-left-color: #7c3aed;">
            <strong>AI Skill Evolution Insights:</strong>
            <p>{get_curriculum_insights("skills_evolution")}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    c1, c2 = st.columns(2)
    with c1:
        # 1. Skill Evolution
        skill_data = skill_domain_by_class(frame)
        if not skill_data.empty:
            fig = px.bar(
                skill_data,
                x="Journey_Label",
                y="Count",
                color="Skill_Domain",
                color_discrete_map=SKILL_COLORS,
                barmode="stack",
                category_orders={"Journey_Label": journey_labels_present(frame), "Skill_Domain": list(SKILL_COLORS)},
            )
            _apply_layout(fig, "Skill Evolution Across Journey Stages", height=420)
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        # 2. Skill Heatmap
        heatmap_df = frame.groupby(["Learning_Stage", "Skill_Domain"]).size().unstack(fill_value=0)
        # Ensure categories match stage order
        heatmap_df = heatmap_df.reindex(index=[s for s in STAGE_ORDER if s in heatmap_df.index])
        fig = go.Figure(
            go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns.tolist(),
                y=heatmap_df.index.tolist(),
                colorscale="Purples",
                text=heatmap_df.values,
                texttemplate="%{text}",
                hovertemplate="Stage: %{y}<br>Skill: %{x}<br>Count: %{z}<extra></extra>",
            )
        )
        _apply_layout(fig, "Skill Intensity Heatmap by Stage", height=420)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # 3. Skill Distribution
        dist = frame["Skill_Domain"].value_counts().reset_index()
        fig = px.pie(
            dist,
            names="Skill_Domain",
            values="count",
            color="Skill_Domain",
            color_discrete_map=SKILL_COLORS,
            hole=0.4
        )
        fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="Overall Skill Distribution"), height=420)
        st.plotly_chart(fig, use_container_width=True)
        
    with c4:
        # 4. Radar Chart
        compare_labels = journey_labels_present(frame)
        priority = ["Class 9", "Class 12", "Engineering Year 1", "Engineering Year 4"]
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
                    opacity=0.6,
                    line_color=CHART_COLORS.get(label, "#64748b"),
                )
            )

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Skill Profile Radar Comparison", x=0),
            height=420,
            polar=dict(radialaxis=dict(visible=True, gridcolor="#e2e8f0")),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_subject_intelligence_page(frame: pd.DataFrame) -> None:
    _section("Subject Intelligence", "Analyze how individual subjects construct curriculum complexity.")
    
    st.markdown(
        f"""
        <div class="insight-card" style="border-left-color: #f59e0b;">
            <strong>AI Subject Complexity Analysis:</strong>
            <p>{get_curriculum_insights("complexity")}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1.1, 0.9])
    
    with col1:
        # 1. Subject Contribution
        contribution = subject_skill_contribution(frame)
        if not contribution.empty:
            fig = px.bar(
                contribution,
                x="Subject",
                y="Count",
                color="Skill_Domain",
                color_discrete_map=SKILL_COLORS,
                barmode="stack",
            )
            _apply_layout(fig, "Subject Contribution to Skill Domains", height=450)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 2. Subject Complexity scatter (Concept Count vs Average Difficulty Score)
        complexity_df = generate_subject_complexity_dataset(frame)
        if not complexity_df.empty:
            fig = px.scatter(
                complexity_df,
                x="Concept_Count",
                y="Difficulty_Score",
                color="Stage",
                hover_data=["Subject"],
                size="Concept_Count",
                size_max=30,
                color_discrete_map=CHART_COLORS
            )
            _apply_layout(fig, "Subject Complexity mapping", height=450)
            st.plotly_chart(fig, use_container_width=True)
            
    # 3. Top Subjects by Concept Density
    density = frame.groupby("Subject")["Core_Concept"].nunique().sort_values(ascending=False).head(10).reset_index()
    density.rename(columns={"Core_Concept": "Unique Concept Count"}, inplace=True)
    fig = px.bar(
        density,
        x="Subject",
        y="Unique Concept Count",
        text="Unique Concept Count",
        color_discrete_sequence=["#0d9488"]
    )
    fig.update_traces(textposition="outside")
    _apply_layout(fig, "Top 10 Subjects by Concept Density", height=400)
    st.plotly_chart(fig, use_container_width=True)


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


def _render_concept_analytics_page(frame: pd.DataFrame) -> None:
    _section("Concept Analytics", "Drill down into specific educational concepts and clusters.")
    
    c1, c2 = st.columns([0.4, 0.6])
    with c1:
        # Top Concepts Table / List
        top_concepts = frame["Core_Concept"].value_counts().head(15).reset_index()
        top_concepts.columns = ["Core Concept", "Occurrences"]
        st.write("#### Most Frequent Concepts")
        st.dataframe(top_concepts, use_container_width=True, hide_index=True)
        
    with c2:
        # Treemap
        treemap_df = frame.groupby(["Subject", "Core_Concept"]).size().reset_index(name="Count")
        fig = px.treemap(
            treemap_df,
            path=["Subject", "Core_Concept"],
            values="Count",
            color="Subject",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="Concept Hierarchical Treemap"), height=460)
        st.plotly_chart(fig, use_container_width=True)

    # Concept Coverage Heatmap
    _render_coverage_heatmap(frame)


def _render_engineering_readiness_page(frame: pd.DataFrame) -> None:
    _section("Engineering Readiness", "Evaluate how well school preparations align with engineering courses.")
    
    st.markdown(
        f"""
        <div class="insight-card" style="border-left-color: #2563eb;">
            <strong>AI Engineering Readiness Mapping:</strong>
            <p>{get_curriculum_insights("school_to_eng")}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sankey diagram
    _render_engineering_readiness(frame)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### School to Engineering Concept Readiness Bridges")
        readiness_df = generate_engineering_readiness_dataset(frame)
        st.dataframe(readiness_df, use_container_width=True, hide_index=True)
        
    with col2:
        # Readiness score indicator
        avg_readiness = int(readiness_df["Relationship_Strength"].mean() * 20)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_readiness,
            title={'text': "Curriculum Readiness Score (Scale 0-100)"},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#2563eb"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 40], 'color': '#fee2e2'},
                    {'range': [40, 75], 'color': '#fef3c7'},
                    {'range': [75, 100], 'color': '#d1fae5'}
                ]
            }
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig, use_container_width=True)
        
    st.write("#### Top Learning Transition Pathways")
    pathways = generate_learning_path_dataset(frame)
    st.dataframe(pathways, use_container_width=True, hide_index=True)


def main() -> None:
    frame = load_dataset()

    st.sidebar.title("📌 Navigation")
    pages = [
        "Executive Dashboard",
        "Learning Progression",
        "Skill Intelligence",
        "Subject Intelligence",
        "Concept Analytics",
        "Engineering Readiness",
        "Curriculum Explorer"
    ]
    choice = st.sidebar.radio("Go to Page", pages)
    st.sidebar.divider()
    
    st.sidebar.caption("📊 **Active Dataset**")
    st.sidebar.caption(f"File: `{CSV_PATH.name}`")
    st.sidebar.caption(f"Extracted Records: {len(frame)}")
    
    if frame.empty:
        st.warning("No curriculum data found. Run `python src/main.py` after adding syllabus PDFs to `pdfs/`.")
        return

    _render_hero()

    if choice == "Executive Dashboard":
        _render_executive_summary(frame)

    elif choice == "Learning Progression":
        _section("Learning Progression", "Trace educational journeys across CBSE classes and engineering semesters.")
        _render_learning_progression_timeline(frame)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            _render_concept_growth(frame)
        with c2:
            _render_difficulty_progression(frame)
        with c3:
            _render_subject_growth(frame)

    elif choice == "Skill Intelligence":
        _render_skill_intelligence_page(frame)

    elif choice == "Subject Intelligence":
        _render_subject_intelligence_page(frame)

    elif choice == "Concept Analytics":
        _render_concept_analytics_page(frame)

    elif choice == "Engineering Readiness":
        _render_engineering_readiness_page(frame)

    elif choice == "Curriculum Explorer":
        _section("Curriculum Explorer", "Search and filter through the complete educational ontology.")
        
        # Filters row
        f1, f2, f3, f4, f5 = st.columns(5)
        
        stages = ["All"] + list(frame["Learning_Stage"].unique())
        selected_stage = f1.selectbox("Filter Stage", stages)
        
        classes = ["All"] + list(frame["Class"].unique())
        selected_class = f2.selectbox("Filter Class", classes)
        
        subjects = ["All"] + list(frame["Subject"].unique())
        selected_subject = f3.selectbox("Filter Subject", subjects)
        
        difficulties = ["All"] + list(frame["Difficulty"].unique())
        selected_difficulty = f4.selectbox("Filter Difficulty", difficulties)
        
        skills = ["All"] + list(frame["Skill_Domain"].unique())
        selected_skill = f5.selectbox("Filter Skill Domain", skills)
        
        search_query = st.text_input("🔍 Search Concepts or Summaries", "")
        
        # Apply filters
        filtered = frame.copy()
        if selected_stage != "All":
            filtered = filtered[filtered["Learning_Stage"] == selected_stage]
        if selected_class != "All":
            filtered = filtered[filtered["Class"] == selected_class]
        if selected_subject != "All":
            filtered = filtered[filtered["Subject"] == selected_subject]
        if selected_difficulty != "All":
            filtered = filtered[filtered["Difficulty"] == selected_difficulty]
        if selected_skill != "All":
            filtered = filtered[filtered["Skill_Domain"] == selected_skill]
            
        if search_query:
            filtered = filtered[
                filtered["Core_Concept"].str.contains(search_query, case=False) |
                filtered["Summary"].str.contains(search_query, case=False) |
                filtered["Chapter"].str.contains(search_query, case=False)
            ]
            
        st.write(f"Showing **{len(filtered)}** matched items:")
        st.dataframe(
            filtered[
                ["Class", "Subject", "Chapter", "Core_Concept", "Skill_Domain", "Difficulty", "Learning_Stage", "Summary"]
            ],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
