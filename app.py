"""Main Streamlit application for AI Resume Screening System."""

import io
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.models.job import JobDescription

# Import our modules
from src.parsers.resume_parser import ResumeParser
from src.ranking.ranking_engine import RankingEngine

# Page configuration
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .candidate-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables."""
    if "processed_results" not in st.session_state:
        st.session_state.processed_results = None
    if "job_description" not in st.session_state:
        st.session_state.job_description = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []


def create_job_description_from_text(title: str, description: str) -> JobDescription:
    """Create JobDescription object from user input."""
    job_desc = JobDescription(title=title, description=description)

    # The JobDescription model will automatically extract basic skills
    # from the description in its __post_init__ method

    return job_desc


def process_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded files to temporary directory and return file paths."""
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    for uploaded_file in uploaded_files:
        # Save file to temporary location
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(temp_path)

    return file_paths


def display_candidate_card(candidate, rank):
    """Display a candidate card with key information."""
    resume = candidate.resume
    name = resume.contact_info.name if resume.contact_info else "Unknown"

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**#{rank} - {name}**")
            if resume.contact_info and resume.contact_info.email:
                st.text(f"📧 {resume.contact_info.email}")
            if resume.skills:
                st.text(f"🔧 {', '.join(resume.skills[:5])}{'...' if len(resume.skills) > 5 else ''}")

        with col2:
            st.metric("Overall Score", f"{candidate.hybrid_score:.1%}")
            st.metric("Semantic", f"{candidate.semantic_score:.1%}")

        with col3:
            st.metric("Skill Match", f"{candidate.skill_score:.1%}")
            if candidate.explanation:
                with st.expander("View Explanation"):
                    st.write(candidate.explanation)


def create_score_distribution_chart(candidates):
    """Create score distribution visualization."""
    scores_data = []
    for candidate in candidates:
        name = candidate.resume.contact_info.name if candidate.resume.contact_info else "Unknown"
        scores_data.append(
            {
                "Candidate": name[:20] + "..." if len(name) > 20 else name,
                "Semantic Score": candidate.semantic_score,
                "Skill Score": candidate.skill_score,
                "Hybrid Score": candidate.hybrid_score,
                "Rank": candidate.rank,
            }
        )

    df = pd.DataFrame(scores_data)

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(name="Semantic Score", x=df["Candidate"], y=df["Semantic Score"], marker_color="#667eea"))

    fig.add_trace(go.Bar(name="Skill Score", x=df["Candidate"], y=df["Skill Score"], marker_color="#764ba2"))

    fig.add_trace(go.Bar(name="Hybrid Score", x=df["Candidate"], y=df["Hybrid Score"], marker_color="#28a745"))

    fig.update_layout(
        title="Candidate Score Comparison", xaxis_title="Candidates", yaxis_title="Score", barmode="group", height=500
    )

    return fig


def create_skills_analysis_chart(candidates, job_desc):
    """Create skills analysis visualization."""
    # Analyze skill coverage
    required_skills = set(job_desc.required_skills)
    skill_coverage = {}

    for skill in required_skills:
        coverage_count = sum(1 for candidate in candidates if skill.lower() in [s.lower() for s in candidate.resume.skills])
        skill_coverage[skill] = coverage_count

    if skill_coverage:
        df = pd.DataFrame(list(skill_coverage.items()), columns=["Skill", "Candidates"])

        fig = px.bar(
            df,
            x="Skill",
            y="Candidates",
            title="Required Skills Coverage Across Candidates",
            color="Candidates",
            color_continuous_scale="viridis",
        )

        fig.update_layout(height=400)
        return fig

    return None


def display_fairness_report(fairness_report):
    """Display fairness analysis results."""
    if not fairness_report:
        st.info("No fairness analysis available")
        return

    st.subheader("🔍 Fairness Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Overall Fairness Score", f"{fairness_report.get_overall_fairness_score():.1%}")

        if fairness_report.bias_flags:
            st.warning("⚠️ Potential Bias Detected")
            for flag in fairness_report.bias_flags:
                st.text(f"• {flag}")
        else:
            st.success("✅ No significant bias detected")

    with col2:
        if fairness_report.recommendations:
            st.subheader("Recommendations")
            for rec in fairness_report.recommendations:
                st.text(f"• {rec}")


def generate_pdf_report(results, job_desc):
    """Generate a PDF report of the screening results."""
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"AI Resume Screening Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)

    # Job Info
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Job Information", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Title: {job_desc.title}", ln=True)
    # Removing non-ascii chars to avoid fpdf encoding issues
    req_skills = ", ".join(job_desc.required_skills).encode("ascii", "ignore").decode("ascii")
    pdf.multi_cell(0, 8, f"Required Skills: {req_skills}")
    pdf.ln(5)

    # Summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Screening Summary", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Total Resumes Processed: {results.successfully_parsed}", ln=True)
    if results.fairness_report:
        pdf.cell(0, 8, f"Overall Fairness Score: {results.fairness_report.get_overall_fairness_score():.1%}", ln=True)
    pdf.ln(5)

    # Top Candidates
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Top Rank Candidates", ln=True)

    for c in results.ranked_candidates[:10]:
        name = c.resume.contact_info.name if c.resume.contact_info else "Unknown"
        name = name.encode("ascii", "ignore").decode("ascii")
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"#{c.rank} - {name} (Score: {c.hybrid_score:.1%})", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 6, f"Semantic: {c.semantic_score:.1%} | Skill Match: {c.skill_score:.1%}", ln=True)

        if c.explanation:
            expl = c.explanation.replace("\n", " ").encode("ascii", "ignore").decode("ascii")
            pdf.multi_cell(0, 6, f"Summary: {expl}")
        pdf.ln(4)

    return pdf.output(dest="S").encode("latin1")


def create_candidate_comparison_chart(candidates):
    """Create a radar chart comparing selected highly ranked candidates."""
    if not candidates:
        return None

    fig = go.Figure()

    for candidate in candidates[:3]:  # Compare top 3
        name = candidate.resume.contact_info.name if candidate.resume.contact_info else "Unknown"

        fig.add_trace(
            go.Scatterpolar(
                r=[
                    candidate.semantic_score,
                    candidate.skill_score,
                    candidate.hybrid_score,
                    candidate.semantic_score,
                ],  # closing the loop
                theta=["Semantic", "Skill", "Hybrid", "Semantic"],
                fill="toself",
                name=name,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Candidate Profile Comparison (Top 3)"
    )
    return fig


def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🤖 AI Resume Screener</h1>', unsafe_allow_html=True)
    st.markdown("**Intelligent resume screening using semantic analysis and skill matching**")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Scoring weights
        st.subheader("Scoring Weights")
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.7, 0.1)
        skill_weight = 1.0 - semantic_weight
        st.text(f"Skill Weight: {skill_weight:.1f}")

        # Model selection
        st.subheader("Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"], index=0
        )

        # Processing options
        st.subheader("Processing Options")
        include_fairness = st.checkbox("Include Fairness Analysis", value=True)
        max_candidates = st.number_input("Max Candidates to Display", 1, 50, 10)

    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results", "📈 Analytics"])

    with tab1:
        st.header("Upload Resumes and Job Description")

        # Job description input
        col1, col2 = st.columns([2, 1])

        with col1:
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
            job_description = st.text_area(
                "Job Description",
                height=200,
                placeholder="Enter the complete job description including requirements, responsibilities, and preferred qualifications...",
            )

        with col2:
            st.info("""
            **Tips for better results:**
            - Include specific technical skills
            - Mention experience requirements
            - Add preferred qualifications
            - Use clear, descriptive language
            """)

        # File upload
        st.subheader("Upload Resume Files")
        uploaded_files = st.file_uploader(
            "Choose resume files", type=["pdf", "docx"], accept_multiple_files=True, help="Upload PDF or DOCX resume files"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")

            # Display uploaded files
            with st.expander("View Uploaded Files"):
                for file in uploaded_files:
                    st.text(f"📄 {file.name} ({file.size} bytes)")

        # Process button
        if st.button("🚀 Process Resumes", type="primary", disabled=not (job_title and job_description and uploaded_files)):
            with st.spinner("Processing resumes... This may take a few minutes."):
                try:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Initialize components
                    status_text.text("Initializing components...")
                    progress_bar.progress(10)

                    parser = ResumeParser()
                    embedding_generator = EmbeddingGenerator(model_name=embedding_model)
                    ranking_engine = RankingEngine(
                        semantic_weight=semantic_weight, skill_weight=skill_weight, embedding_generator=embedding_generator
                    )

                    # Create job description object
                    status_text.text("Processing job description...")
                    progress_bar.progress(20)

                    job_desc = create_job_description_from_text(job_title, job_description)

                    # Save uploaded files
                    status_text.text("Saving uploaded files...")
                    progress_bar.progress(30)

                    file_paths = process_uploaded_files(uploaded_files)

                    # Parse resumes
                    status_text.text("Parsing resumes...")
                    progress_bar.progress(50)

                    resumes = parser.batch_parse(file_paths)

                    # Process batch
                    status_text.text("Ranking candidates...")
                    progress_bar.progress(80)

                    results = ranking_engine.process_batch(resumes, job_desc, include_fairness=include_fairness)

                    # Store results
                    st.session_state.processed_results = results
                    st.session_state.job_description = job_desc

                    progress_bar.progress(100)
                    status_text.text("✅ Processing completed!")

                    st.success(f"Successfully processed {len(resumes)} resumes in {results.processing_time:.2f} seconds")

                    # Clean up temporary files
                    for file_path in file_paths:
                        try:
                            os.unlink(file_path)
                        except:
                            pass

                except Exception as e:
                    st.error(f"Error processing resumes: {str(e)}")
                    logger.error(f"Processing error: {str(e)}")

    with tab2:
        st.header("📊 Screening Results")

        if st.session_state.processed_results:
            results = st.session_state.processed_results
            candidates = results.ranked_candidates[:max_candidates]

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Resumes", results.total_resumes)

            with col2:
                st.metric("Successfully Parsed", results.successfully_parsed)

            with col3:
                st.metric("Processing Time", f"{results.processing_time:.2f}s")

            with col4:
                if candidates:
                    st.metric("Top Score", f"{candidates[0].hybrid_score:.1%}")

            # Display candidates
            st.subheader(f"🏆 Top {len(candidates)} Candidates")

            for i, candidate in enumerate(candidates, 1):
                with st.container():
                    display_candidate_card(candidate, i)
                    st.divider()

            # Fairness analysis
            if results.fairness_report:
                display_fairness_report(results.fairness_report)

            # Export options
            st.subheader("📥 Export Results")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📊 Export to CSV"):
                    # Create CSV data
                    csv_data = []
                    for candidate in results.ranked_candidates:
                        resume = candidate.resume
                        csv_data.append(
                            {
                                "Rank": candidate.rank,
                                "Name": resume.contact_info.name if resume.contact_info else "Unknown",
                                "Email": resume.contact_info.email if resume.contact_info else "",
                                "Hybrid Score": candidate.hybrid_score,
                                "Semantic Score": candidate.semantic_score,
                                "Skill Score": candidate.skill_score,
                                "Skills": ", ".join(resume.skills),
                                "Experience Years": resume.get_years_of_experience(),
                                "Explanation": candidate.explanation or "",
                            }
                        )

                    df = pd.DataFrame(csv_data)
                    csv = df.to_csv(index=False)

                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"resume_screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

            with col2:
                if st.button("📄 Generate Report"):
                    # Generate PDF bytes
                    try:
                        pdf_bytes = generate_pdf_report(results, st.session_state.job_description)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"resume_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                        )
                    except Exception as e:
                        st.error(f"Failed to generate PDF: {str(e)}")

        else:
            st.info("👆 Upload resumes and process them in the 'Upload & Process' tab to see results here.")

    with tab3:
        st.header("📈 Analytics & Insights")

        if st.session_state.processed_results:
            results = st.session_state.processed_results
            candidates = results.ranked_candidates
            job_desc = st.session_state.job_description

            # Score distribution chart
            if candidates:
                fig_scores = create_score_distribution_chart(candidates[:10])
                st.plotly_chart(fig_scores, use_container_width=True)

            # Skills analysis
            if candidates and job_desc:
                fig_skills = create_skills_analysis_chart(candidates, job_desc)
                if fig_skills:
                    st.plotly_chart(fig_skills, use_container_width=True)

            # Statistics
            if candidates:
                st.subheader("📊 Statistical Summary")

                scores = [c.hybrid_score for c in candidates]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Average Score", f"{sum(scores)/len(scores):.1%}")

                with col2:
                    st.metric("Median Score", f"{sorted(scores)[len(scores)//2]:.1%}")

                with col3:
                    st.metric("Score Range", f"{min(scores):.1%} - {max(scores):.1%}")

                # Candidate Comparison Radar
                st.subheader("Candidate Comparison")
                fig_compare = create_candidate_comparison_chart(candidates)
                if fig_compare:
                    st.plotly_chart(fig_compare, use_container_width=True)

        else:
            st.info("👆 Process some resumes first to see analytics here.")

    # Footer
    st.markdown("---")
    st.markdown(
        "**AI Resume Screener** - Built with Streamlit, Sentence Transformers, and ❤️ | "
        f"Model: {embedding_model} | "
        f"Weights: {semantic_weight:.1f}/{skill_weight:.1f}"
    )


if __name__ == "__main__":
    main()
