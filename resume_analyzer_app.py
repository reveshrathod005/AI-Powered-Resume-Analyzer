import os
import json

import streamlit as st

from dotenv import load_dotenv

from utils import extract_pdf, create_vector_text

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# Page configuration
st.set_page_config(
    page_title="🤖 AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("🤖 AI-Powered Resume Analyzer")

st.write(
    "Analyze your resume against a job description using RAG and Gemini AI."
)

st.divider()


# Input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Resume")

    resume_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"]
    )

with col2:
    st.subheader("💼 Job Description")

    jd_text = st.text_area(
        "Paste Job Description",
        height=200
    )


# Analyze
if st.button(
    "✨ Analyze Resume",
    type="primary",
    use_container_width=True
):

    if resume_file and jd_text:

        # Extract Resume
        with st.spinner("📄 Reading your resume..."):
            resume_text = extract_pdf(resume_file)

        # Create FAISS Vector Store
        with st.spinner("🔍 Creating semantic search index..."):
            vectorstore = create_vector_text(resume_text)

        # Create Retriever
        retriever = vectorstore.as_retriever()

        # LLM Integration
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )

        # Prompt Template
        prompt = ChatPromptTemplate.from_template("""
        You are an AI Resume Analyzer.

        Analyze the resume against the given job description.

        Resume Context:
        {context}

        Job Description:
        {jd_text}

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "ats_score": 0,

            "match_overview": {{
                "skills": 0,
                "experience": 0,
                "education": 0,
                "keywords": 0
            }},

            "skill_match": {{
                "Python": 0,
                "SQL": 0,
                "Machine Learning": 0
            }},

            "skill_gaps": [
                "skill 1",
                "skill 2"
            ],

            "overall_summary": "Short explanation of the overall match.",

            "skill_summary": "Short explanation of the skill match.",

            "gap_summary": "Short explanation of the skill gaps.",

            "strengths": [
                "strength 1",
                "strength 2",
                "strength 3"
            ],

            "weaknesses": [
                "weakness 1",
                "weakness 2"
            ],

            "recommendations": [
                "recommendation 1",
                "recommendation 2",
                "recommendation 3"
            ],

            "interview_questions": [
                "question 1",
                "question 2",
                "question 3",
                "question 4",
                "question 5"
            ]
        }}

        Rules:

        - ats_score must be between 0 and 100.
        - All match scores must be between 0 and 100.
        - Include only important skills from the Job Description.
        - Do not invent skills, experience, projects, certifications,
          education, or achievements.
        - Skill scores must be based only on evidence from the resume.
        - If something is not mentioned in the resume, treat it as missing.
        - Keep summaries short and clear.
        - Keep recommendations practical.
        - Interview questions must be relevant to the resume and Job Description.
        - Return ONLY JSON. Do not use markdown or ```json.
        """)

        # RAG Chain
        chain = (
            {
                "context": retriever,
                "jd_text": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Run Chain
        with st.spinner("🤖 Analyzing your resume..."):
            response = chain.invoke(jd_text)

        try:

            result = json.loads(response)

            # Display Result
            st.divider()

            st.header("📊 Resume Analysis")


            # ATS Score
            st.subheader("🎯 ATS Compatibility")

            ats_score = result["ats_score"]

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:

                st.metric(
                    "ATS Score",
                    f"{ats_score}%"
                )

                st.progress(
                    ats_score / 100
                )

            st.write(
                result["overall_summary"]
            )


            st.divider()


            # Match Overview
            st.subheader("📊 Match Overview")

            match_data = result["match_overview"]

            match_chart = [
                {
                    "Category": "Skills",
                    "Score": match_data["skills"]
                },
                {
                    "Category": "Experience",
                    "Score": match_data["experience"]
                },
                {
                    "Category": "Education",
                    "Score": match_data["education"]
                },
                {
                    "Category": "Keywords",
                    "Score": match_data["keywords"]
                }
            ]

            st.vega_lite_chart(
                match_chart,
                {
                    "mark": "bar",
                    "encoding": {
                        "y": {
                            "field": "Category",
                            "type": "nominal",
                            "sort": "-x"
                        },
                        "x": {
                            "field": "Score",
                            "type": "quantitative",
                            "scale": {
                                "domain": [0, 100]
                            },
                            "title": "Match Score (%)"
                        },
                        "tooltip": [
                            {
                                "field": "Category",
                                "type": "nominal"
                            },
                            {
                                "field": "Score",
                                "type": "quantitative"
                            }
                        ]
                    }
                },
                use_container_width=True
            )

            st.write(
                f"**Skills:** {match_data['skills']}%  |  "
                f"**Experience:** {match_data['experience']}%  |  "
                f"**Education:** {match_data['education']}%  |  "
                f"**Keywords:** {match_data['keywords']}%"
            )


            st.divider()


            # Skill Match
            st.subheader("🧩 Skill Match")

            skill_match = result["skill_match"]

            skill_chart = [
                {
                    "Skill": skill,
                    "Score": score
                }
                for skill, score in skill_match.items()
            ]

            st.vega_lite_chart(
                skill_chart,
                {
                    "mark": "bar",
                    "encoding": {
                        "y": {
                            "field": "Skill",
                            "type": "nominal",
                            "sort": "-x"
                        },
                        "x": {
                            "field": "Score",
                            "type": "quantitative",
                            "scale": {
                                "domain": [0, 100]
                            },
                            "title": "Skill Match (%)"
                        },
                        "tooltip": [
                            {
                                "field": "Skill",
                                "type": "nominal"
                            },
                            {
                                "field": "Score",
                                "type": "quantitative"
                            }
                        ]
                    }
                },
                use_container_width=True
            )

            st.write(
                result["skill_summary"]
            )


            st.divider()


            # Skill Gaps
            st.subheader("⚠️ Skill Gaps")

            if result["skill_gaps"]:

                for skill in result["skill_gaps"]:
                    st.warning(skill)

                st.write(
                    result["gap_summary"]
                )

            else:

                st.success(
                    "No major skill gaps identified."
                )


            st.divider()


            # Strengths and Weaknesses
            col1, col2 = st.columns(2)

            with col1:

                st.subheader("💪 Resume Strengths")

                for strength in result["strengths"]:
                    st.success(strength)

            with col2:

                st.subheader("⚠️ Resume Weaknesses")

                for weakness in result["weaknesses"]:
                    st.warning(weakness)


            st.divider()


            # Recommendations
            st.subheader("💡 Recommendations")

            for recommendation in result["recommendations"]:
                st.info(recommendation)


            st.divider()


            # Interview Questions
            st.subheader("🎯 Interview Questions")

            for i, question in enumerate(
                result["interview_questions"],
                start=1
            ):

                with st.expander(
                    f"Question {i}"
                ):

                    st.write(question)


        except json.JSONDecodeError:

            st.error(
                "Unable to process the AI response. Please try again."
            )

            st.write(response)


    else:

        st.warning(
            "Please upload a resume and job description."
        )