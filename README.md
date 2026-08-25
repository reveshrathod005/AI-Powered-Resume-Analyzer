# 🤖 AI-Powered Resume Analyzer

<p align="center"><em>github.com/reveshrathod005/AI-Powered-Resume-Analyzer</em></p>

### Stop guessing why your resume gets rejected — get evidence-based, RAG-grounded feedback in seconds.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-RAG%20Pipeline-1C3C3C?style=for-the-badge">
  <img alt="Gemini" src="https://img.shields.io/badge/Google%20Gemini-LLM-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white">
  <img alt="FAISS" src="https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</p>

---

## 🎥 Demo

https://ai-powered-resume-analyzer-rrev.streamlit.app/

---

## 📝 Description

Job seekers routinely struggle to understand how well their resume actually aligns with a specific job description — manual, line-by-line comparison is slow, error-prone, and offers no real preparation for the interview that follows. **AI-Powered Resume Analyzer** solves this with a **Retrieval-Augmented Generation (RAG)** pipeline: your resume is chunked, embedded, and semantically retrieved against the job description, then analyzed by **Google Gemini** to produce a structured, evidence-based breakdown — not a hallucinated guess. The result is a single dashboard with an ATS compatibility score, skill-gap analysis, and tailored interview questions, generated only from what's actually in your resume.

---

## ✨ Features

- 🎯 **ATS Compatibility Score** — an AI-driven overall match score between resume and job description
- 📊 **Match Overview** — breakdown across Skills, Experience, Education, and Keywords
- 🧩 **Skill Match Analysis** — per-skill evaluation of how strongly each JD skill is demonstrated
- ⚠️ **Skill Gap Detection** — critical missing or weakly demonstrated skills, clearly flagged
- 💪 **Strengths & Weaknesses** — objective assessment of resume-to-role relevance
- 💡 **Actionable Recommendations** — practical, resume-specific improvement suggestions
- 🎤 **Custom Interview Questions** — prep questions tailored to the candidate and the JD
- 🔒 **Evidence-Based Guardrails** — the model is grounded strictly on retrieved resume context; it never invents skills, experience, or credentials, and explicitly flags what isn't mentioned
- 📈 **Interactive Visual Dashboard** — Vega-Lite charts for match scores and skill breakdowns, rendered natively in Streamlit

---

## 🛠️ Tech Stack

**Core Language & Framework**
- 🐍 Python — core application logic
- 🎈 Streamlit — interactive web UI and layout

**RAG / AI Pipeline**
- 🔗 LangChain — RAG pipeline orchestration (retriever → prompt → LLM → parser)
- 💎 Google Gemini (`langchain-google-genai`) — large language model for analysis generation
- 🧠 HuggingFace Embeddings (`all-MiniLM-L6-v2`) — semantic embeddings for resume chunks
- 📦 FAISS — in-memory vector store for semantic similarity search
- ✂️ `RecursiveCharacterTextSplitter` — resume chunking for retrieval

**Document & Data Handling**
- 📄 `pypdf` — PDF text extraction from uploaded resumes
- 🔐 `python-dotenv` — environment variable / API key management
- 📊 Vega-Lite (via `st.vega_lite_chart`) — score and skill-match visualizations

**Experimented With (R&D Sandbox)**
- 🦙 Ollama / local Gemma models — explored for offline LLM inference capability

---

## 📦 Installation & Setup

### Prerequisites

- Python **3.10+**
- A **Google AI Studio (Gemini) API key** → [Get one here](https://aistudio.google.com/app/apikey)
- `pip` for dependency management (a virtual environment is recommended)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/ai-resume-analyzer.git
   cd ai-resume-analyzer
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv

   # macOS / Linux
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If you don't yet have a `requirements.txt`, install the core packages directly:
   ```bash
   pip install streamlit python-dotenv pypdf \
       langchain langchain-core langchain-text-splitters \
       langchain-google-genai langchain-huggingface \
       langchain-community faiss-cpu sentence-transformers
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run resume_analyzer_app.py
   ```

   The app will open automatically in your browser at `http://localhost:8501`.

---

## 🚀 Usage

1. Launch the app with `streamlit run resume_analyzer_app.py`.
2. **Upload your resume** as a PDF in the left panel.
3. **Paste the job description** you're targeting in the right panel.
4. Click **✨ Analyze Resume**.
5. Review your results:
   - 🎯 ATS Compatibility Score
   - 📊 Match Overview (Skills / Experience / Education / Keywords)
   - 🧩 Skill Match chart
   - ⚠️ Skill Gaps
   - 💪 Strengths & Weaknesses
   - 💡 Recommendations
   - 🎤 Tailored Interview Questions

**Minimal code path** (how the RAG chain is wired under the hood):

```python
from utils import extract_pdf, create_vector_text

resume_text = extract_pdf(resume_file)          # PDF -> raw text
vectorstore = create_vector_text(resume_text)    # chunk -> embed -> FAISS index
retriever = vectorstore.as_retriever()           # semantic retriever

chain = (
    {"context": retriever, "jd_text": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke(job_description_text)      # structured JSON analysis
```

---

## 📁 Project Structure

```
ai-resume-analyzer/
│
├── resume_analyzer_app.py     # Streamlit app: UI, RAG chain, prompt, results dashboard
├── utils.py                   # PDF extraction, text chunking, FAISS vector store creation
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed — see .gitignore)
└── .gitignore                 # Ignored files (venv, .env, __pycache__, etc.)
```

---

## 🧭 Architecture Overview

```
Resume PDF ─┐
            ├─▶ Extract Text ─▶ Chunk ─▶ Embed (HuggingFace) ─▶ Store in FAISS
Job Desc ───┘                                                        │
                                                          Semantic Retrieval
                                                                       │
                                             Context + JD ─▶ Gemini (LLM)
                                                                       │
                                              Structured JSON Analysis
                                                                       │
                                                  Streamlit Dashboard
```

The system is deliberately grounded: only retrieved resume context is passed to the LLM, so the model cannot invent skills, projects, or credentials that aren't actually present in the resume — anything missing is explicitly reported as a gap rather than assumed.

---

## 🗺️ Roadmap / Known Limitations

- Current ATS scoring is an AI-based estimation, not an official enterprise ATS metric
- Output quality depends on the completeness of the parsed resume/JD text
- Semantically similar but isolated resume sections may occasionally be missed
- Interview questions and recommendations still benefit from human review

**Planned direction:** evolving from a single RAG pipeline into a multi-agent **LangGraph Career Assistant** with dedicated agents for resume analysis, JD analysis, skill-gap detection, recommendations, and report generation.

---

## 🤝 Contributing

Contributions are welcome and appreciated!

1. **Fork** the repository
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add: your feature description"
   ```
4. **Push to your branch**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** describing your changes and why they're valuable

Please open an issue first for major changes so we can discuss the approach.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ and a healthy respect for grounded AI outputs.</p>
