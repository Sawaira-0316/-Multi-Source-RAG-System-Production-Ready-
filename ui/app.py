# ui/app.py
import os
import sys
from pathlib import Path
from typing import List

import streamlit as st
from langchain_core.documents import Document

# -------------------------------------------------------------------
# Make sure Python can find your project modules
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from rag.pipeline import build_and_save_index, answer_question  # noqa: E402


# -------------------------------------------------------------------
# Page config + custom CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Source RAG System",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { padding-top: 1.5rem; }

    /* Hero section */
    .hero {
        display: flex;
        flex-direction: row;
        gap: 2.5rem;
        padding: 1.75rem 1.75rem;
        border-radius: 1rem;
        background: radial-gradient(circle at top left, #2b6cb0 0, #1a202c 40%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1.5rem;
    }
    .hero-left {
        flex: 2;
        color: #f9fafb;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        max-width: 40rem;
    }
    .hero-badges {
        margin-top: 0.9rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .badge {
        font-size: 0.78rem;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.25);
        background: rgba(15,23,42,0.25);
        backdrop-filter: blur(6px);
    }

    .hero-right {
        flex: 1.1;
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
        color: #e5e7eb;
    }
    .hero-card {
        padding: 0.8rem 0.9rem;
        border-radius: 0.9rem;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.35);
        font-size: 0.85rem;
    }
    .hero-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #93c5fd;
    }
    .hero-card ul {
        padding-left: 1.1rem;
        margin: 0;
    }
    .hero-card li {
        margin-bottom: 0.12rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .card {
        padding: 1rem 1.1rem;
        border-radius: 0.9rem;
        border: 1px solid rgba(148,163,184,0.35);
        background: rgba(15,23,42,0.85);
        font-size: 0.9rem;
        color: #e5e7eb;
    }

    .answer-box {
        margin-top: 0.7rem;
        padding: 0.9rem 1rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(56,189,248,0.6);
        background: rgba(8,47,73,0.7);
        color: #e5e7eb;
        font-size: 0.96rem;
    }

    /* DARK BLUE BUTTONS */
    div.stButton > button {
        background-color: #0a2540 !important;
        color: white !important;
        border: 1px solid #1e40af !important;
        border-radius: 8px !important;
        padding: 0.45rem 1.1rem !important;
        font-weight: 600 !important;
    }

    div.stButton > button:hover {
        background-color: #1e3a8a !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
    }
    
    /* DARK BLUE FILE UPLOADER BUTTON */
div[data-testid="stFileUploader"] button {
    background-color: #0a2540 !important;
    color: #ffffff !important;
    border: 1px solid #1e40af !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.1rem !important;
    font-weight: 600 !important;
}

div[data-testid="stFileUploader"] button:hover {
    background-color: #1e3a8a !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# HERO (text exactly like you want)
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-left">
        <div class="hero-title">🧠 Multi-Source RAG Assistant</div>
        <div class="hero-subtitle">
          Upload your PDFs, Word docs, spreadsheets, CSVs, slides, or paste web URLs – 
          then ask natural language questions and get context-aware answers powered by RAG.
        </div>
        <div class="hero-badges">
          <span class="badge">LangChain · FAISS · OpenAI</span>
          <span class="badge">Document Q&A</span>
          <span class="badge">Multi-format Ingestion</span>
          <span class="badge">Local + Web Sources</span>
        </div>
      </div>
      <div class="hero-right">
        <div class="hero-card">
          <h4>Tech Stack</h4>
          <ul>
            <li><b>Frontend:</b> Streamlit</li>
            <li><b>Backend:</b> Python</li>
            <li><b>AI Orchestration:</b> LangChain-style RAG</li>
            <li><b>Embeddings:</b> OpenAI &amp; FAISS</li>
          </ul>
        </div>
        <div class="hero-card">
          <h4>How it works</h4>
          <ul>
            <li>Ingest docs + URLs → text chunks</li>
            <li>Embed and store in a FAISS index</li>
            <li>Retrieve top-k chunks per question</li>
            <li>LLM generates grounded answers</li>
          </ul>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Upload + URLs section
# -------------------------------------------------------------------
st.markdown("### 📂 Add your documents & web pages")

col_left, col_right = st.columns([1.1, 1])

with col_left:
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, XLSX, CSV, TXT, MD, PPTX, HTML",
        type=[
            "pdf",
            "docx",
            "xlsx",
            "xls",
            "csv",
            "txt",
            "md",
            "pptx",
            "html",
            "htm",
        ],
        accept_multiple_files=True,
        help="Drag and drop or browse files.",
    )

with col_right:
    urls_text = st.text_area(
        "Web URLs ",
        height=120,
        placeholder="https://example.com",
    )

urls = [u.strip() for u in urls_text.splitlines() if u.strip()]

# Save uploaded files to disk
upload_dir = Path("uploaded_docs")
upload_dir.mkdir(exist_ok=True)

uploaded_paths: List[str] = []
if uploaded_files:
    for f in uploaded_files:
        out_path = upload_dir / f.name
        with open(out_path, "wb") as out_file:
            out_file.write(f.getbuffer())
        uploaded_paths.append(str(out_path))

build_col, tip_col = st.columns([0.55, 1.45])

with build_col:
    if st.button("🔨 Build / Rebuild Index", use_container_width=True):
        if not uploaded_paths and not urls:
            st.error("Please upload at least one document or provide at least one URL.")
        else:
            try:
                with st.spinner("Indexing documents... this may take a moment."):
                    count = build_and_save_index(paths=uploaded_paths, urls=urls or None)
                st.success(f"✅ Index built successfully with **{count}** documents.")
                st.session_state["index_built"] = True
            except Exception as e:
                st.error(f"❌ Error while indexing:\n\n{e}")




# -------------------------------------------------------------------
# Q&A Section
# -------------------------------------------------------------------
st.markdown("### ❓ Ask a Question About Your Data")

question = st.text_input(
    "",
    placeholder="Ask a question from your uploaded documents...",
)

if st.button("💬 Get Answer"):
    if not question.strip():
        st.error("Please enter a question first.")
    else:
        try:
            with st.spinner("Thinking with your indexed data..."):
                answer, docs = answer_question(question)

            st.markdown("#### 🧠 Answer")
            st.markdown(
                f"<div class='answer-box'>{answer}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"❌ Error while answering:\n\n{e}")
