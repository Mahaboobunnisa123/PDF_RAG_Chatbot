# app.py
import os
import shutil
import json
from typing import List, Dict, Any

import streamlit as st
from sentence_transformers import SentenceTransformer

from document_loader import save_uploaded_pdfs, load_pdfs_from_paths, detect_scanned
from vector_store import create_or_load_store
from rag_chain import rag_answer

# Page config
st.set_page_config(page_title="PDF RAG Assistant", layout="wide", initial_sidebar_state="expanded")

# ------------------ CUSTOM CSS ------------------
CUSTOM_CSS = """
<style>
.block-container { padding: 1rem 1.2rem; }
h1 { letter-spacing: 0.2px; }
.chat-user { background: rgba(59,130,246,0.10); padding:10px; border-radius:10px; margin-bottom:8px;}
.chat-bot { background: rgba(255,255,255,0.02); padding:10px; border-radius:10px; margin-bottom:8px;}
.source { font-size:13px; color:#cbd5e1; margin-top:6px; }
.info { background: rgba(255,255,255,0.02); padding:10px; border-radius:8px; }
.small { font-size:13px; color: #94a3b8; }
.status-badge { display:inline-block; padding:6px 10px; border-radius:999px; font-size:13px; border:1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------ SESSION STATE ------------------
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "model" not in st.session_state:
    st.session_state.model = None
if "index" not in st.session_state:
    st.session_state.index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_pdf_quality" not in st.session_state:
    st.session_state.last_pdf_quality = None
if "is_scanned_pdf" not in st.session_state:
    st.session_state.is_scanned_pdf = False

# ------------------ HEADER ------------------
st.markdown("<h1>📄 PDF RAG Chatbot </h1>", unsafe_allow_html=True)
st.markdown("<div class='small'>Upload text-based PDFs → Build index → Ask multiple questions. Toggle Advanced Mode for better precision (slower on CPU).</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K chunks to return", 3, 10, 5)
    threshold = st.slider("Similarity threshold", 0.10, 0.90, 0.20, step=0.01)
    st.info("Please set Top-K chunks=5, threshold= 0.15/0.16/0.20 for better results")
    force_rebuild = st.checkbox("Force rebuild index (re-embed PDFs)", value=False)
    advanced_mode = st.checkbox("Advanced Mode (reranker, slower on CPU)", value=False)

    st.markdown("---")
    st.subheader("Storage / Debug")
    if st.button("Clear index and cache"):
        if os.path.exists("storage"):
            shutil.rmtree("storage")
        st.session_state.index_ready = False
        st.session_state.model = None
        st.session_state.index = None
        st.session_state.metadata = None
        st.success("Cleared stored index. Rebuild to re-create.")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.success("Chat history cleared.")

    st.markdown("---")
    st.info("Note: This app supports **text-based PDFs** (selectable text/selectable text with images). Scanned/image PDFs may not give accurate responses because OCR is not enabled.")
    
# ------------------ UPLOAD / BUILD ------------------
left, right = st.columns([2, 1])
with left:
    st.subheader("Upload / Build")
    uploaded = st.file_uploader("Upload one or more PDF files (required)", type=["pdf"], accept_multiple_files=True)
    st.caption("Only text-readable/selectable PDFs[including images/only text] are supported. Scanned/image[no text selectable] PDFs may not work (OCR not enabled).")
    if st.button("Build / Load Index"):
        st.session_state.is_scanned_pdf = False
        st.session_state.last_pdf_quality = None

        if not uploaded or len(uploaded) == 0:
            st.warning("Please upload at least one PDF file to build the index.")
            st.stop()
        # Always reset previous index when user uploads new PDFs

        st.session_state.index_ready = False
        st.session_state.model = None
        st.session_state.index = None
        st.session_state.metadata = None
        if os.path.exists("storage"):
            shutil.rmtree("storage", ignore_errors=True)

        # save files and load pages
        try:
            saved_paths = save_uploaded_pdfs(uploaded)
            pages = load_pdfs_from_paths(saved_paths)
        except Exception as e:
            st.error(f"Failed to read PDFs: {e}")
            st.stop()

        if not pages:
            st.error("No readable pages were extracted from the uploaded PDFs.")
            st.stop()

        st.session_state.last_pdf_quality = detect_scanned(pages)
        st.session_state.is_scanned_pdf = bool(st.session_state.last_pdf_quality.get("likely_scanned"))
        if st.session_state.is_scanned_pdf:
            st.error("This PDF seems scanned (image-based). Please upload a text-based PDF (selectable text).")
            st.stop()
        try:
            with st.spinner("Creating/loading vector store (FAISS, embedding model may download once)..."):
                model, index, metadata = create_or_load_store(pages, force_rebuild=True)
            st.session_state.model = model
            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.index_ready = True
            st.success("Index ready — ask questions below.")
        except Exception as e:
            st.error(f"Failed to build/load index: {e}")
            st.session_state.index_ready = False

with right:
    st.subheader("PDF checks")
    if st.session_state.last_pdf_quality:
        q = st.session_state.last_pdf_quality
        if q.get("likely_scanned"):
            st.warning(q.get("message"))
            st.caption(f"Pages with tiny/empty text: {q.get('empty_pages')} / {q.get('total_pages')}")
        else:
            st.success(q.get("message"))
    else:
        st.info("No PDFs inspected yet.")

st.markdown("---")

# ------------------ CHAT ------------------
st.subheader("Chat")

# Render chat history
for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(f"<div class='chat-user'><b>You:</b><br>{m['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'><b>Bot:</b><br>{m['text']}</div>", unsafe_allow_html=True)
        if m.get("sources"):
            for s in m["sources"][:3]:
                st.markdown(f"<div class='source'>• {s.get('source')} — page {s.get('page')} — score {float(s.get('score') or 0.0):.3f}</div>", unsafe_allow_html=True)

# Chat input
query = st.chat_input("Ask a question from the PDF...")

if query:
    st.session_state.messages.append({"role": "user", "text": query})
    if not st.session_state.index_ready or st.session_state.model is None:
        st.session_state.messages.append({"role": "assistant", "text": "Index not ready. Build the index first to ask questions."})
        st.rerun()

    with st.spinner("Retrieving and answering..."):
        try:
            answer, sources = rag_answer(
                query=query,
                model=st.session_state.model,
                index=st.session_state.index,
                metadata=st.session_state.metadata,
                top_k=top_k,
                threshold=threshold,
                use_reranker=advanced_mode,
                scanned_pdf=st.session_state.is_scanned_pdf
            )
            
        except Exception as e:
            answer, sources = "Please ask a relevant question from the uploaded PDF.", []
            st.error(f"Error: {e}")

    st.session_state.messages.append({"role": "assistant", "text": answer, "sources": sources})
    st.rerun()

st.markdown("---")

# ------------------ SOURCES PANEL ------------------
st.subheader("Sources (from last answer)")
latest_sources = []
for m in reversed(st.session_state.messages):
    if m["role"] == "assistant" and m.get("sources"):
        latest_sources = m.get("sources")
        break

if not latest_sources:
    st.info("No sources available yet (ask a question).")
else:
    for i, s in enumerate(latest_sources[:6], 1):
        st.markdown(f"**{i}. {s.get('source')}** — page {s.get('page')} — score {float(s.get('score') or 0.0):.3f}")
        st.write((s.get("text") or "")[:400] + ("..." if len((s.get("text") or "")) > 400 else ""))

# ------------------ EXPORT CHAT ------------------
if st.session_state.messages:
    chat_txt = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            chat_txt.append("USER: " + m["text"] + "\n")
        else:
            chat_txt.append("BOT: " + m["text"] + "\n")
            if m.get("sources"):
                chat_txt.append("SOURCES:\n")
                for s in m.get("sources")[:4]:
                    chat_txt.append(f"- {s.get('source')} | page {s.get('page')} | score {float(s.get('score') or 0.0):.3f}\n")
            chat_txt.append("-" * 60 + "\n")
    st.download_button("Export chat (.txt)", data="".join(chat_txt).encode("utf-8"), file_name="chat_history.txt")
