# app.py
"""
Document Q&A System - Refactored single-file Streamlit app
Features:
 - Upload & process multiple PDFs
 - Fast cached SentenceTransformer embeddings
 - FAISS vector store saved/loaded locally
 - Document preview (per-page), per-chunk source metadata
 - Ask questions (retrieval + LLM chain) and show source snippets
 - Summarize document + download summary
 - Clear / delete index & session management
 - Improved UI, status badges, and safer error handling
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# langchain pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Embeddings + vectorstore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# Google generative adapter (used if you want Gemini; optional)
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration & constants ---
load_dotenv()

st.set_page_config(page_title="Document Q&A System", layout="wide", initial_sidebar_state="expanded")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_DIR = Path("faiss_index")
CHAT_HISTORY_FILE = "chat_history.json"
MODEL_ORDER = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Helper utilities ---


def load_google_api_key():
    key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key


def configure_google_genai(api_key: str):
    """Configure google generative ai (optional). Returns True if success else False."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception:
        return False


def safe_write_json(path: Path, data):
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"Could not save file {path}: {e}")


def safe_read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# --- Streamlit caching & resources ---

@st.cache_resource
def get_embedding_model():
    """Return SentenceTransformerEmbeddings object (cached)."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource
def get_faiss_if_exists():
    """Return loaded FAISS vectorstore if FAISS_DIR exists, else None."""
    if FAISS_DIR.exists():
        try:
            embeddings = get_embedding_model()
            return FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None


# --- Document processing functions ---


def extract_text_from_pdf(file) -> Dict[int, str]:
    """
    Read a PdfReader file-like object and return a dict of {page_num: text}.
    file is a streamlit UploadedFile (has .read() and .name).
    """
    page_texts = {}
    try:
        reader = PdfReader(file)
        for i, p in enumerate(reader.pages):
            t = p.extract_text()
            page_texts[i + 1] = t or ""
    except Exception as e:
        st.warning(f"Failed to read {getattr(file, 'name', 'uploaded file')}: {e}")
    return page_texts


def build_chunks_from_pages(file_name: str, page_texts: Dict[int, str]) -> List[Dict]:
    """
    Split pages into chunks and attach metadata for source-tracking.
    Returns list of dicts: {"text": ..., "metadata": {...}}
    """
    text_concat = []
    metadata_map = []

    for page_no, text in page_texts.items():
        if text.strip():
            # prefix small header for each page to keep trace
            section_text = f"--- Source: {file_name}, Page: {page_no} ---\n{text}"
            text_concat.append(section_text)
            metadata_map.append({"source": file_name, "page": page_no})

    all_text = "\n\n".join(text_concat)
    if not all_text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(all_text)

    # Create per-chunk metadata by attempting to preserve page info (simple heuristic):
    chunk_objs = []
    for idx, ch in enumerate(chunks):
        # Try to find the closest page header in the chunk (if present)
        source = None
        page_no = None
        if "--- Source:" in ch:
            # parse first header occurrence
            try:
                header_line = [ln for ln in ch.splitlines() if ln.startswith("--- Source:")][0]
                # header format: --- Source: filename, Page: N ---
                header_line = header_line.replace("---", "").strip()
                parts = header_line.split(",")
                source = parts[0].replace("Source:", "").strip()
                page_part = parts[1].replace("Page:", "").strip()
                page_no = int(page_part)
            except Exception:
                source = file_name
        else:
            source = file_name

        md = {"source": source or file_name, "page": page_no or 0, "chunk_id": idx}
        chunk_objs.append({"text": ch, "metadata": md})
    return chunk_objs


def create_faiss_from_chunks(chunk_objs: List[Dict]):
    """
    Create FAISS vectorstore from chunk objects (list of {"text","metadata"}).
    Save to FAISS_DIR.
    """
    if not chunk_objs:
        return None
    texts = [c["text"] for c in chunk_objs]
    metadatas = [c["metadata"] for c in chunk_objs]

    embeddings = get_embedding_model()
    vs = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    # ensure directory
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(FAISS_DIR))
    return vs


# --- QA chain ---


def get_conversational_chain(prompt_template: str, temperature: float = 0.2):
    """
    Create a QA chain using the first available model in MODEL_ORDER.
    Returns chain or None.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Try Gemini via ChatGoogleGenerativeAI (if configured). If not available, raise to let caller handle.
    for model_name in MODEL_ORDER:
        try:
            model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            st.session_state["last_model_used"] = model_name
            return chain
        except Exception:
            continue
    return None


# --- Chat history persistence ---


def save_chat_history():
    try:
        safe_write_json(Path(CHAT_HISTORY_FILE), st.session_state.get("messages", []))
    except Exception:
        st.warning("Could not save chat history.")


def load_chat_history():
    data = safe_read_json(Path(CHAT_HISTORY_FILE))
    if isinstance(data, list):
        return data
    return []


# --- Streamlit UI & App logic ---


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()
    if "faiss_ready" not in st.session_state:
        st.session_state.faiss_ready = get_faiss_if_exists() is not None
    if "source_toggle" not in st.session_state:
        st.session_state.source_toggle = {}
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []  # list of filenames processed
    if "last_model_used" not in st.session_state:
        st.session_state.last_model_used = None


def sidebar_controls():
    st.sidebar.title("📄 Document Q&A")
    st.sidebar.markdown("A fast, simple assistant for your PDFs.")

    # Status badge
    status = "Ready" if st.session_state.faiss_ready else "No Documents"
    status_class = "✅" if st.session_state.faiss_ready else "❗"
    st.sidebar.markdown(f"**Status:** {status_class} {status}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("1) Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"], key="uploader")

    if uploaded_files:
        st.sidebar.write(f"{len(uploaded_files)} file(s) selected")
        if st.sidebar.button("Process uploaded files", use_container_width=True):
            process_uploaded_files(uploaded_files)

    st.sidebar.markdown("---")
    st.sidebar.subheader("2) Document Tools")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("Summarize Documents", disabled=not st.session_state.faiss_ready, use_container_width=True):
            summarize_documents()
    with col2:
        if st.sidebar.button("Delete Index & Session", use_container_width=True):
            delete_index_and_session()

    st.sidebar.markdown("---")
    st.sidebar.subheader("3) Session")
    if st.sidebar.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.source_toggle = {}
        save_chat_history()
        st.experimental_rerun()

    if st.session_state.messages:
        chat_txt = format_chat_history(st.session_state.messages)
        st.sidebar.download_button("Download Chat", chat_txt, file_name=f"chat_{int(time.time())}.txt")

    st.sidebar.markdown("---")
    # API key check (optional)
    api_key = load_google_api_key()
    if not api_key:
        st.sidebar.error("Google API key not set. Put in Streamlit secrets or .env to use Gemini.")
    else:
        if configure_google_genai(api_key):
            st.sidebar.success("Google GenAI configured (Gemini models available if your key has access).")
        else:
            st.sidebar.warning("Google key present but GenAI config failed (check key).")


def format_chat_history(messages):
    s = "Chat History\n" + "=" * 20 + "\n\n"
    for m in messages:
        role = "User" if m.get("role") == "user" else "Assistant"
        s += f"[{role}]\n{m.get('content', '')}\n"
        if m.get("sources"):
            s += "Sources:\n"
            for src in m["sources"]:
                s += "- " + src + "\n"
        s += "\n" + ("-" * 30) + "\n\n"
    return s


def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files into FAISS index (overwrites previous index)."""
    st.session_state.messages.append({"role": "assistant", "content": "Document processing started..."})
    save_chat_history()

    all_chunk_objs = []
    processed_files = []

    with st.spinner("Extracting text & chunking..."):
        for f in uploaded_files:
            page_texts = extract_text_from_pdf(f)
            if not page_texts:
                st.warning(f"No readable text extracted from {f.name}. It might be scanned images or unsupported PDF.")
                continue
            chunk_objs = build_chunks_from_pages(f.name, page_texts)
            if chunk_objs:
                all_chunk_objs.extend(chunk_objs)
                processed_files.append(f.name)

    if not all_chunk_objs:
        st.session_state.messages.append({"role": "assistant", "content": "No valid text found in uploaded PDFs. Index not created."})
        save_chat_history()
        st.experimental_rerun()

    with st.spinner("Creating vector index (FAISS)..."):
        try:
            vs = create_faiss_from_chunks(all_chunk_objs)
            if vs:
                st.session_state.faiss_ready = True
                st.session_state.uploaded_docs = processed_files
                st.success("Documents processed and indexed successfully.")
                st.session_state.messages.append({"role": "assistant", "content": f"Processed documents: {', '.join(processed_files)}"})
            else:
                st.error("Failed to create vector store.")
        except Exception as e:
            st.error(f"Error while creating vector store: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error creating index: {e}"})

    save_chat_history()
    st.experimental_rerun()


def delete_index_and_session():
    """Delete FAISS directory and clear session data."""
    try:
        if FAISS_DIR.exists():
            for p in FAISS_DIR.glob("*"):
                p.unlink()
            FAISS_DIR.rmdir()
    except Exception:
        # fallback: attempt removing tree carefully
        import shutil
        shutil.rmtree(str(FAISS_DIR), ignore_errors=True)

    st.session_state.faiss_ready = False
    st.session_state.uploaded_docs = []
    st.session_state.messages = []
    st.session_state.source_toggle = {}
    save_chat_history()
    st.success("Index and session cleared.")
    st.experimental_rerun()


def summarize_documents():
    """Create a short summary of the whole index by doing a similarity search and asking the model to summarize."""
    if not st.session_state.faiss_ready:
        st.warning("No documents indexed.")
        return

    embeddings = get_embedding_model()
    vs = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)

    # fetch top-k docs (k = up to 10 or number of docs)
    try:
        docs = vs.similarity_search("Summarize the document", k=min(10, len(vs.index_to_docstore_id)))
    except Exception:
        docs = []

    if not docs:
        st.warning("Could not retrieve document chunks for summary.")
        return

    summary_prompt = """
    Based on the following context, provide a concise summary (bullet points, 6-10 items max).
    Focus on key topics, findings, and any concrete facts found in the text.

    Context:
    {context}

    Question:
    {question}

    Summary:
    """

    chain = get_conversational_chain(summary_prompt)
    if not chain:
        st.error("Could not initialize model chain for summarization. Check API / model access.")
        return

    with st.spinner("Generating summary..."):
        try:
            resp = chain({"input_documents": docs, "question": "Summarize the document."}, return_only_outputs=True)
            summary = resp.get("output_text") or "No summary produced."
            # store in messages
            st.session_state.messages.append({"role": "assistant", "content": summary, "sources": [d.page_content[:150] + "..." for d in docs]})
            save_chat_history()
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Summary generation failed: {e}")


def main():
    st.write(
        """
        <div style="text-align:center;">
            <h1 style="color: #fff; font-weight:700;">📄 Document Q&A System</h1>
            <p style="color: #bcd; margin-top:-12px;">Upload PDFs, index them and ask questions. Source-aware answers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()
    sidebar_controls()

    # show uploaded docs summary
    st.markdown("---")
    st.subheader("Indexed Documents")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        if st.session_state.uploaded_docs:
            st.info(", ".join(st.session_state.uploaded_docs))
        else:
            st.write("No documents indexed yet. Upload PDFs in the sidebar to get started.")
    with col_b:
        if st.session_state.faiss_ready:
            st.success("Index ready")
        else:
            st.warning("Index not available")

    # Chat interface
    st.markdown("---")
    st.subheader("Chat with your documents")
    chat_box, sidebar_box = st.columns([3, 1])

    with chat_box:
        # display messages
        for idx, msg in enumerate(st.session_state.messages):
            role = msg.get("role", "assistant")
            with st.chat_message(role):
                st.markdown(msg.get("content", ""))

            # add view source button for assistant messages with sources
            if role == "assistant" and msg.get("sources"):
                key = f"src_{idx}"
                if st.button("📄 View Source", key=key):
                    st.session_state.source_toggle[idx] = not st.session_state.source_toggle.get(idx, False)

                if st.session_state.source_toggle.get(idx):
                    # show sources as info boxes
                    for s in msg.get("sources", [])[:5]:
                        st.info(s)

        # input
        prompt_placeholder = "Process documents first..." if not st.session_state.faiss_ready else "Ask a question..."
        user_input = st.chat_input(prompt_placeholder, disabled=not st.session_state.faiss_ready)

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            save_chat_history()
            st.experimental_rerun()

        # generate assistant response if last msg is from user
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            last_user_prompt = st.session_state.messages[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        embeddings = get_embedding_model()
                        vs = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
                        docs = vs.similarity_search(last_user_prompt, k=4)

                        qa_prompt = """
                        Answer the question using ONLY the provided context. If the answer is not in the context, reply:
                        "The answer is not available in the context." Be concise and show references.
                        Context:
                        {context}

                        Question:
                        {question}

                        Answer:
                        """

                        chain = get_conversational_chain(qa_prompt)
                        if not chain:
                            answer = "Could not initialize the LLM chain. Check model/API access."
                        else:
                            resp = chain({"input_documents": docs, "question": last_user_prompt}, return_only_outputs=True)
                            answer = resp.get("output_text", "No answer generated.")

                        # sources: create friendly source strings
                        sources = []
                        for d in docs:
                            md = getattr(d, "metadata", {}) or {}
                            src = md.get("source") or md.get("file") or "unknown"
                            page = md.get("page", 0)
                            sources.append(f"{src} (page {page})")

                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                        save_chat_history()
                        st.experimental_rerun()

                    except Exception as e:
                        err = f"An internal error occurred while answering: {e}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})
                        save_chat_history()
                        st.experimental_rerun()

    with sidebar_box:
        st.markdown("### Quick tools")
        if st.button("Show index info"):
            if st.session_state.faiss_ready:
                embeddings = get_embedding_model()
                vs = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
                n_docs = len(vs.index_to_docstore_id)
                st.info(f"Indexed chunks: {n_docs}")
            else:
                st.warning("No index available.")

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            """
            - Simple Document Q&A using embeddings + a generative model.
            - Keeps message history locally in chat_history.json.
            - For best results, upload searchable PDFs (not scanned images).
            """
        )

    # end main


if __name__ == "__main__":
    main()
