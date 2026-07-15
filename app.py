import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================== Environment ==============================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env")
    st.stop()

if not hf_token:
    st.error("HF_TOKEN not found in .env")
    st.stop()

# ============================== Page Config ==============================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp{ background:#0f172a; }
#MainMenu{ visibility:hidden; }
footer{ visibility:hidden; }
header{ background: transparent; }

.block-container{
    max-width:1100px;
    padding-top:2rem;
    margin:auto;
}

.main-title{
    font-size:48px;
    font-weight:700;
    text-align:center;
    color:white;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#CBD5E1;
    margin-bottom:30px;
}

.hero{
    background:linear-gradient(90deg,#6D28D9,#9333EA,#EC4899);
    padding:16px;
    border-radius:16px;
    text-align:center;
    font-size:20px;
    font-weight:bold;
    color:white;
    margin-bottom:30px;
    box-shadow:0px 8px 25px rgba(0,0,0,.35);
}

[data-testid="stMetric"]{
    background:#1E293B;
    padding:15px;
    border-radius:15px;
    border:1px solid #334155;
}

.stChatMessage{
    background:#1E293B;
    padding:18px;
    border-radius:16px;
    border:1px solid #334155;
    margin-bottom:16px;
    color:white;
}

.stMarkdown, .stMarkdown p, .stMarkdown li{
    color:white !important;
}

.stChatInput textarea{
    background:#1E293B !important;
    color:white !important;
    border-radius:14px;
}

section[data-testid="stSidebar"]{
    background:#111827;
    border-right:1px solid #334155;
}

/* Force high-contrast text everywhere in the sidebar */
section[data-testid="stSidebar"] *{
    color:#F1F5F9 !important;
}
section[data-testid="stSidebar"] h1{
    font-size:26px !important;
    font-weight:800 !important;
    color:#FFFFFF !important;
}
section[data-testid="stSidebar"] h3{
    color:#FFFFFF !important;
}
section[data-testid="stSidebar"] li{
    color:#E2E8F0 !important;
}

.stFileUploader{
    border:2px dashed #7C3AED;
    border-radius:16px;
    padding:20px;
    background:#1E293B;
}

/* "Drag and drop file here" + size-limit hint text */
[data-testid="stFileUploaderDropzoneInstructions"] div{
    color:#F1F5F9 !important;
    font-weight:600;
}
[data-testid="stFileUploaderDropzoneInstructions"] span{
    color:#94A3B8 !important;
    font-weight:400;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg{
    fill:#A78BFA !important;
}
[data-testid="stFileUploader"] label{
    color:#F1F5F9 !important;
    font-weight:600;
    font-size:15px;
}
[data-testid="stFileUploaderFile"]{
    background:#0f172a;
    border-radius:10px;
}
.stFileUploader button{
    background:#334155 !important;
    color:#F1F5F9 !important;
    border-radius:8px !important;
}

.upload-hint{
    font-size:13px;
    color:#94A3B8 !important;
    margin:6px 0 14px 0;
}

.stButton button{
    background:linear-gradient(90deg,#7C3AED,#EC4899);
    color:white !important;
    border:none;
    border-radius:12px;
    font-weight:bold;
    width:100%;
}

.stSuccess{ background:#052e16; color:white; }

.source-box{
    background:#111827;
    border:1px solid #334155;
    border-radius:10px;
    padding:10px 14px;
    margin-top:6px;
    font-size:13px;
    color:#CBD5E1;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-title">🤖 PDF AI Assistant</div>
<div class="subtitle">Upload PDFs • Ask Questions • Get Instant, Grounded Answers</div>
<div class="hero">⚡ Powered by Groq + LangChain + Chroma + HuggingFace</div>
""",
    unsafe_allow_html=True,
)

# ============================== Session State ==============================

defaults = {
    "vectorstore": None,
    "retriever": None,
    "messages": [],
    "processed_files": None,     # names+sizes of the last processed batch
    "collection_name": None,     # unique per upload batch, avoids stale-data bleed
    "doc_stats": None,           # (pdf_count, page_count, chunk_count)
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================== LLM ==============================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0,  # deterministic, sticks to context instead of embellishing
)

# ============================== Sidebar: Upload ==============================

with st.sidebar:
    st.title("📂 Upload your PDFs")
    st.markdown(
        """
Upload one or more PDFs and start chatting.

**Supported**
- 📄 Research Papers
- 📚 Books
- 📝 Notes
- 📘 Documentation
"""
    )

    st.markdown(
        '<div class="upload-hint">Upload 1 PDF, or up to 5 at once — '
        "all of them are read together. Remove a file with the "
        "✕ and drop in a different one any time, no refresh needed.</div>",
        unsafe_allow_html=True,
    )

    MAX_FILES = 5
    raw_uploaded_files = st.file_uploader(
        "Choose PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    uploaded_files = raw_uploaded_files
    if raw_uploaded_files and len(raw_uploaded_files) > MAX_FILES:
        st.warning(
            f"You added {len(raw_uploaded_files)} files — only the first "
            f"{MAX_FILES} will be read. Remove a few to include others."
        )
        uploaded_files = raw_uploaded_files[:MAX_FILES]

    if uploaded_files:
        st.markdown("### 📑 Uploaded Files")
        for file in uploaded_files:
            st.success(f"✅ {file.name}")

    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

# ============================== Process PDFs ==============================

def fingerprint(files):
    """Identify a batch of uploads by name+size so we know when it changes."""
    return tuple(sorted((f.name, f.size) for f in files))

if uploaded_files:
    current_fingerprint = fingerprint(uploaded_files)

    # Reprocess if this is a new/changed batch of files (fixes the bug where
    # uploading more PDFs later was silently ignored).
    if current_fingerprint != st.session_state.processed_files:
        with st.spinner("🧠 AI is reading and understanding your documents..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            documents = []
            tmp_paths = []

            try:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_paths.append(tmp.name)

                    loader = PyPDFLoader(tmp_paths[-1])
                    documents.extend(loader.load())
            finally:
                for path in tmp_paths:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            splits = splitter.split_documents(documents)

            # Fresh, uniquely-named in-memory collection per batch.
            # No persist_directory -> nothing lingers on disk between
            # sessions/uploads, so old PDFs can never bleed into new answers.
            collection_name = f"session_{uuid.uuid4().hex[:8]}"

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name=collection_name,
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = vectorstore.as_retriever(
                search_kwargs={"k": 6}
            )
            st.session_state.processed_files = current_fingerprint
            st.session_state.collection_name = collection_name
            st.session_state.doc_stats = (
                len(uploaded_files),
                len(documents),
                len(splits),
            )
            st.session_state.messages = []  # new documents -> fresh conversation

        st.balloons()
        st.success("🎉 PDFs processed successfully!")

    if st.session_state.doc_stats:
        st.divider()
        st.markdown("## 📊 Document Analytics")
        c1, c2, c3 = st.columns(3)
        pdf_count, page_count, chunk_count = st.session_state.doc_stats
        c1.metric("📄 PDFs", pdf_count)
        c2.metric("📑 Pages", page_count)
        c3.metric("🧩 Chunks", chunk_count)

elif st.session_state.processed_files is not None:
    # All files were removed (the "✕" / undo) — drop the old index instead
    # of silently answering from documents that are no longer uploaded.
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.processed_files = None
    st.session_state.collection_name = None
    st.session_state.doc_stats = None
    st.session_state.messages = []

# ============================== Prompt ==============================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer ONLY using the context provided below. Do not use outside knowledge
and do not guess. If the answer is not found in the context, reply exactly:

"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}
"""
)

output_parser = StrOutputParser()

# ============================== Chat Interface ==============================

st.divider()

if st.session_state.retriever is None:
    st.markdown(
        '<div class="upload-hint" style="text-align:center;font-size:15px;">'
        "👈 Upload a PDF (or a few) from the sidebar to start chatting."
        "</div>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources used for this answer"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-box">📄 {src}</div>',
                        unsafe_allow_html=True,
                    )

question = st.chat_input("Ask anything about your PDFs...")

if question:
    if st.session_state.retriever is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        docs = st.session_state.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        sources = []
        for doc in docs:
            name = os.path.basename(doc.metadata.get("source", "document"))
            page = doc.metadata.get("page")
            label = f"{name} (page {page + 1})" if page is not None else name
            if label not in sources:
                sources.append(label)

        chain = prompt | llm | output_parser

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke({"context": context, "question": question})
            st.write(answer)
            if sources:
                with st.expander("📎 Sources used for this answer"):
                    for src in sources:
                        st.markdown(
                            f'<div class="source-box">📄 {src}</div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
