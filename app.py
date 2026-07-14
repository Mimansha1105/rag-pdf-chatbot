import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load Environment Variables


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env")
    st.stop()

if not hf_token:
    st.error("HF_TOKEN not found in .env")
    st.stop()


# Streamlit Page Config

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

.stApp{
    background:#0f172a;
}

/* Hide Streamlit Menu */
#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
background: transparent;
}

/* Main container */

.block-container{

max-width:1100px;

padding-top:2rem;

margin:auto;

}

/* Title */

.main-title{

font-size:55px;

font-weight:700;

text-align:center;

color:white;

margin-bottom:5px;

}

/* Subtitle */

.subtitle{

text-align:center;

font-size:20px;

color:#CBD5E1;

margin-bottom:40px;

}

/* Hero Card */

.hero{

background:linear-gradient(
90deg,
#6D28D9,
#9333EA,
#EC4899
);

padding:18px;

border-radius:18px;

text-align:center;

font-size:24px;

font-weight:bold;

color:white;

margin-bottom:35px;

box-shadow:0px 8px 25px rgba(0,0,0,.35);

}

/* Cards */

[data-testid="stMetric"]{

background:#1E293B;

padding:15px;

border-radius:15px;

border:1px solid #334155;

}

/* Chat */

.stChatMessage{

background:#1E293B;

padding:20px;

border-radius:18px;

border:1px solid #334155;

margin-bottom:20px;

color:white;

}

/* All markdown text */

.stMarkdown,
.stMarkdown p,
.stMarkdown li{

color:white !important;

}

/* Chat Input */

.stChatInput textarea{

background:#1E293B !important;

color:white !important;

border-radius:15px;

}

/* Sidebar */


/* Upload */

.stFileUploader{

border:2px dashed #7C3AED;

border-radius:20px;

padding:35px;

background:#111827;

}

/* Buttons */

.stButton button{

background:linear-gradient(90deg,#7C3AED,#EC4899);

color:white;

border:none;

border-radius:12px;

font-weight:bold;

}

/* Success */

.stSuccess{

background:#052e16;

color:white;

}

</style>
""",unsafe_allow_html=True)
st.markdown(
"""
<div class="main-title">

🤖 PDF AI Assistant

</div>

<div class="subtitle">

Upload PDFs • Ask Questions • Get Instant Answers

</div>

<div class="hero">

⚡ Powered by Groq + LangChain + Chroma + HuggingFace

</div>

""",
unsafe_allow_html=True
)
# Session State


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Initialize LLM


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
)


# Initialize Embeddings




# Sidebar
# ================= Upload Section =================

left, center, right = st.columns([1,2,1])

with center:

    st.markdown("## 📂 Upload your PDFs")

    st.markdown(
    """
    Upload one or more PDFs and start chatting with them.

    **Supported**

    • 📄 Research Papers

    • 📚 Books

    • 📝 Notes

    • 📘 Documentation
    """
    )

    uploaded_files = st.file_uploader(
        "Drag & Drop your PDF(s) here",
        type=["pdf"],
        accept_multiple_files=True
    )
    if uploaded_files:

        st.markdown("### 📑 Uploaded Files")

        for file in uploaded_files:
           st.success(f"✅ {file.name}")
# Process PDFs


if uploaded_files and st.session_state.vectorstore is None:

    with st.spinner("🧠 AI is reading and understanding your documents..."):
        embeddings = HuggingFaceEmbeddings(
         model_name="sentence-transformers/all-MiniLM-L6-v2",
         model_kwargs={"device": "cpu"},
         encode_kwargs={"normalize_embeddings": True},
       )

        documents = []

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

                tmp.write(uploaded_file.read())

                loader = PyPDFLoader(tmp.name)

                docs = loader.load()

                documents.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = vectorstore.as_retriever()

        st.balloons()

        st.success("🎉 PDFs processed successfully!")

        st.divider()

        st.markdown("## 📊 Document Analytics")

        c1, c2, c3 = st.columns(3)

        pdf_count = len(uploaded_files) if uploaded_files else 0
        page_count = len(documents) if uploaded_files else 0
        chunk_count = len(splits) if uploaded_files else 0

        c1.metric("📄 PDFs", pdf_count)
        c2.metric("📑 Pages", page_count)
        c3.metric("🧩 Chunks", chunk_count)



# Prompt


prompt = ChatPromptTemplate.from_template(
"""
You are a helpful AI assistant.

Answer ONLY using the context provided below.

If the answer is not found in the context, simply reply:

"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}
"""
)

output_parser = StrOutputParser()

# Chat Interface


st.divider()

question = st.chat_input("Ask anything about your PDFs...")
if question:

    if st.session_state.retriever is None:

        st.warning("Please upload a PDF first.")

        st.stop()

    

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )
    docs = st.session_state.retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    chain = (
        prompt
        | llm
        | output_parser
    )
    answer = chain.invoke(
        {
            "context":context,
            "question":question
        }
    )
    

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )
    st.divider()

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])




