import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
import hashlib
import tempfile
import os
from datetime import datetime

# Load secrets
try:
    ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_NAMESPACE = st.secrets.get("ASTRA_DB_NAMESPACE", None)
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()

# Inject custom CSS with exact Tailwind styling pattern
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: ui-sans-serif, system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji" !important;
        color: #000000 !important;
        background-color: #FFFFFF;
        position: relative;
        overflow-x: hidden;
    }

    .stApp {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        position: relative;
        z-index: 1;
    }

    .stFileUploader > div > div > div {
        background-color: #f2f3f5 !important;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 12px;
        color: #000000;
    }

    .stFileUploader > div > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        font-weight: 700;
    }

    .stAlert, .stSuccess, .stWarning, .stInfo {
        background-color: #E3E6E8 !important;
        color: #000000 !important;
        border-radius: 10px;
        font-size: 1rem;
    }

    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 0.9rem;
        font-weight: 400;
        color: #000000;
    }

    .footer a {
        color: #000000;
        text-decoration: none;
        font-weight: 700;
    }

    .footer a:hover {
        color: #333333;
    }

    ::-webkit-scrollbar {
        display: none;
    }

    /* 🎨 Gradient + Blur Background Light Effects */
    .ray-container {
        position: fixed;
        inset: 0;
        pointer-events: none;
        user-select: none;
        z-index: 0;
    }

    .ray {
        position: absolute;
        border-radius: 100%;
        background: radial-gradient(rgba(83, 196, 255, 0.5) 0%, rgba(43, 166, 255, 0) 100%);
        mix-blend-mode: overlay;
        opacity: 0.6;
    }

    .ray1 {
        width: 480px;
        height: 680px;
        top: -540px;
        left: 250px;
        transform: rotate(80deg);
        filter: blur(110px);
    }

    .ray2 {
        width: 110px;
        height: 400px;
        top: -280px;
        left: 350px;
        transform: rotate(-20deg);
        filter: blur(60px);
    }

    .ray3 {
        width: 400px;
        height: 370px;
        top: -350px;
        left: 200px;
        filter: blur(21px);
    }

    .ray4 {
        width: 330px;
        height: 370px;
        top: -330px;
        left: 50px;
        filter: blur(21px);
    }

    .ray5 {
        width: 110px;
        height: 400px;
        top: -280px;
        left: -10px;
        transform: rotate(-40deg);
        filter: blur(60px);
    }
    </style>

    <div class="ray-container">
        <div class="ray ray1"></div>
        <div class="ray ray2"></div>
        <div class="ray ray3"></div>
        <div class="ray ray4"></div>
        <div class="ray ray5"></div>
    </div>
""", unsafe_allow_html=True)

# Header
st.markdown('<div style="text-align:center;font-size:2rem;font-weight:700;margin-bottom:10px">GenIAlab.Space</div>', unsafe_allow_html=True)
st.title("DocVectorizer for RAG")

# Upload section
st.subheader("Upload Documents")
uploaded_files = st.file_uploader("", type=["pdf", "txt", "md", "json"], accept_multiple_files=True)

# Define vectorstore collection
use_case = "default"
collection_name = f"{use_case.lower().replace(' ', '_')}"

if uploaded_files:
    documents = []
    skipped_files = []
    total_files = len(uploaded_files)

    progress = st.progress(0, text="Processing files...")

    for idx, file in enumerate(uploaded_files):
        file_hash = hashlib.sha256(file.getvalue()).hexdigest()

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name

            # Initialize vector store
            vectorstore = AstraDBVectorStore(
                collection_name=collection_name,
                embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
                namespace=ASTRA_DB_NAMESPACE
            )

            # Check for duplicates based on file hash
            existing = vectorstore.similarity_search(file_hash, k=1)
            if existing:
                skipped_files.append(file.name)
                continue

            # Select proper loader
            if file.type == "application/pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file.type in ["text/plain", "text/markdown"]:
                loader = TextLoader(tmp_file_path)
            elif file.type == "application/json":
                loader = UnstructuredFileLoader(tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {file.type}")
                continue

            # Load and prepare document
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "filename": file.name,
                    "upload_date": datetime.now().isoformat(),
                    "file_type": file.type,
                    "hash": file_hash
                })
            documents.extend(docs)

        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)

        progress.progress((idx + 1) / total_files, text=f"Processed {idx + 1} of {total_files} files")

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        chunks = text_splitter.split_documents(documents)

        try:
            vectorstore.add_documents(chunks)
            st.success(f"Vectorized and stored {len(chunks)} chunks in collection '{collection_name}'")
        except Exception as e:
            st.error(f"Failed to store documents: {str(e)}")

    if skipped_files:
        st.warning(f"{', '.join(skipped_files)} was already vectorized. Skipping.")
    elif not documents:
        st.info("No documents were processed.")
else:
    st.info("Please upload documents to proceed.")

# Footer
st.markdown('<div class="footer">Powered by <a href="https://genialab.space/" target="_blank">GenIAlab.Space</a></div>', unsafe_allow_html=True)