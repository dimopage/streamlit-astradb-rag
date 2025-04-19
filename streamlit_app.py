import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
import os
import tempfile
from datetime import datetime
import hashlib

# Load settings from secrets.toml
try:
    ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_NAMESPACE = st.secrets.get("ASTRA_DB_NAMESPACE", None)
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()

# Force font and style for UI
st.markdown("""
    <style>
    html, body, body * {
        font-family: ui-sans-serif, system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji" !important;
        color: #000000 !important;
    }

    .stApp {
        background-color: #FFFFFF;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }

    div[data-testid="stFileUploader"] {
        background-color: #f2f3f5 !important;
        border: 1px solid #E5E5E5 !important;
        border-radius: 10px !important;
        padding: 16px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
    }

    div[data-testid="stFileUploader"] button {
        background-color: #000 !important;
        color: #fff !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: 0.3s ease;
    }

    div[data-testid="stFileUploader"] button:hover {
        background-color: #e5e7eb !important;
        color: #000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Logo text
st.markdown('<div style="text-align:center; font-size:2rem; font-weight:700; margin:30px 0 20px;">GenIAlab.Space</div>', unsafe_allow_html=True)

# Title
st.title("DocVectorizer for RAG")

# File uploader
uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "md", "json"], accept_multiple_files=True)

# Hardcoded collection name
collection_name = "rag_default"

# Helper to compute SHA256 hash from file bytes
def compute_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

# Process files if uploaded
if uploaded_files:
    documents = []
    file_hashes = []
    progress_bar = st.progress(0, text="Checking and processing files...")

    for i, file in enumerate(uploaded_files):
        file_bytes = file.getvalue()
        file_hash = compute_file_hash(file_bytes)

        # Check if file hash already exists in AstraDB collection
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = AstraDBVectorStore(
                collection_name=collection_name,
                embedding=embeddings,
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
                namespace=ASTRA_DB_NAMESPACE
            )
            existing = vectorstore.similarity_search(query="", k=100, filter={"file_hash": file_hash})
            if existing:
                st.warning(f"File '{file.name}' was already vectorized. Skipping.")
                progress_bar.progress((i + 1) / len(uploaded_files))
                continue
        except Exception as e:
            st.error(f"Failed to check for duplicates: {str(e)}")
            continue

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        try:
            if file.type == "application/pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file.type in ["text/plain", "text/markdown"]:
                loader = TextLoader(tmp_file_path)
            elif file.type == "application/json":
                loader = UnstructuredFileLoader(tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {file.type}")
                continue

            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "filename": file.name,
                    "upload_date": datetime.now().isoformat(),
                    "file_type": file.type,
                    "file_hash": file_hash
                })
            documents.extend(docs)
            file_hashes.append(file_hash)

        except Exception as e:
            st.error(f"Error processing file '{file.name}': {str(e)}")
        finally:
            os.unlink(tmp_file_path)

        progress_bar.progress((i + 1) / len(uploaded_files))

    if documents:
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
            chunks = splitter.split_documents(documents)
            vectorstore.add_documents(chunks)
            st.success(f"Successfully vectorized and stored documents in collection '{collection_name}'")
        except Exception as e:
            st.error(f"Error storing documents in AstraDB: {str(e)}")
    else:
        st.warning("No documents were processed.")
else:
    st.info("Please upload documents to proceed.")

# Footer
st.markdown('<div style="text-align:center; margin-top:50px; font-size:0.9rem; font-weight:400;">Powered by <a href="https://genialab.space/" target="_blank">GenIAlab.Space</a></div>', unsafe_allow_html=True)
