import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
import os
import tempfile
from datetime import datetime
import hashlib

# Load settings from secrets.toml with error handling
try:
    ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_NAMESPACE = st.secrets.get("ASTRA_DB_NAMESPACE", None)
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()

# Custom CSS
st.markdown("""
    <style>
    /* Force font stack globally */
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

    .logo-text {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #000000;
        margin: 30px 0 20px 0;
    }

    h1 {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 40px;
    }

    /* Uploader box styling */
    section[data-testid="stFileUploader"] > div {
        border: 1px solid #E5E5E5 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        background-color: #f2f3f5 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
    }

    section[data-testid="stFileUploader"] button {
        background-color: #f2f3f5 !important;
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s;
    }

    section[data-testid="stFileUploader"] button:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
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
    </style>
""", unsafe_allow_html=True)

# Logo text
st.markdown('<div class="logo-text">GenIAlab.Space</div>', unsafe_allow_html=True)
st.title("DocVectorizer for RAG")

# Use case (hardcoded)
use_case = "default"
collection_name = f"rag_{use_case.lower().replace(' ', '_')}"

# File uploader
uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "md", "json"], accept_multiple_files=True)

# Show progress area
progress_placeholder = st.empty()

# Track duplicate files by hash
def get_file_hash(file_data):
    return hashlib.sha256(file_data).hexdigest()

processed_hashes = set()

if uploaded_files:
    documents = []
    total_files = len(uploaded_files)
    current_file_index = 0

    for file in uploaded_files:
        current_file_index += 1
        progress_percent = int((current_file_index - 1) / total_files * 100)
        progress_placeholder.progress(progress_percent, text=f"Processing file {current_file_index} of {total_files}")

        file_bytes = file.getvalue()
        file_hash = get_file_hash(file_bytes)

        if file_hash in processed_hashes:
            st.warning(f"Duplicate file skipped: {file.name}")
            continue

        processed_hashes.add(file_hash)

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        # Load document based on file type
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
        except Exception as e:
            st.error(f"Error loading file {file.name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)

    if documents:
        try:
            progress_placeholder.progress(60, text="Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
            chunks = text_splitter.split_documents(documents)

            progress_placeholder.progress(70, text="Initializing embeddings...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            progress_placeholder.progress(80, text="Connecting to vector store...")
            vectorstore = AstraDBVectorStore(
                collection_name=collection_name,
                embedding=embeddings,
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
                namespace=ASTRA_DB_NAMESPACE
            )

            progress_placeholder.progress(90, text="Adding documents to vector store...")
            vectorstore.add_documents(chunks)

            progress_placeholder.progress(100, text="Completed successfully")
            st.success(f"Documents successfully vectorized and stored in collection {collection_name}")

        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Error during vectorization: {str(e)}")
    else:
        progress_placeholder.empty()
        st.warning("No documents were processed")
else:
    st.info("Please upload documents to proceed")

# Footer
st.markdown(
    '<div class="footer">Powered by <a href="https://genialab.space/" target="_blank">GenIAlab.Space</a></div>',
    unsafe_allow_html=True
)
