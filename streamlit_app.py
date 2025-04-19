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
        font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        background-color: #FFFFFF;
        color: #111827;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .stApp {
        max-width: 80rem;
        margin: 0 auto;
        padding: 2rem;
    }

    h1, .stTitle {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
        color: #111827;
    }

    .stSubheader {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1F2937;
    }

    .stFileUploader > div > div > div {
        background-color: #F3F4F6;
        border: 1px dashed #D1D5DB;
        border-radius: 1rem;
        padding: 1.25rem;
        transition: background 0.3s ease;
    }

    .stFileUploader > div > button {
        background-color: #475569 !important;
        color: #FFFFFF !important;
        border-radius: 9999px;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stFileUploader > div > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 10px rgba(0,0,0,0.08);
    }

    .stAlert, .stSuccess, .stWarning, .stInfo {
        background-color: #F9FAFB !important;
        color: #1F2937 !important;
        border-radius: 1rem;
        padding: 1rem;
        border: 1px solid #E5E7EB;
    }

    .stProgress > div > div {
        background-color: #475569 !important;
        border-radius: 9999px;
    }

    .stProgress {
        height: 0.5rem !important;
    }

    p, li, span, label {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #374151;
    }

    ::-webkit-scrollbar {
        width: 6px;
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background-color: #D1D5DB;
        border-radius: 3px;
    }

    .footer {
        margin-top: 4rem;
        text-align: center;
        font-size: 0.875rem;
        color: #6B7280;
    }

    .footer a {
        color: #475569;
        font-weight: 600;
        text-decoration: none;
        transition: color 0.2s ease;
    }

    .footer a:hover {
        color: #1E293B;
    }

    .stSpinner {
        animation: pulse 1s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
""", unsafe_allow_html=True)


# Header
st.markdown('<div style="text-align:center;font-size:2rem;font-weight:700;margin-bottom:10px">GenIAlab.Space</div>', unsafe_allow_html=True)
st.title("DocVectorizer for RAG")

# Upload section
st.subheader("Upload Documents")
uploaded_files = st.file_uploader("", type=["pdf", "txt", "md", "json"], accept_multiple_files=True)

# Define vectorstore collection
use_case = "default"
collection_name = f"rag_{use_case.lower().replace(' ', '_')}"

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