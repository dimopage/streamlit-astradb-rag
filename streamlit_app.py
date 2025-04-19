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
        background-color: #FAFAFA;
        color: #0F172A;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    .stApp {
        max-width: 82rem;
        margin: 0 auto;
        padding: 2.5rem;
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    h1, .stTitle {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.75rem;
        text-align: center;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .stSubheader {
        font-size: 1.375rem;
        font-weight: 600;
        margin-bottom: 1.25rem;
        color: #1E293B;
    }
    .stFileUploader > div > div > div {
        background-color: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 1.25rem;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div > div:hover {
        background-color: #F1F5F9;
        border-color: #94A3B8;
    }
    .stFileUploader > div > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 9999px;
        font-weight: 600;
        padding: 0.625rem 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
    }
    .stFileUploader > div > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        background-color: #1D4ED8 !important;
    }
    .stAlert, .stSuccess, .stWarning, .stInfo {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        border-radius: 1.25rem;
        padding: 1.25rem;
        border: 1px solid #D1D5DB;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    .stProgress > div > div {
        background-color: #2563EB !important;
        border-radius: 9999px;
    }
    .stProgress {
        height: 0.625rem !important;
        background-color: #E2E8F0;
        border-radius: 9999px;
    }
    p, li, span, label {
        font-size: 1rem;
        line-height: 1.75;
        color: #334155;
    }
    ::-webkit-scrollbar {
        width: 8px;
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #A1B2C3;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background-color: #64748B;
    }
    .footer {
        margin-top: 4.5rem;
        text-align: center;
        font-size: 0.875rem;
        color: #6B7280;
        padding-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
    }
    .footer a {
        color: #2563EB;
        font-weight: 600;
        text-decoration: none;
        transition: color 0.2s ease;
    }
    .footer a:hover {
        color: #1D4ED8;
    }
    .stSpinner {
        animation: pulse 1.2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
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