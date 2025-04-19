import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
import hashlib
import tempfile
import os
from datetime import datetime

# Custom CSS that matches the design system
st.markdown("""
    <style>
        /* Base Styles */
        :root {
            --primary: #000000;
            --secondary: #6B7280;
            --background: #FFFFFF;
            --gray-50: #F9FAFB;
            --gray-100: #F3F4F6;
            --gray-200: #E5E7EB;
            --gray-600: #4B5563;
            --gray-900: #111827;
        }

        /* Typography */
        .stApp, .stMarkdown, [data-testid="stMarkdownContainer"] {
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
            color: var(--gray-900);
        }

        /* Layout */
        .stApp {
            max-width: 80rem;
            margin: 0 auto;
            padding: 6rem 1.5rem;
            background: var(--background);
        }

        /* Headers */
        h1 {
            font-size: 3rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em;
            line-height: 1.2;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .stSubheader {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin: 2rem 0 1rem;
            color: var(--gray-900);
        }

        /* File Uploader */
        .stFileUploader {
            margin: 2rem 0;
        }

        .stFileUploader > div {
            background: var(--gray-50);
            border: 2px dashed var(--gray-200);
            border-radius: 1rem;
            padding: 2rem;
            text-align: center;
        }

        .stFileUploader [data-testid="stFileUploadDropzone"] {
            background: transparent !important;
            border: none !important;
        }

        /* Buttons */
        .stButton > button {
            background: var(--primary) !important;
            color: var(--background) !important;
            border-radius: 9999px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.2s;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        /* Progress Bar */
        .stProgress > div > div {
            background: var(--primary) !important;
            border-radius: 9999px;
        }

        /* Messages */
        .stAlert, .stInfo, .stSuccess, .stWarning {
            background: var(--gray-50) !important;
            border: 1px solid var(--gray-200);
            border-radius: 1rem;
            padding: 1rem !important;
            color: var(--gray-900) !important;
        }

        /* Footer */
        .footer {
            margin-top: 6rem;
            text-align: center;
            color: var(--secondary);
            font-size: 0.875rem;
        }

        .footer a {
            color: var(--primary);
            text-decoration: none;
            font-weight: 600;
            transition: color 0.2s;
        }

        .footer a:hover {
            color: var(--gray-600);
        }

        /* Hide Streamlit Branding */
        #MainMenu, footer, header {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
try:
    ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_NAMESPACE = st.secrets.get("ASTRA_DB_NAMESPACE", None)
except KeyError as e:
    st.error(f"Missing required configuration: {e}")
    st.stop()

# Page Header
st.markdown('<div style="text-align:center;font-size:1.25rem;font-weight:600;color:#6B7280;margin-bottom:0.5rem">GenIALab.Space</div>', unsafe_allow_html=True)
st.title("Document Vectorizer")
st.markdown('<p style="text-align:center;color:#6B7280;margin-bottom:3rem">Transform your documents into vector embeddings for RAG applications</p>', unsafe_allow_html=True)

# Document Upload Section
st.subheader("Upload Documents")
uploaded_files = st.file_uploader(
    "Supported formats: PDF, TXT, MD, JSON",
    type=["pdf", "txt", "md", "json"],
    accept_multiple_files=True
)

# Processing Logic
use_case = "default"
collection_name = f"rag_{use_case.lower().replace(' ', '_')}"

if uploaded_files:
    documents = []
    skipped_files = []
    total_files = len(uploaded_files)

    progress = st.progress(0, text="Processing documents...")

    for idx, file in enumerate(uploaded_files, 1):
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

            # Check for duplicates
            existing = vectorstore.similarity_search(file_hash, k=1)
            if existing:
                skipped_files.append(file.name)
                continue

            # Load document based on type
            loader = {
                "application/pdf": PyPDFLoader,
                "text/plain": TextLoader,
                "text/markdown": TextLoader,
                "application/json": UnstructuredFileLoader
            }.get(file.type)

            if not loader:
                st.warning(f"Unsupported file type: {file.type}")
                continue

            # Process document
            docs = loader(tmp_file_path).load()
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

        progress.progress(idx / total_files, text=f"Processed {idx} of {total_files} files")

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=250
        )
        chunks = text_splitter.split_documents(documents)

        try:
            vectorstore.add_documents(chunks)
            st.success(f"Successfully vectorized {len(chunks)} chunks from {len(documents)} documents")
        except Exception as e:
            st.error(f"Failed to store vectors: {str(e)}")

    if skipped_files:
        st.warning(f"Skipped previously processed files: {', '.join(skipped_files)}")
    elif not documents:
        st.info("No new documents to process")
else:
    st.info("Upload your documents to begin processing")

# Footer
st.markdown("""
    <div class="footer">
        Powered by <a href="https://genialab.space" target="_blank">GenIALab.Space</a>
    </div>
""", unsafe_allow_html=True)
