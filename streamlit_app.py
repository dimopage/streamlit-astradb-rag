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

# Inject custom CSS for clean design
st.markdown("""
    <style>
    :root {
        --font-sans: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-size-base: 16px;
        --font-size-sm: 14px;
        --font-size-lg: 18px;
        --font-size-xl: 24px;
        --text-color: #1F1F1F;
        --bg-color: #FFFFFF;
        --primary-color: #000000;
        --border-color: #E0E0E0;
        --radius: 10px;
    }

    html, body, [class*="css"] {
        font-family: var(--font-sans);
        font-size: var(--font-size-base);
        background-color: var(--bg-color);
        color: var(--text-color);
        line-height: 1.6;
    }

    .stApp {
        max-width: 860px;
        margin: auto;
        padding: 2.5rem 1rem;
    }

    h1, h2, h3 {
        font-weight: 700;
        margin-bottom: 0.5em;
    }

    .stTitle {
        font-size: var(--font-size-xl);
    }

    .stSubheader {
        font-size: var(--font-size-lg);
        font-weight: 600;
        color: var(--text-color);
    }

    .stFileUploader > div > div > div {
        background-color: #F9FAFB;
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        padding: 16px;
        color: var(--text-color);
    }

    .stFileUploader > div > button {
        background-color: var(--primary-color) !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 600;
        font-size: var(--font-size-sm);
    }

    .stAlert, .stSuccess, .stWarning, .stInfo {
        background-color: #F1F5F9 !important;
        color: var(--text-color) !important;
        border-radius: var(--radius);
        font-size: var(--font-size-sm);
        padding: 12px;
    }

    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-thumb {
        background-color: #CBD5E1;
        border-radius: 6px;
    }

    .footer {
        margin-top: 4rem;
        text-align: center;
        font-size: var(--font-size-sm);
        color: #6B7280;
    }

    .footer a {
        color: var(--text-color);
        text-decoration: none;
        font-weight: 600;
    }

    .footer a:hover {
        color: #333;
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
