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
    /* Base Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        background-color: #FFFFFF;
        color: #111827;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Layout */
    .stApp {
        max-width: 76rem;
        margin: 0 auto;
        padding: 2.5rem 2rem;
    }
    
    /* Typography */
    h1, .stTitle {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
        color: #111827;
        letter-spacing: -0.025em;
    }
    
    .stSubheader {
        font-size: 1.375rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem;
        color: #1F2937;
    }
    
    p, li, span, label {
        font-size: 1rem;
        line-height: 1.7;
        color: #374151;
    }
    
    /* File Uploader */
    .stFileUploader > div > div > div {
        background-color: #F9FAFB;
        border: 2px dashed #D1D5DB;
        border-radius: 1rem;
        padding: 1.5rem;
        transition: all 0.25s ease;
    }
    
    .stFileUploader > div > div > div:hover {
        background-color: #F3F4F6;
        border-color: #9CA3AF;
    }
    
    .stFileUploader > div > button {
        background-color: #4B5563 !important;
        color: #FFFFFF !important;
        border-radius: 9999px;
        font-weight: 600;
        padding: 0.625rem 1.5rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stFileUploader > div > button:hover {
        background-color: #374151 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .stFileUploader > div > button:active {
        transform: translateY(0);
    }
    
    /* Alerts and Info Boxes */
    .stAlert, .stSuccess, .stWarning, .stInfo {
        background-color: #F9FAFB !important;
        color: #1F2937 !important;
        border-radius: 0.75rem;
        padding: 1.25rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin: 1.5rem 0;
    }
    
    /* Progress Bar */
    .stProgress {
        height: 0.5rem !important;
        margin: 1rem 0;
    }
    
    .stProgress > div > div {
        background-color: #4B5563 !important;
        border-radius: 9999px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
        background: transparent;
    }
    
    ::-webkit-scrollbar-track {
        background: #F3F4F6;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #D1D5DB;
        border-radius: 4px;
        border: 2px solid #F3F4F6;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: #9CA3AF;
    }
    
    /* Footer */
    .footer {
        margin-top: 5rem;
        padding-top: 2rem;
        text-align: center;
        font-size: 0.875rem;
        color: #6B7280;
        border-top: 1px solid #F3F4F6;
    }
    
    .footer a {
        color: #4B5563;
        font-weight: 600;
        text-decoration: none;
        transition: color 0.2s ease;
    }
    
    .footer a:hover {
        color: #1F2937;
        text-decoration: underline;
    }
    
    /* Spinner Animation */
    .stSpinner {
        animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Select Boxes and Inputs */
    .stSelectbox > div > div, .stTextInput > div > div {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:hover, .stTextInput > div > div:hover {
        border-color: #D1D5DB;
    }
    
    .stSelectbox > div > div:focus-within, .stTextInput > div > div:focus-within {
        border-color: #6B7280;
        box-shadow: 0 0 0 3px rgba(107, 114, 128, 0.1);
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