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
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    background-color: #F8FAFC;
    color: #1E293B;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Layout */
.stApp {
    max-width: 80rem;
    margin: 0 auto;
    padding: 3rem 1.5rem;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Typography */
h1, .stTitle {
    font-size: 2.25rem;
    font-weight: 800;
    margin-bottom: 2.5rem;
    text-align: center;
    color: #0F172A;
    letter-spacing: -0.02em;
}

.stSubheader {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 2rem 0 1.25rem;
    color: #1E293B;
}

p, li, span, label {
    font-size: 1.1rem;
    line-height: 1.8;
    color: #475569;
}

/* File Uploader */
.stFileUploader > div > div > div {
    background-color: #f2f2f2 !ّImportant;
    border: 2px dashed #CBD5E1;
    border-radius: 12px;
    padding: 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(242, 242, 242,0.05);
}

.stFileUploader > div > div > div:hover {
    background-color: #F8FAFC;
    border-color: #94A3B8;
    box-shadow: 0 4px 8px rgba(242, 242, 242,0.1);
}

.stFileUploader > div > button {
    background-color: #f2f2f2 !important;
    color: #FFFFFF !important;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.75rem 1.75rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    border: none !important;
}

.stFileUploader > div > button:hover {
    background-color: #1F2937 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(242, 242, 242,0.2);
}

.stFileUploader > div > button:active {
    transform: translateY(0);
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

/* Alerts and Info Boxes */
.stAlert, .stSuccess, .stWarning, .stInfo {
    background-color: #f2f2f2 !important;
    color: #1E293B !important;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 12px rgba(242, 242, 242,0.05);
    margin: 1.75rem 0;
    font-size: 1.1rem;
}

/* Progress Bar */
.stProgress {
    height: 0.75rem !important;
    margin: 1.5rem 0;
    background-color: #E2E8F0;
    border-radius: 9999px;
}

.stProgress > div > div {
    background-color: #f2f2f2 !important;
    border-radius: 9999px;
    transition: width 0.3s ease;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: #F1F5F9;
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: #94A3B8;
    border-radius: 5px;
    border: 2px solid #F1F5F9;
}

::-webkit-scrollbar-thumb:hover {
    background: #64748B;
}

/* Footer */
.footer {
    margin-top: auto;
    padding: 2rem 0;
    text-align: center;
    font-size: 0.95rem;
    color: #64748B;
    border-top: 1px solid #E2E8F0;
}

.footer a {
    color: #000000;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.2s ease;
}

.footer a:hover {
    color: #1F2937;
    text-decoration: underline;
}

/* Spinner Animation */
.stSpinner {
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Select Boxes and Inputs */
.stSelectbox > div > div, .stTextInput > div > div {
    background-color: #FFFFFF;
    border-radius: 8px;
    border: 1px solid #CBD5E1;
    transition: all 0.3s ease;
    font-size: 1.1rem;
    padding: 0.75rem;
}

.stSelectbox > div > div:hover, .stTextInput > div > div:hover {
    border-color: #94A3B8;
    box-shadow: 0 2px 4px rgba(242, 242, 242,0.05);
}

.stSelectbox > div > div:focus-within, .stTextInput > div > div:focus-within {
    border-color: #000000;
    box-shadow: 0 0 0 3px rgba(242, 242, 242,0.1);
}

/* Responsive Design */
@media (max-width: 768px) {
    .stApp {
        padding: 1.5rem 1rem;
    }
    
    h1, .stTitle {
        font-size: 1.875rem;
    }
    
    .stSubheader {
        font-size: 1.25rem;
    }
    
    p, li, span, label {
        font-size: 1rem;
    }
    
    .stFileUploader > div > div > div {
        padding: 1.5rem;
    }
    
    .stFileUploader > div > button {
        padding: 0.625rem 1.25rem;
        font-size: 0.95rem;
    }
}

@media (max-width: 480px) {
    h1, .stTitle {
        font-size: 1.5rem;
    }
    
    .stSubheader {
        font-size: 1.125rem;
    }
    
    .stFileUploader > div > div > div {
        padding: 1rem;
    }
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