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

# Custom CSS for GenIAlab.Space-inspired modern UI with forced font
st.markdown("""
    <style>
    * {
        font-family: ui-sans-serif, system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji" !important;
    }

    .stApp {
        background-color: #FFFFFF;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }

    /* Logo text */
    .logo-text {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #000000;
        margin: 30px 0 20px 0;
    }

    /* Title */
    h1 {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 40px;
    }

    /* File uploader container (outer box) */
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E5E5 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }

    /* File uploader drop area (inner box) */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #f2f3f5 !important;
        border: none !important;
        border-radius: 8px !important;
        color: #1f2937 !important;
    }

    /* File uploader drop area text and icons */
    div[data-testid="stFileUploaderDropzone"] span,
    div[data-testid="stFileUploaderDropzone"] svg {
        color: #1f2937 !important;
    }

    /* Upload button */
    div[data-testid="stFileUploader"] button {
        background-color: #1f2937 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border: none !important;
        transition: background-color 0.3s !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background-color: #374151 !important;
    }

    /* Feedback messages */
    .stSuccess, .stWarning, .stInfo {
        background-color: #F5F5F5;
        color: #000000;
        border-radius: 8px;
        padding: 15px;
        font-size: 1rem;
        animation: fadeIn 0.5s;
    }

    /* Progress bar container */
    .progress-container {
        margin: 20px 0;
        padding: 15px;
        background-color: #f2f3f5;
        border-radius: 8px;
        color: #1f2937;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Footer */
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

    /* Responsive design */
    @media (max-width: 768px) {
        .stApp {
            padding: 10px;
        }
        h1 {
            font-size: 2.5rem;
        }
        .logo-text {
            font-size: 1.5rem;
            margin: 20px 0;
        }
        div[data-testid="stFileUploaderDropzone"] {
            font-size: 0.9rem;
            padding: 10px;
        }
        div[data-testid="stFileUploader"] button {
            font-size: 0.9rem;
            padding: 8px 16px;
        }
        .stSuccess, .stWarning, .stInfo {
            font-size: 0.9rem;
        }
        .progress-container {
            font-size: 0.9rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Logo text
st.markdown('<div class="logo-text">GenIAlab.Space</div>', unsafe_allow_html=True)

# App title
st.title("DocVectorizer for RAG")

# Main container
with st.container():
    # Hardcode collection name
    collection_name = "rag_default"

    # File uploader
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "md", "json"], accept_multiple_files=True)

    # Track processed files to prevent duplicates
    processed_files = set()

    # Process uploaded files
    if uploaded_files:
        documents = []
        progress_container = st.container()
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()

        # Function to compute file hash
        def get_file_hash(file):
            hasher = hashlib.sha256()
            hasher.update(file.getvalue())
            return hasher.hexdigest()

        for i, file in enumerate(uploaded_files):
            file_hash = get_file_hash(file)
            if file_hash in processed_files:
                status_text.warning(f"Skipping duplicate file: {file.name}")
                continue

            processed_files.add(file_hash)
            status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {file.name}")
            progress = (i + 0.5) / len(uploaded_files) * 0.5  # First 50% for file processing
            progress_bar.progress(progress)

            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
                tmp_file.write(file.getvalue())
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
                    status_text.warning(f"Unsupported file type: {file.type}")
                    continue
                docs = loader.load()
                # Add metadata to each document
                for doc in docs:
                    doc.metadata.update({
                        "filename": file.name,
                        "upload_date": datetime.now().isoformat(),
                        "file_type": file.type,
                        "file_hash": file_hash
                    })
                documents.extend(docs)
            except Exception as e:
                status_text.error(f"Error processing {file.name}: {str(e)}")
            finally:
                os.unlink(tmp_file_path)  # Delete temporary file

        if documents:
            status_text.text("Splitting documents and generating embeddings...")
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
            chunks = text_splitter.split_documents(documents)
            progress_bar.progress(0.6)  # 60% after splitting

            # Generate embeddings with Hugging Face
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            except Exception as e:
                status_text.error(f"Failed to initialize HuggingFaceEmbeddings: {str(e)}")
                st.stop()

            # Create or access vector store with error handling
            try:
                vectorstore = AstraDBVectorStore(
                    collection_name=collection_name,
                    embedding=embeddings,
                    api_endpoint=ASTRA_DB_API_ENDPOINT,
                    token=ASTRA_DB_APPLICATION_TOKEN,
                    namespace=ASTRA_DB_NAMESPACE
                )
            except Exception as e:
                status_text.error(f"Failed to initialize AstraDBVectorStore: {str(e)}")
                st.stop()

            # Add documents to vector store with error handling
            try:
                status_text.text("Vectorizing documents...")
                vectorstore.add_documents(chunks)
                progress_bar.progress(1.0)  # 100% when done
                status_text.success(f"Documents successfully vectorized and stored in collection {collection_name}")
            except Exception as e:
                status_text.error(f"Failed to store documents in AstraDB: {str(e)}")
        else:
            status_text.warning("No documents were processed")
    else:
        st.info("Please upload documents to proceed")

# Footer
st.markdown(
    '<div class="footer">Powered by <a href="https://genialab.space/" target="_blank">GenIAlab.Space</a></div>',
    unsafe_allow_html=True
)