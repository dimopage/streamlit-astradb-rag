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

# Logo text
st.markdown('<div class="logo-text">GenIAlab.Space</div>', unsafe_allow_html=True)

# App title
st.title("DocVectorizer for RAG")

# Main container
with st.container():
    # Hardcode use case
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