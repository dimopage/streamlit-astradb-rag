import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
import os
import tempfile
from datetime import datetime

# Load settings from secrets.toml with error handling
try:
    ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_NAMESPACE = st.secrets.get("ASTRA_DB_NAMESPACE", None)
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()

# Debug: Print secrets to verify (remove in production)
st.write("Secrets loaded:", {
    "ASTRA_DB_API_ENDPOINT": ASTRA_DB_API_ENDPOINT,
    "ASTRA_DB_APPLICATION_TOKEN": "****" if ASTRA_DB_APPLICATION_TOKEN else None,
    "ASTRA_DB_NAMESPACE": ASTRA_DB_NAMESPACE
})

# Custom CSS for Inter font and GenIAlab-inspired styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #000000;
    }
    .stApp {
        background-color: #ffffff;
    }
    .logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 100px;
    }
    h1 {
        text-align: center;
        color: #000000;
        font-weight: 700;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .stTextInput > div > div > input {
        border: 2px solid #000000;
        border-radius: 8px;
        padding: 8px;
    }
    .stButton > button {
        background-color: #000000;
        color: #ffffff;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 700;
    }
    .stButton > button:hover {
        background-color: #333333;
    }
    .stFileUploader > div > div > div {
        border: 2px solid #000000;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.title("DocVectorizer for RAG")

# Input for use case to dynamically set collection name
use_case = st.text_input("Enter use case (e.g., technical, marketing)", value="default")
collection_name = f"rag_{use_case.lower().replace(' ', '_')}"

# File uploader with JSON support
uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "md", "json"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    for file in uploaded_files:
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
                st.warning(f"Unsupported file type: {file.type}")
                continue
            docs = loader.load()
            # Add metadata to each document
            for doc in docs:
                doc.metadata.update({
                    "filename": file.name,
                    "upload_date": datetime.now().isoformat(),
                    "file_type": file.type
                })
            documents.extend(docs)
        finally:
            os.unlink(tmp_file_path)  # Delete temporary file

    if documents:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
        chunks = text_splitter.split_documents(documents)

        # Generate embeddings with Hugging Face
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Failed to initialize HuggingFaceEmbeddings: {str(e)}")
            st.stop()

        # Create or access vector store with detailed error handling
        try:
            vectorstore = AstraDBVectorStore(
                collection_name=collection_name,
                embedding=embeddings,
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
                namespace=ASTRA_DB_NAMESPACE
            )
        except Exception as e:
            st.error(f"Failed to initialize AstraDBVectorStore: {str(e)}")
            st.stop()

        # Add documents to vector store with error handling
        try:
            vectorstore.add_documents(chunks)
            st.success(f"Documents successfully vectorized and stored in collection {collection_name}")
        except Exception as e:
            st.error(f"Failed to store documents in AstraDB: {str(e)}")
    else:
        st.warning("No documents were processed")
else:
    st.info("Please upload documents to proceed")