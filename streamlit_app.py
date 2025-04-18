import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredJSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_astradb import AstraDBVectorStore
import os
import tempfile
from datetime import datetime

# Load settings from secrets.toml
ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_NAMESPACE = st.secrets.get("ASTRA_DB_NAMESPACE", None)
OLLAMA_HOST = st.secrets["OLLAMA_HOST"]

# App title
st.title("Document Vectorizer for RAG with Ollama")

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
                loader = UnstructuredJSONLoader(tmp_file_path)
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

        # Generate embeddings with Ollama
        embeddings = OllamaEmbeddings(model="all-minilm:latest", base_url=OLLAMA_HOST)

        # Create or access vector store
        vectorstore = AstraDBVectorStore(
            collection_name=collection_name,
            embedding=embeddings,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_NAMESPACE
        )

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