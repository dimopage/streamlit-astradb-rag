import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from astrapy import DataAPIClient
import os
import tempfile
from datetime import datetime
import hashlib

# Load settings from secrets.toml with error handling
try:
    ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()

# Initialize AstraDB client
client = DataAPIClient(token=ASTRA_DB_APPLICATION_TOKEN)
database = client.get_database(ASTRA_DB_API_ENDPOINT)

# Custom CSS for GenIAlab.Space-inspired modern UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: ui-sans-serif, system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
        color: #000000;
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

    /* Selectbox and Input field */
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 12px;
        font-size: 1rem;
        background-color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.3s;
    }

    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-color: #000000;
    }

    /* File uploader */
    .stFileUploader > div > div > div {
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 12px;
        background-color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #000000;
    }

    /* Upload button */
    .stFileUploader > div > button {
        background-color: #000000;
        color: #FFFFFF;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1rem;
        font-weight: 700;
        border: none;
        transition: background-color 0.3s;
    }

    .stFileUploader > div > button:hover {
        background-color: #333333;
    }

    /* Feedback messages */
    .stSuccess, .stWarning, .stInfo, .stError {
        background-color: #F5F5F5;
        color: #000000;
        border-radius: 8px;
        padding: 15px;
        font-size: 1rem;
        animation: fadeIn 0.5s;
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
        .stSelectbox > div > div > select,
        .stTextInput > div > div > input,
        .stFileUploader > div > div > div {
            font-size: 0.9rem;
            padding: 10px;
        }
        .stFileUploader > div > button {
            font-size: 0.9rem;
            padding: 8px 16px;
        }
        .stSuccess, .stWarning, .stInfo, .stError {
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
    # Fetch namespaces
    try:
        namespaces = database.list_namespaces()
        if not namespaces:
            st.error("No namespaces found in the database.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to fetch namespaces: {str(e)}")
        st.stop()

    # Namespace selection
    selected_namespace = st.selectbox("Select Namespace", namespaces, index=0)

    # Fetch collections for the selected namespace
    try:
        collections = database.get_namespace(selected_namespace).list_collections()
        collections = collections if collections else ["rag_default"]
    except Exception as e:
        st.error(f"Failed to fetch collections: {str(e)}")
        collections = ["rag_default"]

    # Two-column layout for collection and uploader
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        # Collection selection
        selected_collection = st.selectbox("Select or Enter Collection", collections + ["New Collection"], index=0)
        if selected_collection == "New Collection":
            collection_name = st.text_input("Enter new collection name", value="rag_default")
        else:
            collection_name = selected_collection

    with col2:
        # File uploader with JSON support
        uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "md", "json"], accept_multiple_files=True)

    # Initialize metadata collection for tracking uploaded files
    try:
        metadata_store = AstraDBVectorStore(
            collection_name="file_metadata",
            embedding=None,  # No embeddings needed for metadata
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=selected_namespace
        )
    except Exception as e:
        st.error(f"Failed to initialize metadata store: {str(e)}")
        st.stop()

    # Process uploaded files
    if uploaded_files:
        documents = []
        for file in uploaded_files:
            # Calculate file hash
            file_content = file.getvalue()
            file_hash = hashlib.sha256(file_content).hexdigest()

            # Check if file is duplicate
            existing_files = metadata_store._collection.find({"filename": file.name, "file_hash": file_hash})
            if existing_files.count() > 0:
                st.warning(f"File '{file.name}' is a duplicate and will not be uploaded.")
                continue

            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
                tmp_file.write(file_content)
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
                        "file_type": file.type,
                        "file_hash": file_hash
                    })
                documents.extend(docs)

                # Store file metadata
                metadata_store._collection.insert_one({
                    "filename": file.name,
                    "file_hash": file_hash,
                    "upload_date": datetime.now().isoformat(),
                    "collection_name": collection_name
                })
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

            # Create or access vector store with error handling
            try:
                vectorstore = AstraDBVectorStore(
                    collection_name=collection_name,
                    embedding=embeddings,
                    api_endpoint=ASTRA_DB_API_ENDPOINT,
                    token=ASTRA_DB_APPLICATION_TOKEN,
                    namespace=selected_namespace
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
            st.warning("No new documents were processed")
    else:
        st.info("Please upload documents to proceed")

# Footer
st.markdown(
    '<div class="footer">Powered by <a href="https://genialab.space/" target="_blank">GenIAlab.Space</a></div>',
    unsafe_allow_html=True
)