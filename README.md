# DocVectorizer for RAG

A minimal Streamlit application to vectorize documents (PDF, TXT, MD, JSON) using Ollama embeddings and store them in AstraDB for Retrieval-Augmented Generation (RAG).

Inspired by the clean and modern design of [GenIAlab](https://genialab.space/), this project provides a simple interface to process documents and prepare them for RAG workflows.

## Features
- Upload and process multiple document types: PDF, TXT, Markdown, JSON.
- Dynamic collection naming based on use case (e.g., `rag_technical`, `rag_marketing`).
- Embeddings generated using `all-minilm:latest` model via Ollama.
- Vector storage in AstraDB with metadata support (`filename`, `upload_date`, `file_type`).
- Clean UI with Inter font, black/white theme, and custom logo.

## Prerequisites
To run this project locally or deploy it, you need:
- **Python 3.8+**: Install from [python.org](https://www.python.org/).
- **Git**: For cloning the repository.
- **AstraDB Account**: Sign up at [astra.datastax.com](https://astra.datastax.com/) and create a vector database.
- **Ollama Server**: Install Ollama and pull the `all-minilm:latest` model. Ensure the server is accessible via a public URL (e.g., `http://ollama.yourdomain.com:11434`).
- **Streamlit Community Cloud Account**: For deployment (optional).

## Installation
Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dimopage/streamlit-astradb-rag.git
   cd streamlit-astradb-rag
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Secrets**:
   Create a file named `secrets.toml` in the `.streamlit/` directory:
   ```bash
   mkdir .streamlit
   nano .streamlit/secrets.toml
   ```
   Add the following content, replacing placeholders with your credentials:
   ```toml
   [secrets]
   ASTRA_DB_API_ENDPOINT = "https://your-astra-db-api-endpoint"
   ASTRA_DB_APPLICATION_TOKEN = "AstraCS:your-token"
   ASTRA_DB_NAMESPACE = "your_namespace"
   OLLAMA_HOST = "http://ollama.yourdomain.com:11434"
   ```

## Usage
1. **Run the Application Locally**:
   ```bash
   streamlit run streamlit_app.py
   ```
   The app will open in your browser at `http://localhost:8501`.

2. **Upload Documents**:
   - Enter a use case (e.g., `technical`, `marketing`) in the text input to name the AstraDB collection (e.g., `rag_technical`).
   - Upload one or more files (PDF, TXT, MD, JSON) using the file uploader.
   - The app will:
     - Load and chunk documents (2500 characters per chunk, 250 characters overlap).
     - Generate embeddings using `all-minilm:latest` via Ollama.
     - Store vectors and metadata (`filename`, `upload_date`, `file_type`) in AstraDB.
   - A success message will confirm storage (e.g., "Documents successfully vectorized and stored in collection rag_technical").

3. **Verify in AstraDB**:
   - Log in to [astra.datastax.com](https://astra.datastax.com/).
   - Check your database and namespace for the collection (e.g., `rag_technical`).
   - Ensure vectors and metadata are stored correctly.

## Deployment on Streamlit Community Cloud
To deploy the app publicly:

1. **Push to GitHub**:
   Ensure all files are committed and pushed to your public repository:
   ```bash
   git add .
   git commit -m "Update project for deployment"
   git push origin main
   ```

2. **Create a Streamlit Cloud App**:
   - Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
   - Click **New App** and select your GitHub repository (`dimopage/streamlit-astradb-rag`).
   - Set the main file to `streamlit_app.py`.
   - Click **Advanced Settings** and add the following secrets (same as `secrets.toml`):
     ```toml
     ASTRA_DB_API_ENDPOINT = "https://your-astra-db-api-endpoint"
     ASTRA_DB_APPLICATION_TOKEN = "AstraCS:your-token"
     ASTRA_DB_NAMESPACE = "your_namespace"
     OLLAMA_HOST = "http://ollama.yourdomain.com:11434"
     ```

3. **Deploy**:
   - Click **Deploy** and wait for the app to build.
   - Once deployed, access the app via the provided URL (e.g., `https://your-app-name.streamlit.app`).
   - Test by uploading documents and verifying success messages.

## Troubleshooting
- **Ollama Connection Error**: Ensure your Ollama server is running and accessible at the specified `OLLAMA_HOST` URL.
- **AstraDB Error**: Verify your API endpoint, token, and namespace in `secrets.toml` or Streamlit Cloud secrets.
- **File Upload Issues**: Check that files are valid (e.g., non-corrupted PDFs, well-formed JSON).
- **Dependencies**: If installation fails, ensure `requirements.txt` matches the listed versions.

## Contributing
Feel free to open issues or submit pull requests on [GitHub](https://github.com/dimopage/streamlit-astradb-rag). Suggestions for new features (e.g., RAG retrieval, additional file types) are welcome.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or support, open an issue on GitHub or contact [hi@genialab.space].

---
Built with ❤️ for document processing and RAG workflows.