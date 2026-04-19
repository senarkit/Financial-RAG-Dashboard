# QNA generated through Agent Workflow

## Question (a): How does the workflow add files to the RAG engine?
### Step-by-Step Explanation:
1. **Uploading Files via Streamlit Dashboard**:
   - The Streamlit dashboard in `app.py` provides a user interface for uploading files.
   - The `st.file_uploader` widget is used to upload `.txt` or `.pdf` files.
   - Example code snippet:
     ```python
     uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["txt", "pdf"])
     ```

2. **Processing Uploaded Files**:
   - The uploaded files are passed to the `process_uploaded_files` function in `utils.py`.
   - This function extracts text from `.txt` and `.pdf` files and converts them into LangChain `Document` objects.
   - Key steps:
     - `.txt` files: Read and decode the content.
     - `.pdf` files: Use the `pypdf` library to extract text from each page.
   - Example usage:
     ```python
     documents = process_uploaded_files(uploaded_files, engine)
     ```

3. **Adding Documents to the RAG Engine**:
   - Once the documents are processed, they are added to the vector store using the `add_documents` method of the `RAGEngine` class in `engine.py`.
   - Example usage:
     ```python
     engine.add_documents(documents)
     ```

### Summary:
- Files are uploaded via the Streamlit dashboard.
- The `process_uploaded_files` function extracts text and converts files into `Document` objects.
- The `add_documents` method in `RAGEngine` adds the documents to the vector store for embedding and retrieval.

---

## Question (b): Where are the embeddings stored?
### Step-by-Step Explanation:
1. **Persistence Directory**:
   - The `RAGEngine` class defines a `persist_directory` parameter for Chroma, set to `./chroma_db`:
     ```python
     self.persist_directory = "./chroma_db"
     ```

2. **Chroma Vector Store**:
   - The `Chroma` vector store is initialized with this directory:
     ```python
     self.vectorstore = Chroma(
         persist_directory=self.persist_directory, 
         embedding_function=self.embeddings
     )
     ```

3. **Storage on Disk**:
   - Chroma uses the `persist_directory` to store embeddings and metadata on disk.
   - This ensures that the vector store can be reloaded later without needing to recompute embeddings.

4. **Persistence Behavior**:
   - When documents are added to the vector store using the `add_documents` method, the embeddings are computed and stored in the `./chroma_db` directory.
   - This directory contains files and data structures managed by Chroma for efficient retrieval.

### Summary:
- The embeddings and metadata are stored persistently in the `./chroma_db` directory.
- If you want to back up or move the vector store, you can copy this directory.
- If you delete the `./chroma_db` directory, the stored embeddings will be lost, and you will need to reprocess and re-embed the documents.

---

## Question (c): What is meant by Hybrid? Explain the embeddings and search used. Does it include keyword-based (BM25) and semantic search?

### Explanation:
1. **Hybrid Search**:
   - Hybrid search combines **semantic search** (using embeddings) and **keyword-based search** (like BM25) to retrieve the most relevant documents.
   - This approach leverages the strengths of both methods:
     - **Semantic Search**: Finds results based on the meaning of the query and documents, even if exact keywords are not present.
     - **Keyword-Based Search (BM25)**: Matches exact terms in the query and documents, ensuring precision for keyword-heavy queries.

2. **Embeddings Used**:
   - The `RAGEngine` uses embeddings to represent documents and queries as vectors in a high-dimensional space.
   - Depending on the configuration:
     - **OpenAI Embeddings**: Used when `use_gemini=False`.
     - **HuggingFace Embeddings**: Used when `use_gemini=True`.
   - These embeddings capture the semantic meaning of text, enabling similarity-based retrieval.

3. **Search Mechanism**:
   - The `Chroma` vector store supports hybrid search by combining:
     - **BM25**: A traditional keyword-based ranking algorithm.
     - **Semantic Search**: Finds documents with similar vector representations to the query.
   - The `search_type="mmr"` (Maximal Marginal Relevance) is used to diversify results by balancing relevance and novelty.

4. **Does it include BM25 and Semantic Search?**
   - Yes, the hybrid approach includes both:
     - **BM25** ensures that keyword matches are prioritized.
     - **Semantic Search** ensures that conceptually similar documents are retrieved, even if exact keywords are missing.

### Summary:
- Hybrid search combines BM25 (keyword-based) and semantic search (embedding-based).
- The `RAGEngine` uses embeddings (OpenAI or HuggingFace) and the `Chroma` vector store to implement this hybrid approach.
- This ensures both precision and semantic relevance in document retrieval.

---

## Question (d): Once a file is uploaded and embeddings are stored in the `chroma_db` folder on disk, how can the same on-disk information be reused for answering questions without redoing embedding?

### Explanation:
1. **Persistence of Embeddings**:
   - The `RAGEngine` uses the `Chroma` vector store with a `persist_directory` set to `./chroma_db`.
   - This ensures that embeddings and metadata are stored persistently on disk.

2. **Reusing Stored Embeddings**:
   - When the `RAGEngine` is initialized, it automatically loads the existing embeddings from the `chroma_db` directory.
   - The `Chroma` vector store reads the stored data and makes it available for retrieval without recomputing embeddings.

3. **Workflow for Reuse**:
   - Ensure the `chroma_db` directory is intact and contains the previously stored embeddings.
   - Initialize the `RAGEngine` as usual:
     ```python
     engine = RAGEngine()
     ```
   - The `init_vectorstore` method in `RAGEngine` will load the stored embeddings:
     ```python
     self.vectorstore = Chroma(
         persist_directory=self.persist_directory, 
         embedding_function=self.embeddings
     )
     ```
   - You can then query the engine without re-uploading or re-embedding files:
     ```python
     result = engine.query("Your question here")
     print(result)
     ```

4. **Advantages**:
   - Saves time and computational resources by avoiding redundant embedding calculations.
   - Ensures consistency in results as the same embeddings are reused.

### Summary:
- The `chroma_db` folder stores embeddings persistently on disk.
- The `RAGEngine` automatically loads these embeddings during initialization, allowing you to query the system without redoing the embedding process.

---

## Question (e): Provide a flow diagram on the order of files and functions that get executed from the time the Python Streamlit command is run and the app opens.

### Explanation:
Below is a flow diagram that outlines the sequence of file executions and functions invoked:

```mermaid
graph TD
    A[Run Streamlit Command] --> B[app.py]
    B --> C[st.set_page_config() (app.py)]
    C --> D[load_environment_variables() (utils.py)]
    D --> E[RAGEngine.__init__() (engine.py)]
    E --> F[init_llm_and_embeddings() (engine.py)]
    F --> G[init_vectorstore() (engine.py)]
    G --> H[Streamlit Dashboard Opens]
    H --> I[st.file_uploader() (app.py)]
    I --> J[process_uploaded_files() (utils.py)]
    J --> K[RAGEngine.add_documents() (engine.py)]
    K --> L[RAGEngine.query() (engine.py)]
```

### Detailed Steps:
1. **Run Streamlit Command**:
   - The Streamlit app is started using the command:
     ```bash
     streamlit run app.py
     ```

2. **`app.py` Execution**:
   - The `app.py` file is executed, and the Streamlit UI is initialized using:
     ```python
     st.set_page_config()
     ```

3. **Load Environment Variables**:
   - The `load_environment_variables()` function in `utils.py` is called to load API keys and other configurations.

4. **Initialize RAGEngine**:
   - The `RAGEngine` class in `engine.py` is instantiated:
     ```python
     engine = RAGEngine()
     ```
   - This triggers the `__init__()` method, which calls:
     - `init_llm_and_embeddings()` to set up the language model and embeddings.
     - `init_vectorstore()` to initialize the Chroma vector store.

5. **Streamlit Dashboard Opens**:
   - The Streamlit dashboard is displayed, allowing user interaction.

6. **File Upload**:
   - Users upload files via the `st.file_uploader()` widget in `app.py`.

7. **Process Uploaded Files**:
   - The `process_uploaded_files()` function in `utils.py` processes the uploaded files, extracting text and converting them into LangChain `Document` objects.

8. **Add Documents to Vector Store**:
   - The `add_documents()` method in `RAGEngine` adds the processed documents to the Chroma vector store.

9. **Query the Engine**:
   - The `query()` method in `RAGEngine` is invoked to answer user queries based on the uploaded documents.

### Summary:
- The flow starts with running the Streamlit command, which triggers `app.py`.
- Functions from `utils.py` and `engine.py` are invoked to load configurations, initialize the engine, and process user inputs.
- The dashboard allows users to upload files, which are processed and added to the vector store for querying.