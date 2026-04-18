# 📈 Financial RAG Intelligence Dashboard

An enterprise-grade Retrieve-and-Generate (RAG) system built to analyze and extract insights from dense financial documents like SEC filings and earnings transcripts. The architecture is specifically optimized to avoid API rate-limit bottlenecks by performing advanced semantic chunking and mathematical vector matching locally on your hardware, relying on Foundation Models strictly for conversational generation and robust evaluation.

## 🌟 Key Features

- **Hybrid Embedding Architecture:** Utilizes local CPU-bound `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) to perform semantic breakpoint chunking and vector storage. This entirely bypasses Google/OpenAI free-tier embedding rate limits.
- **Native Evaluator System (LLM-as-a-Judge):** Dynamically scores every generated answer on a three-tier metric scale:
  - `Vector Distance`: Mathematical cosine-similarity score of the retrieved chunk.
  - `Answer Relevance`: An evaluation prompt verifying the AI effectively answered the core question (0-10).
  - `Faithfulness`: An evaluation prompt checking against hallucinations strictly using the source context (0-100%).
- **Semantic Chunking:** Leverages the 95th-percentile breakpoint shift algorithm to ensure financial paragraphs are grouped intelligently by meaning, rather than arbitrary character counts.
- **Custom UI:** Premium Streamlit dashboard featuring custom Light-Mode CSS, micro-animations, and dynamic sidebar metadata extracting active documents from ChromaDB.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Foundational LLMs:** Google Generative AI (`gemini-2.5-flash`), OpenAI (`gpt-4`)
- **Embeddings:** HuggingFace `sentence-transformers`
- **Vector Database:** ChromaDB (Locally Persisted)
- **Pipeline:** LangChain (LCEL `VectorStoreRetrieverMemory` with Maximum Marginal Relevance)
- **Document Processing:** `pypdf`

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Financial RAG Dashboard"
   ```

2. **Install Dependencies:**
   Ensure you are running Python 3.12, then install the required locked dependencies.
   *(Note: Certain packages like `transformers` and `numpy` are strictly bounded to prevent known macOS PyTorch definition crashes).*
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   Create a `.env` file in the root directory and add your foundational API keys. The system will throw an error on startup if these are incorrectly formatted.
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 🎮 Usage

Launch the intelligent dashboard via terminal:

```bash
python3.12 -m streamlit run app.py
```

1. **Select an Architect:** In the left sidebar, choose whether the system should be powered by OpenAI or Google Gemini.
2. **Process Documents:** Upload your PDF/TXT transcripts. The system will parse, chunk, and embed them locally in the `chroma_db` folder. You will not hit API rate limits during this phase.
3. **Query Engine:** Type a prompt in the chat. The system will answer the question, display its extraction sources, and publicly grade its own hallucination rate dynamically below the answer!

## 📁 Repository Structure

- `app.py`: The Main Streamlit front-end handling State Logic, custom CSS, layout, and visual evaluation gauges.
- `engine.py`: Defines the `RAGEngine` object. Initializes LCEL retrieval pipelines, sets ChromaDB persisting constraints, and houses all LLM-as-a-judge evaluation prompts.
- `utils.py`: Contains helpers for document ingestion and the percentile-based local `SemanticChunker`.
- `chroma_db/`: Local persisting directory for storing vectorized financial datasets natively.
