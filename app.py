import streamlit as st
import os
from engine import RAGEngine
from utils import load_environment_variables, process_uploaded_files

st.set_page_config(page_title="Financial RAG Dashboard", layout="wide", page_icon="📈")

# Premium CSS Injection (Clean Light Mode)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background: #f8fafc;
    }
    
    .stSidebar {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #1e40af, #4f46e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3.5rem !important;
        margin-bottom: 0rem;
    }
    
    .header-subtext {
        color: #475569 !important;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #6366f1);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease;
        width: 100%;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        color: white !important;
        border: none;
    }
    
    .stChatInputContainer {
        border-radius: 12px;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
    }
    
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    .stMarkdown p {
        color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
warnings = load_environment_variables()

st.title("Financial RAG Intelligence")
st.markdown('<p class="header-subtext">Enterprise-grade semantic analysis of earnings transcripts & SEC filings.</p>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Core Settings")
    model_choice = st.selectbox("LLM Architect", ["OpenAI", "Gemini"], index=None, placeholder="Awaiting configuration...")
    
    if model_choice:
        use_gemini = model_choice == "Gemini"
        # Only show specific API Key warnings if they selected that specific choice
        if use_gemini and any("GEMINI" in w for w in warnings):
            st.error("GEMINI_API_KEY is missing/invalid in your .env file!")
        elif not use_gemini and any("OPENAI" in w for w in warnings):
            st.error("OPENAI_API_KEY is missing/invalid in your .env file!")
    else:
        use_gemini = None

# Initialize Engine
@st.cache_resource
def get_engine(use_gemini):
    try:
        return RAGEngine(use_gemini=use_gemini)
    except Exception as e:
        return None

engine = None
if use_gemini is not None:
    engine = get_engine(use_gemini)
    if not engine:
        st.error("Engine failed to initialize. Check your API keys and restart.")
        st.stop()
else:
    # --- SPLASH SCREEN (Provider not selected) ---
    st.info("👋 **Welcome to Financial RAG Intelligence**")
    st.markdown("""
        To launch the intelligence engine, please select a core LLM Architect from the sidebar.
        
        **System Capabilities:**
        - **Chunking:** 95th Percentile Semantic Boundaries via Local HuggingFace (`all-MiniLM`)
        - **Storage:** Vectorized ChromaDB
        - **Retrieval:** Maximum Marginal Relevance (MMR)
    """)
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.header("📂 Document Ops")
    uploaded_files = st.file_uploader(
        "Upload Transcripts (PDF/TXT)", 
        accept_multiple_files=True,
        type=['txt', 'pdf']
    )
    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            with st.spinner("Processing documents into ChromaDB..."):
                try:
                    chunks = process_uploaded_files(uploaded_files, engine)
                    st.success(f"Successfully processed {len(uploaded_files)} files into {chunks} chunks!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

    # Show active knowledge base
    if engine and hasattr(engine, 'get_loaded_documents'):
        loaded_docs = engine.get_loaded_documents()
        if loaded_docs:
            st.markdown("---")
            st.header("📚 Active Knowledge Base")
            for doc in loaded_docs:
                st.markdown(f"- `{doc}`")

# Main UI
st.subheader("Ask Questions about the Transcripts")
user_query = st.chat_input("E.g., What were the key growth drivers mentioned in the last quarter?")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = engine.query(user_query)
                answer = response.get("result", response.get("answer", "Sorry, I couldn't process that."))
                metrics = response.get("metrics", {})
                sources = response.get("source_documents", [])
                
                st.markdown(answer)
                
                # Render Evaluation Metrics Dashboard
                if metrics:
                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    
                    dist = metrics.get('distance', 0)
                    m1.metric("Vector Distance (0 is Perfect)", f"{dist}", help="Cosine similarity distance of the best matched dataset chunk.")
                    
                    rel = metrics.get('relevance')
                    rel_str = f"{rel} / 10" if rel is not None else "N/A"
                    m2.metric("Answer Relevance", rel_str, help="AI Judge grading if the answer solves the user question.")
                    
                    faith = metrics.get('faithfulness')
                    faith_str = f"{faith}%" if faith is not None else "N/A"
                    m3.metric("Faithfulness (No Hallucination)", faith_str, help="AI Judge grading if claims are strictly backed by source facts.")
                
                if sources:
                    with st.expander("View Sources", expanded=False):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i+1}** (from `{doc.metadata.get('source', 'Unknown')}`):")
                            st.write(doc.page_content)
                            st.markdown("---")
                            
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
