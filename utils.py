import os
from dotenv import load_dotenv
import streamlit as st

def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Check for required keys and warn if they are missing
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    warnings = []
    if not openai_key or openai_key == "your_openai_api_key_here":
        warnings.append("OPENAI_API_KEY is not set correctly in .env")
        
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        warnings.append("GEMINI_API_KEY is not set correctly in .env")
        
    return warnings

def process_uploaded_files(uploaded_files, engine):
    """
    Process uploaded files and add them to the vector store.
    This is a helper function that can extract text from PDFs or TXTs.
    """
    from langchain_core.documents import Document
    import tempfile
    
    documents = []
    for file in uploaded_files:
        # Simple text extraction based on file extension
        if file.name.endswith('.txt'):
            text = file.read().decode('utf-8')
            documents.append(Document(page_content=text, metadata={"source": file.name}))
        elif file.name.endswith('.pdf'):
            try:
                from pypdf import PdfReader
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                documents.append(Document(page_content=text, metadata={"source": file.name}))
            except ImportError:
                st.error("pypdf is not installed. Please try uploading txt files or install pypdf.")
            except Exception as e:
                st.error(f"Error processing PDF {file.name}: {e}")
                
    if documents:
        # Split documents using SemanticChunker with a LOCAL model to bypass rate limits
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # This runs 100% locally on your Mac's CPU to determine chunk breakpoints
        local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        text_splitter = SemanticChunker(
            local_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        split_docs = text_splitter.split_documents(documents)
        engine.add_documents(split_docs)
        return len(split_docs)
        
    return 0
