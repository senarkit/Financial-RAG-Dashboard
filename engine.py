import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGEngine:
    def __init__(self, use_gemini=False, openai_embedding_model="text-embedding-3-large", gemini_embedding_model="models/gemini-embedding-001"):
        self.use_gemini = use_gemini
        self.openai_embedding_model = openai_embedding_model
        self.gemini_embedding_model = gemini_embedding_model
        self.persist_directory = "./chroma_db"
        self.vectorstore = None
        self.qa_chain = None
        
        self.init_llm_and_embeddings()

    def init_llm_and_embeddings(self):
        try:
            if self.use_gemini:
                # Use local embeddings to completely bypass Google's 100/min embedding rate limit
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Keep Gemini as the actual thinking LLM for answering questions
                self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            else:
                self.embeddings = OpenAIEmbeddings(model=self.openai_embedding_model)
                self.llm = ChatOpenAI(temperature=0, model="gpt-4")
                
            self.init_vectorstore()
        except Exception as e:
            print(f"Error initializing models. Make sure API keys are set correctly: {e}")
            raise e
            
    def init_vectorstore(self):
        # Initialize chroma db
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embeddings
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        # Setup the QA chain using modern LCEL
        template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}

Question: {question}
Helpful Answer:"""
        prompt = PromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.qa_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def add_documents(self, documents):
        """
        Add documents to the vector store.
        documents: List of Langchain Document objects
        """
        if self.vectorstore is None:
            self.init_vectorstore()
            
        self.vectorstore.add_documents(documents)
        
    def evaluate_answer_relevance(self, question, answer):
        try:
            prompt = PromptTemplate.from_template(
                "You are an impartial judge. Rate the following answer on how well it addresses the user's question. "
                "Score 0 to 10 (10 being perfect). Return ONLY the integer number.\n\n"
                "Question: {question}\nAnswer: {answer}\nScore:"
            )
            chain = prompt | self.llm | StrOutputParser()
            score_str = chain.invoke({"question": question, "answer": answer})
            import re
            match = re.search(r'\d+', score_str)
            return int(match.group()) if match else None
        except Exception:
            return None

    def evaluate_faithfulness(self, answer, source_docs):
        try:
            context = "\n".join([doc.page_content for doc in source_docs])
            prompt = PromptTemplate.from_template(
                "You are an impartial judge fact-checking an AI answer. "
                "Read the source context carefully. Then read the generated answer. "
                "Calculate what percentage of the claims in the answer are strictly backed up by the source context. "
                "Return a score from 0 to 100 (100 being completely factual with zero hallucinations). "
                "Return ONLY the percentage number (e.g. 85).\n\n"
                "Source Context: {context}\n\nAnswer: {answer}\nScore:"
            )
            chain = prompt | self.llm | StrOutputParser()
            score_str = chain.invoke({"context": context, "answer": answer})
            import re
            match = re.search(r'\d+', score_str)
            return int(match.group()) if match else None
        except Exception:
            return None

    def get_loaded_documents(self):
        """Returns a list of unique source filenames currently loaded in ChromaDB."""
        if not self.vectorstore:
            return []
        try:
            results = self.vectorstore.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            sources = set()
            for m in metadatas:
                if m and "source" in m:
                    sources.add(m["source"])
            return sorted(list(sources))
        except Exception:
            return []

    def query(self, question):
        """
        Query the RAG system and run self-evaluation metrics.
        """
        if not self.qa_chain:
            return {"result": "Engine not initialized fully or no documents available."}
            
        try:
            # 1. Fetch nearest neighbor distance for retrieval confidence
            docs_and_scores = self.vectorstore.similarity_search_with_score(question, k=1)
            raw_distance = docs_and_scores[0][1] if docs_and_scores else 0.0
            
            # 2. Get the official MMR sources
            source_documents = self.retriever.invoke(question)
            
            # 3. Generate the actual answer
            answer = self.qa_chain.invoke(question)
            
            # 4. Auto-Grade the Answer Quality
            relevance_score = self.evaluate_answer_relevance(question, answer)
            
            # 5. Auto-Grade the Hallucination Rate
            faithfulness_score = self.evaluate_faithfulness(answer, source_documents)
            
            return {
                "result": answer,
                "source_documents": source_documents,
                "metrics": {
                    "distance": round(raw_distance, 3),
                    "relevance": relevance_score,
                    "faithfulness": faithfulness_score
                }
            }
        except Exception as e:
            return {"result": f"Error running query: {e}"}
