import os
import google.generativeai as genai
import chromadb
from typing import List, Dict, Tuple, Any
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from document_processor import DocumentProcessor

class RAGPipelineGemini:
    def __init__(self, google_api_key: str, persist_directory: str = "./chroma_db", collection_name: str = "cs_textbooks"):
        self.google_api_key = google_api_key
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize Google AI
        genai.configure(api_key=google_api_key)

        # Initialize embeddings and LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=google_api_key
        )

        # Initialize document processor
        self.doc_processor = DocumentProcessor()

        # Initialize vector store
        self.vector_store = None
        self.qa_chain = None

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create and populate a Chroma vector store from documents."""
        if not documents:
            raise ValueError("No documents provided to create vector store")

        print(f"Creating vector store with {len(documents)} document chunks...")

        # Create Chroma vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

        # Persist the vector store
        vector_store.persist()
        print(f"Vector store created and persisted to {self.persist_directory}")

        return vector_store

    def load_vector_store(self) -> Chroma:
        """Load an existing vector store from disk."""
        try:
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print(f"Loaded existing vector store from {self.persist_directory}")
            return vector_store
        except Exception as e:
            print(f"Could not load existing vector store: {e}")
            return None

    def initialize_vector_store(self, documents: List[Document] = None, force_recreate: bool = False) -> bool:
        """Initialize the vector store, either from existing or new documents."""
        if force_recreate:
            if documents is None:
                raise ValueError("Documents must be provided to recreate vector store")
            self.vector_store = self.create_vector_store(documents)
            return True
        else:
            # Try to load existing vector store
            self.vector_store = self.load_vector_store()

            if self.vector_store is None or documents:
                # Create new vector store if loading failed or new documents provided
                if documents is None:
                    # Try to process documents from data directory
                    documents = self.doc_processor.process_directory("data")

                if documents:
                    self.vector_store = self.create_vector_store(documents)
                    return True
                else:
                    print("No documents available to create vector store")
                    return False

            return True

    def create_qa_chain(self):
        """Create a question-answering chain using the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        # Create a retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most similar chunks
        )

        # Create a custom prompt template
        template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say that you don't know.
Try to be as helpful as possible and provide a detailed answer based on the context.

Context:
{context}

Question: {question}

Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        if self.qa_chain is None:
            self.create_qa_chain()

        try:
            # Query the chain
            result = self.qa_chain({"query": question})

            # Format the response
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": result.get("source_documents", [])
            }

            # Add source information
            sources = []
            for doc in response["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "Unknown")
                })

            response["sources"] = sources
            response["num_sources"] = len(sources)

            return response

        except Exception as e:
            return {
                "question": question,
                "answer": f"An error occurred while processing your question: {str(e)}",
                "source_documents": [],
                "sources": [],
                "num_sources": 0,
                "error": str(e)
            }

    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store."""
        if self.vector_store is None:
            self.initialize_vector_store(documents)
        else:
            # Add documents to existing store
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            print(f"Added {len(documents)} new document chunks to vector store")

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the current vector store."""
        if self.vector_store is None:
            return {"status": "No vector store initialized"}

        try:
            # Get collection info
            collection = self.vector_store._collection
            count = collection.count()

            return {
                "status": "Vector store active",
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"status": f"Error getting stats: {e}"}

    def simple_evaluation(self, test_questions: List[str]) -> Dict[str, Any]:
        """Perform a simple evaluation of the RAG system."""
        results = []

        for question in test_questions:
            print(f"Evaluating question: {question}")
            result = self.query(question)
            results.append(result)

        # Calculate basic metrics
        total_questions = len(test_questions)
        successful_queries = sum(1 for r in results if not r.get("error"))
        avg_sources = sum(r["num_sources"] for r in results) / total_questions if results else 0

        evaluation = {
            "total_questions": total_questions,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / total_questions if total_questions > 0 else 0,
            "average_sources_retrieved": avg_sources,
            "results": results
        }

        return evaluation

if __name__ == "__main__":
    # Test the RAG pipeline with Gemini
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Please set GOOGLE_API_KEY in your environment variables")
    else:
        # Initialize RAG pipeline
        rag = RAGPipelineGemini(api_key)

        # Initialize with documents from data directory
        if rag.initialize_vector_store():
            # Test queries
            test_questions = [
                "What is an algorithm?",
                "What are the main properties of good algorithms?",
                "What is Big O notation?",
                "What are the common data structures in computer science?"
            ]

            print("\nTesting RAG pipeline with Gemini...")
            evaluation = rag.simple_evaluation(test_questions)

            print(f"\nEvaluation Results:")
            print(f"Success Rate: {evaluation['success_rate']:.2%}")
            print(f"Average Sources Retrieved: {evaluation['average_sources_retrieved']:.1f}")

            # Print detailed results
            for result in evaluation["results"]:
                print(f"\nQ: {result['question']}")
                print(f"A: {result['answer'][:200]}...")
                print(f"Sources: {result['num_sources']}")
        else:
            print("Failed to initialize RAG pipeline")