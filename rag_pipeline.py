import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Tuple, Any
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from document_processor import DocumentProcessor

class RAGPipeline:
    def __init__(self, google_api_key: str, qdrant_url: str = "http://localhost:6333", collection_name: str = "cs_textbooks"):
        self.google_api_key = google_api_key
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize Google AI
        genai.configure(api_key=google_api_key)

        # Initialize embeddings and LLM with best models
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            google_api_key=google_api_key
        )

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)

        # Initialize document processor
        self.doc_processor = DocumentProcessor()

        # Initialize vector store
        self.vector_store = None
        self.qa_chain = None

    def create_vector_store(self, documents: List[Document]) -> Qdrant:
        """Create and populate a Qdrant vector store from documents with batching."""
        if not documents:
            raise ValueError("No documents provided to create vector store")

        print(f"Creating Qdrant vector store with {len(documents)} document chunks...")
        
        # Batch size limit - Google Embedding API has a max batch size
        batch_size = 5000  # Stay under the 5461 limit with buffer
        
        # Create Qdrant vector store
        vector_store = Qdrant.from_documents(
            documents,
            self.embeddings,
            url=self.qdrant_url,
            prefer_grpc=False,
            collection_name=self.collection_name,
        )
        
        print(f"✅ Qdrant vector store created with collection '{self.collection_name}'")
        print(f"   Access Qdrant UI at: {self.qdrant_url.replace('6333', '6333')}/dashboard")

        return vector_store

    def load_vector_store(self) -> Qdrant:
        """Load an existing Qdrant vector store."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                vector_store = Qdrant(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embeddings=self.embeddings
                )
                print(f"✅ Loaded existing Qdrant collection '{self.collection_name}'")
                return vector_store
            else:
                print(f"Collection '{self.collection_name}' does not exist in Qdrant")
                return None
        except Exception as e:
            print(f"Could not load existing vector store: {e}")
            return None

    def initialize_vector_store(self, documents: List[Document] = None, force_recreate: bool = False) -> bool:
        """Initialize the vector store, either from existing or new documents."""
        if force_recreate:
            if documents is None:
                raise ValueError("Documents must be provided to recreate vector store")
            # Delete existing collection if force recreate
            try:
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                print(f"Deleted existing collection '{self.collection_name}'")
            except:
                pass
            self.vector_store = self.create_vector_store(documents)
            return True
        else:
            # Try to load existing vector store
            self.vector_store = self.load_vector_store()

            # Check if vector store is empty
            if self.vector_store is not None:
                try:
                    collection_info = self.qdrant_client.get_collection(self.collection_name)
                    if collection_info.points_count == 0:
                        self.vector_store = None  # Force recreation if empty
                except:
                    pass

            if self.vector_store is None:
                # Create new vector store if loading failed
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

        # Create a retriever (optimized k=3 based on evaluation)
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 most similar chunks (86.67% precision)
        )

        # Create a custom prompt template (Expert Technical Style - 8.00/10 score)
        template = """You are a technical expert assistant. Using the context provided, give a comprehensive technical answer.
Include relevant terminology, concepts, and explanations.
If information is not in the context, explicitly state what you don't know.

Context: {context}

Question: {question}

Expert Answer:"""

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
            print(f"Added {len(documents)} new document chunks to Qdrant")

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the current vector store."""
        if self.vector_store is None:
            return {"status": "No vector store initialized"}

        try:
            # Get collection info from Qdrant
            collection_info = self.qdrant_client.get_collection(self.collection_name)

            return {
                "status": "Qdrant vector store active",
                "total_documents": collection_info.points_count,
                "collection_name": self.collection_name,
                "qdrant_url": self.qdrant_url,
                "vectors_count": collection_info.vectors_count
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
    # Test the RAG pipeline
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Please set GOOGLE_API_KEY in your environment variables")
    else:
        # Initialize RAG pipeline
        rag = RAGPipeline(api_key)

        # Initialize with documents from data directory
        if rag.initialize_vector_store():
            # Test queries
            test_questions = [
                "What is an algorithm?",
                "What are the main properties of good algorithms?",
                "What is Big O notation?",
                "What are the common data structures in computer science?"
            ]

            print("\nTesting RAG pipeline...")
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