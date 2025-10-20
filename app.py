import streamlit as st
import os
import tempfile
from typing import List, Dict
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="CS Textbooks RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .source-box {
        background-color: #fff5e6;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #ff9500;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'vector_store_initialized' not in st.session_state:
        st.session_state.vector_store_initialized = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None

def check_api_key():
    """Check if Google Gemini API key is available."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ Google Gemini API Key not found!")
        st.info("Please set your GOOGLE_API_KEY in the environment variables or .env file")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return False
    return True

def initialize_rag_pipeline():
    """Initialize the RAG pipeline."""
    if not check_api_key():
        return False

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        st.session_state.rag_pipeline = RAGPipeline(api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        return False

def initialize_vector_store():
    """Initialize or load the vector store."""
    if st.session_state.rag_pipeline is None:
        if not initialize_rag_pipeline():
            return False

    try:
        with st.spinner("Initializing vector store..."):
            success = st.session_state.rag_pipeline.initialize_vector_store()
            if success:
                st.session_state.vector_store_initialized = True
                stats = st.session_state.rag_pipeline.get_vector_store_stats()
                st.success(f"✅ Vector store initialized with {stats['total_documents']} document chunks!")
                return True
            else:
                st.error("❌ Failed to initialize vector store")
                return False
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return False

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add them to the vector store."""
    if not uploaded_files or st.session_state.rag_pipeline is None:
        return 0

    total_chunks = 0
    doc_processor = DocumentProcessor()
    new_documents = []

    with st.spinner(f"Processing {len(uploaded_files)} uploaded files..."):
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Process the file
                documents = doc_processor.process_document(tmp_file_path, uploaded_file.name)
                new_documents.extend(documents)
                total_chunks += len(documents)
                st.success(f"✅ Processed {uploaded_file.name}: {len(documents)} chunks")

            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")

            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)

        # Add new documents to vector store
        if new_documents:
            try:
                st.session_state.rag_pipeline.add_documents(new_documents)
                st.success(f"🎉 Successfully added {total_chunks} chunks to the knowledge base!")
            except Exception as e:
                st.error(f"❌ Error adding documents to vector store: {e}")

    return total_chunks

def run_evaluation():
    """Run evaluation on test questions."""
    test_questions = [
        "What is an algorithm?",
        "What are the main properties of good algorithms?",
        "What is Big O notation used for?",
        "What data structures are discussed in the materials?",
        "What is time complexity?",
        "How do you analyze algorithm efficiency?",
        "What are common programming paradigms?",
        "What is computer architecture about?"
    ]

    if st.session_state.rag_pipeline is None:
        st.error("❌ RAG pipeline not initialized")
        return

    with st.spinner("Running evaluation..."):
        try:
            evaluation = st.session_state.rag_pipeline.simple_evaluation(test_questions)
            st.session_state.evaluation_results = evaluation

            # Display evaluation results
            st.markdown("### 📊 Evaluation Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Questions", evaluation["total_questions"])
            with col2:
                st.metric("Success Rate", f"{evaluation['success_rate']:.1%}")
            with col3:
                st.metric("Avg Sources Retrieved", f"{evaluation['average_sources_retrieved']:.1f}")

            # Show detailed results
            st.markdown("### 🔍 Detailed Results")
            for i, result in enumerate(evaluation["results"]):
                with st.expander(f"Q{i+1}: {result['question']}"):
                    st.write("**Answer:**")
                    st.write(result["answer"])
                    st.write(f"**Sources Retrieved:** {result['num_sources']}")
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")

        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">📚 CS Textbooks RAG System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.markdown("## ⚙️ System Controls")

    # Initialize system button
    if st.sidebar.button("🚀 Initialize System", type="primary"):
        if initialize_vector_store():
            st.rerun()

    # Show system status
    if st.session_state.vector_store_initialized and st.session_state.rag_pipeline:
        stats = st.session_state.rag_pipeline.get_vector_store_stats()
        st.sidebar.markdown("### 📈 System Status")
        st.sidebar.success("✅ System Ready")
        st.sidebar.info(f"📄 Documents: {stats['total_documents']}")
        st.sidebar.info(f"🗂️ Collection: {stats['collection_name']}")
    else:
        st.sidebar.warning("⚠️ System Not Initialized")

    st.sidebar.markdown("---")

    # File upload section
    st.sidebar.markdown("### 📁 Add Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload additional textbooks or documents"
    )

    if uploaded_files and st.sidebar.button("➕ Add Documents"):
        if st.session_state.vector_store_initialized:
            process_uploaded_files(uploaded_files)
        else:
            st.sidebar.error("Please initialize the system first")

    # Evaluation button
    if st.sidebar.button("🧪 Run Evaluation"):
        if st.session_state.vector_store_initialized:
            run_evaluation()
        else:
            st.sidebar.error("Please initialize the system first")

    # Main content area
    if not st.session_state.vector_store_initialized:
        # Welcome screen
        st.markdown("### 🎯 Welcome to the CS Textbooks RAG System!")
        st.markdown("""
        This system allows you to:

        - **📖 Upload and process** computer science textbooks (PDF/TXT)
        - **❓ Ask questions** about the uploaded materials
        - **🔍 Get accurate answers** with source citations
        - **📊 Evaluate system performance** with built-in testing

        **Getting Started:**
        1. Make sure your OpenAI API key is set in the environment
        2. Click "🚀 Initialize System" in the sidebar
        3. Start asking questions!
        """)

        # System requirements check
        st.markdown("### 🔧 System Requirements")
        if check_api_key():
            st.success("✅ Google Gemini API Key configured")
        else:
            st.error("❌ Google Gemini API Key missing")

        # Show available documents
        data_dir = "data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith(('.pdf', '.txt'))]
            if files:
                st.markdown(f"### 📚 Available Documents ({len(files)} files)")
                for file in files:
                    st.info(f"📄 {file}")
            else:
                st.warning("⚠️ No documents found in data/ directory")
    else:
        # Main Q&A interface
        st.markdown("### 💬 Ask a Question")

        # Question input
        question = st.text_input(
            "Enter your question about the uploaded materials:",
            placeholder="e.g., What is time complexity?",
            key="question_input"
        )

        # Ask button
        if st.button("🔍 Ask", type="primary") or question:
            if question and st.session_state.rag_pipeline:
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.rag_pipeline.query(question)

                        # Store in chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result["answer"],
                            "sources": result["sources"]
                        })

                        # Display answer
                        st.markdown(f'<div class="question-box"><strong>❓ Question:</strong> {question}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="answer-box"><strong>💡 Answer:</strong><br>{result["answer"]}</div>', unsafe_allow_html=True)

                        # Display sources
                        if result["sources"]:
                            st.markdown(f"### 📚 Sources ({result['num_sources']})")
                            for i, source in enumerate(result["sources"]):
                                with st.expander(f"📄 Source {i+1}: {source['source']}"):
                                    st.markdown(f"**File:** {source['source']}")
                                    st.markdown(f"**Chunk ID:** {source['chunk_id']}")
                                    st.markdown(f"**Content Preview:** {source['content']}")
                        else:
                            st.warning("⚠️ No sources found for this answer")

                        # Show error if any
                        if result.get("error"):
                            st.error(f"⚠️ Error: {result['error']}")

                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")

        # Chat history
        if st.session_state.chat_history:
            st.markdown("### 💭 Previous Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 questions
                with st.expander(f"Q: {chat['question'][:50]}..."):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
                    st.markdown(f"**Sources:** {len(chat['sources'])}")

if __name__ == "__main__":
    main()