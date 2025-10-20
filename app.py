import streamlit as st
import os
import tempfile
import time
from typing import List, Dict
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor
from feedback_storage import FeedbackStorage
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
        color: #1a1a1a;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #ff6b6b;
    }
    .answer-box {
        background-color: #e8f4fd;
        color: #1a1a1a;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .source-box {
        background-color: #fff5e6;
        color: #1a1a1a;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #ff9500;
    }
    /* Ensure all text in these boxes is dark */
    .question-box *, .answer-box *, .source-box * {
        color: #1a1a1a !important;
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
    if 'feedback_storage' not in st.session_state:
        st.session_state.feedback_storage = FeedbackStorage()
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = set()  # Track which questions have been rated
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'auto_init_done' not in st.session_state:
        st.session_state.auto_init_done = False
    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = None  # Store last Q&A to persist across reruns
    
    # Auto-initialize the system on first load
    if not st.session_state.auto_init_done and not st.session_state.vector_store_initialized:
        auto_initialize_system()

def auto_initialize_system():
    """Auto-initialize the system on startup."""
    st.session_state.auto_init_done = True
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return  # Don't auto-initialize if no API key
    
    # Check if data directory exists and has files
    data_dir = "data"
    if not os.path.exists(data_dir):
        return
    
    files = [f for f in os.listdir(data_dir) if f.endswith(('.pdf', '.txt'))]
    if not files:
        return  # No documents to process
    
    try:
        # Initialize RAG pipeline
        st.session_state.rag_pipeline = RAGPipeline(api_key)
        
        # Try to load existing vector store
        st.session_state.rag_pipeline.vector_store = st.session_state.rag_pipeline.load_vector_store()
        
        if st.session_state.rag_pipeline.vector_store is not None:
            # Check if vector store has data
            try:
                collection = st.session_state.rag_pipeline.vector_store._collection
                if collection.count() > 0:
                    st.session_state.vector_store_initialized = True
                    return  # Successfully loaded existing vector store
            except:
                pass
        
        # If no existing vector store or it's empty, create new one
        doc_processor = DocumentProcessor()
        documents = doc_processor.process_directory(data_dir)
        
        if documents:
            st.session_state.rag_pipeline.initialize_vector_store(documents)
            st.session_state.vector_store_initialized = True
    except Exception as e:
        # Silently fail - user can manually initialize if needed
        pass

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
    st.sidebar.markdown("## ⚙️ System Status")

    # Show system status
    if st.session_state.vector_store_initialized and st.session_state.rag_pipeline:
        stats = st.session_state.rag_pipeline.get_vector_store_stats()
        st.sidebar.success("✅ System Ready")
        st.sidebar.info(f"📄 Documents: {stats['total_documents']}")
        st.sidebar.info(f"🗂️ Collection: {stats['collection_name']}")
        
        # Manual re-initialization option (collapsed by default)
        with st.sidebar.expander("🔄 Advanced Options"):
            if st.button("🚀 Re-initialize System", key="manual_reinit"):
                if initialize_vector_store():
                    st.rerun()
    else:
        st.sidebar.warning("⚠️ Initializing...")
        st.sidebar.info("Loading vector store and documents...")
        # Manual initialization button for troubleshooting
        if st.sidebar.button("🚀 Try Manual Initialize", type="primary"):
            if initialize_vector_store():
                st.rerun()

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

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["💬 Q&A System", "📊 Monitoring Dashboard"])
    
    with tab1:
        # Main content area
        if not st.session_state.vector_store_initialized:
            # Welcome/Loading screen
            st.markdown("### 🎯 Welcome to the CS Textbooks RAG System!")
            st.markdown("""
            This system allows you to:

            - **📖 Upload and process** computer science textbooks (PDF/TXT)
            - **❓ Ask questions** about the uploaded materials
            - **🔍 Get accurate answers** with source citations
            - **📊 Evaluate system performance** with built-in testing

            **System is initializing...**
            - Loading existing vector database
            - If no database exists, processing documents from `data/` folder
            - This may take a moment on first run
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
            if st.button("🔍 Ask", type="primary"):
                if question and st.session_state.rag_pipeline:
                    with st.spinner("Thinking..."):
                        try:
                            start_time = time.time()
                            result = st.session_state.rag_pipeline.query(question)
                            response_time = time.time() - start_time

                            # Log interaction
                            st.session_state.feedback_storage.add_interaction(
                                question=question,
                                answer_length=len(result["answer"].split()),
                                sources_count=result["num_sources"],
                                response_time=response_time
                            )

                            # Store in chat history and last_answer
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": result["answer"],
                                "sources": result["sources"],
                                "timestamp": time.time()
                            })
                            
                            # Store last answer to persist across reruns
                            st.session_state.last_answer = {
                                "question": question,
                                "answer": result["answer"],
                                "sources": result["sources"],
                                "num_sources": result["num_sources"]
                            }

                        except Exception as e:
                            st.error(f"❌ An error occurred: {e}")
            
            # Display last answer (persists across reruns)
            if st.session_state.last_answer:
                answer_data = st.session_state.last_answer
                
                # Display answer with proper styling
                st.markdown(f'<div class="question-box"><strong>❓ Question:</strong> {answer_data["question"]}</div>', unsafe_allow_html=True)
                
                # Use a container with the answer-box class for styling
                answer_container = st.container()
                with answer_container:
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown("**💡 Answer:**")
                    st.markdown(answer_data["answer"])
                    st.markdown('</div>', unsafe_allow_html=True)

                # Feedback buttons
                st.markdown("#### 📊 Was this answer helpful?")
                feedback_key = f"{answer_data['question']}_{answer_data['answer'][:50]}"
                
                # Check if feedback already given for this question-answer pair
                if feedback_key not in st.session_state.feedback_given:
                    col1, col2, col3 = st.columns([1, 1, 8])
                    with col1:
                        if st.button("👍 Yes", key=f"pos_{feedback_key}"):
                            st.session_state.feedback_storage.add_feedback(
                                question=answer_data["question"],
                                answer=answer_data["answer"],
                                feedback_type="positive",
                                sources_count=answer_data["num_sources"]
                            )
                            st.session_state.feedback_given.add(feedback_key)
                            st.rerun()  # Rerun to hide buttons
                    
                    with col2:
                        if st.button("👎 No", key=f"neg_{feedback_key}"):
                            st.session_state.feedback_storage.add_feedback(
                                question=answer_data["question"],
                                answer=answer_data["answer"],
                                feedback_type="negative",
                                sources_count=answer_data["num_sources"]
                            )
                            st.session_state.feedback_given.add(feedback_key)
                            st.rerun()  # Rerun to hide buttons
                else:
                    st.success("✅ Thank you for your feedback!")

                # Display sources
                if answer_data["sources"]:
                    st.markdown(f"### 📚 Sources ({answer_data['num_sources']})")
                    for i, source in enumerate(answer_data["sources"]):
                        with st.expander(f"📄 Source {i+1}: {source['source']}"):
                            st.markdown(f"**File:** {source['source']}")
                            st.markdown(f"**Chunk ID:** {source['chunk_id']}")
                            st.markdown(f"**Content Preview:** {source['content']}")
                else:
                    st.warning("⚠️ No sources found for this answer")

            # Chat history
            if st.session_state.chat_history:
                st.markdown("### 💭 Previous Questions")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 questions
                    with st.expander(f"Q: {chat['question'][:50]}..."):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown(f"**Answer:** {chat['answer']}")
                        st.markdown(f"**Sources:** {len(chat['sources'])}")
    
    with tab2:
        # Monitoring Dashboard
        show_monitoring_dashboard()

def show_monitoring_dashboard():
    """Display the monitoring dashboard with charts and metrics."""
    st.markdown("## 📊 System Monitoring & Analytics")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Real-time performance metrics and user feedback analytics")
    with col2:
        if st.button("🔄 Refresh Data", key="refresh_dashboard"):
            st.rerun()
    
    # Get fresh data from file
    feedback_df = st.session_state.feedback_storage.get_feedback_dataframe()
    interactions_df = st.session_state.feedback_storage.get_interactions_dataframe()
    feedback_stats = st.session_state.feedback_storage.get_feedback_stats()
    interaction_stats = st.session_state.feedback_storage.get_interaction_stats()
    
    # # Debug info (commented out for production)
    # with st.expander("🔍 Debug Info", expanded=False):
    #     st.write(f"**Feedback records:** {len(feedback_df)}")
    #     st.write(f"**Interaction records:** {len(interactions_df)}")
    #     st.write(f"**File path:** {st.session_state.feedback_storage.feedback_file}")
    #     
    #     if not interactions_df.empty:
    #         st.write("**Sample interaction data:**")
    #         st.dataframe(interactions_df.head(3))
    #     else:
    #         st.warning("No interaction data found. Try asking a question in the Q&A tab first!")
    #     
    #     if not feedback_df.empty:
    #         st.write("**Sample feedback data:**")
    #         st.dataframe(feedback_df.head(3))
    #     else:
    #         st.info("No feedback data yet. Rate some answers with 👍/👎 in the Q&A tab!")
    
    # Display summary metrics
    st.markdown("### 📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Queries",
            value=interaction_stats['total_queries'],
            delta=None
        )
    
    with col2:
        positive_rate = feedback_stats['positive_percentage']
        st.metric(
            label="Positive Feedback",
            value=f"{positive_rate:.1f}%",
            delta=f"{feedback_stats['positive_count']} / {feedback_stats['total_feedback']}"
        )
    
    with col3:
        avg_response = interaction_stats['avg_response_time']
        st.metric(
            label="Avg Response Time",
            value=f"{avg_response:.2f}s",
            delta=None
        )
    
    with col4:
        avg_sources = interaction_stats['avg_sources_count']
        st.metric(
            label="Avg Sources",
            value=f"{avg_sources:.1f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Chart 1 & 2: Feedback Analysis
    st.markdown("### 👍👎 Feedback Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_feedback_chart(feedback_df), width='stretch')
    
    with col2:
        st.plotly_chart(create_feedback_timeline(feedback_df), width='stretch')
    
    st.markdown("---")
    
    # Chart 3 & 4: Performance Metrics
    st.markdown("### ⚡ Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_response_time_chart(interactions_df), width='stretch')
    
    with col2:
        st.plotly_chart(create_query_volume_chart(interactions_df), width='stretch')
    
    st.markdown("---")
    
    # Chart 5 & 6: Content Analysis
    st.markdown("### 📚 Content Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_sources_chart(interactions_df), width='stretch')
    
    with col2:
        st.plotly_chart(create_answer_length_chart(interactions_df), width='stretch')
    
    st.markdown("---")
    
    # Chart 7: Activity Pattern
    st.markdown("### 🕐 Activity Patterns")
    st.plotly_chart(create_hourly_activity_chart(interactions_df), width='stretch')
    
    st.markdown("---")
    
    # Recent Feedback Table
    st.markdown("### 📝 Recent Feedback")
    
    if not feedback_df.empty:
        recent_feedback = feedback_df.tail(10)[['timestamp', 'question', 'feedback', 'sources_count', 'answer_length']].copy()
        recent_feedback['timestamp'] = pd.to_datetime(recent_feedback['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        recent_feedback['feedback'] = recent_feedback['feedback'].map({'positive': '👍 Positive', 'negative': '👎 Negative'})
        recent_feedback.columns = ['Time', 'Question', 'Feedback', 'Sources', 'Answer Length']
        
        st.dataframe(recent_feedback, width='stretch', hide_index=True)
    else:
        st.info("No feedback data available yet. Users need to rate answers in the Q&A tab.")
    
    st.markdown("---")
    
    # Export data option
    st.markdown("### 💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if not feedback_df.empty:
            csv_feedback = feedback_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Feedback Data (CSV)",
                data=csv_feedback,
                file_name=f"feedback_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No feedback data to export")
    
    with col2:
        if not interactions_df.empty:
            csv_interactions = interactions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Interaction Data (CSV)",
                data=csv_interactions,
                file_name=f"interaction_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No interaction data to export")

def create_feedback_chart(feedback_df):
    """Chart 1: Feedback Distribution (Positive vs Negative)"""
    if feedback_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No feedback data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Feedback Distribution", height=300)
        return fig
    
    feedback_counts = feedback_df['feedback'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive 👍', 'Negative 👎'],
        values=[
            feedback_counts.get('positive', 0),
            feedback_counts.get('negative', 0)
        ],
        marker=dict(colors=['#00cc44', '#ff4444']),
        hole=0.4
    )])
    
    fig.update_layout(
        title="User Feedback Distribution",
        height=400,
        showlegend=True
    )
    
    return fig

def create_feedback_timeline(feedback_df):
    """Chart 2: Feedback Over Time"""
    if feedback_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No feedback data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Feedback Timeline", height=300)
        return fig
    
    feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
    feedback_df['date'] = feedback_df['timestamp'].dt.date
    
    daily_feedback = feedback_df.groupby(['date', 'feedback']).size().reset_index(name='count')
    
    fig = px.line(
        daily_feedback,
        x='date',
        y='count',
        color='feedback',
        color_discrete_map={'positive': '#00cc44', 'negative': '#ff4444'},
        title='Feedback Timeline',
        labels={'date': 'Date', 'count': 'Number of Feedback', 'feedback': 'Type'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_response_time_chart(interactions_df):
    """Chart 3: Response Time Distribution"""
    if interactions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No interaction data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Response Time Distribution", height=300)
        return fig
    
    fig = go.Figure(data=[go.Histogram(
        x=interactions_df['response_time_seconds'],
        nbinsx=20,
        marker_color='#1f77b4'
    )])
    
    fig.update_layout(
        title="Response Time Distribution",
        xaxis_title="Response Time (seconds)",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def create_query_volume_chart(interactions_df):
    """Chart 4: Query Volume Over Time"""
    if interactions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No interaction data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Query Volume Over Time", height=300)
        return fig
    
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
    interactions_df['date'] = interactions_df['timestamp'].dt.date
    
    daily_queries = interactions_df.groupby('date').size().reset_index(name='count')
    
    fig = px.bar(
        daily_queries,
        x='date',
        y='count',
        title='Daily Query Volume',
        labels={'date': 'Date', 'count': 'Number of Queries'},
        color_discrete_sequence=['#ff7f0e']
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_sources_chart(interactions_df):
    """Chart 5: Sources Retrieved Distribution"""
    if interactions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No interaction data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Sources Retrieved Distribution", height=300)
        return fig
    
    sources_counts = interactions_df['sources_count'].value_counts().sort_index()
    
    fig = go.Figure(data=[go.Bar(
        x=sources_counts.index,
        y=sources_counts.values,
        marker_color='#2ca02c'
    )])
    
    fig.update_layout(
        title="Sources Retrieved per Query",
        xaxis_title="Number of Sources",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def create_answer_length_chart(interactions_df):
    """Chart 6: Answer Length Distribution"""
    if interactions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No interaction data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Answer Length Distribution", height=300)
        return fig
    
    fig = go.Figure(data=[go.Box(
        y=interactions_df['answer_length'],
        marker_color='#9467bd',
        name='Answer Length (words)'
    )])
    
    fig.update_layout(
        title="Answer Length Distribution",
        yaxis_title="Words",
        height=400
    )
    
    return fig

def create_hourly_activity_chart(interactions_df):
    """Chart 7: Hourly Activity Pattern"""
    if interactions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No interaction data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Hourly Activity Pattern", height=300)
        return fig
    
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
    interactions_df['hour'] = interactions_df['timestamp'].dt.hour
    
    hourly_queries = interactions_df.groupby('hour').size().reset_index(name='count')
    
    fig = px.bar(
        hourly_queries,
        x='hour',
        y='count',
        title='Query Activity by Hour of Day',
        labels={'hour': 'Hour (24h format)', 'count': 'Number of Queries'},
        color='count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400)
    
    return fig

if __name__ == "__main__":
    main()