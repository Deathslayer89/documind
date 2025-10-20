# 📚 DocuMind: Intelligent Document Q&A System

An end-to-end RAG (Retrieval-Augmented Generation) application for querying technical documents and textbooks. Built with Google Gemini, LangChain, ChromaDB, and Streamlit for the **DataTalksClub LLM Zoomcamp**.

## 🎯 Problem Description

Technical professionals and students struggle with information overload when searching through hundreds of pages of documentation and textbooks. **DocuMind** solves this by:
- Creating a searchable AI-powered knowledge base from PDF documents and text files
- Using semantic search to find relevant information instantly
- Providing accurate, context-aware answers with source citations

## ✨ Features

- 🔍 **Intelligent Retrieval**: Semantic search using Google's text-embedding-004 model
- 🤖 **AI Answers**: Google Gemini Pro generates comprehensive responses with source attribution
- 📊 **Monitoring Dashboard**: 7 interactive charts tracking usage, feedback, and performance
- 🎨 **Modern UI**: Streamlit interface with Q&A and dashboard tabs
- 🐳 **Docker Ready**: Full containerization with docker-compose
- 📈 **Optimized**: Evaluated 4 retrieval strategies and 4 prompt templates to select best approaches

## 🏗️ Architecture

```
┌─────────────┐
│  Streamlit  │  User Interface
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LangChain  │  RAG Pipeline
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
┌────────┐ ┌──────┐
│ChromaDB│ │Gemini│
│ Vector │ │ LLM  │
└────────┘ └──────┘
```

## 🚀 Quick Start

### Option 1: Local Setup

**Prerequisites:** Python 3.12+ and Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

```bash
# 1. Clone and navigate
git clone <your-repo-url>
cd rag_project

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 5. Add your documents
mkdir -p data
# Copy your PDFs/text files to data/ folder

# 6. Run the app
streamlit run app.py
```

Access at http://localhost:8501

### Option 2: Docker Deployment

```bash
# 1. Create .env file with your API key
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 2. Build and run
docker-compose up -d

# 3. Access at http://localhost:8501

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Full cleanup (removes volumes)
docker-compose down -v
```

**Docker Features:**
- ✅ Full containerization with data persistence
- ✅ Named volumes for vector DB and feedback data
- ✅ Resource limits (2 CPU cores, 4GB RAM)
- ✅ Health checks and security (non-root user)

### Optional: LangSmith Tracing

For debugging and monitoring, add to `.env`:
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=documind-rag
```
Get API key from: https://smith.langchain.com/

## 📖 Usage

### Main Application

1. **Auto-Initialization**: System automatically loads on startup
   - Detects existing vector database or processes documents from `data/` folder
   - First-time processing may take a few minutes

2. **Ask Questions**:
   - Type questions like "What is the Floyd-Warshall algorithm?"
   - Get AI-generated answers with source citations
   - View retrieved document chunks

3. **Provide Feedback**:
   - Click 👍 or 👎 after each answer
   - Feedback is logged for analytics

4. **Monitor Performance**:
   - Switch to "📊 Monitoring Dashboard" tab
   - View 7 charts: feedback distribution, response times, query volume, etc.
   - Export data as CSV

### Loading New Documents

```bash
# Add files to data/ folder
cp new_document.pdf data/

# Remove old vector store
rm -rf chroma_db

# Restart app (it will auto-reinitialize)
streamlit run app.py
```

### Running Evaluations

**Retrieval Evaluation** (compare 4 search strategies):
```bash
python retrieval_evaluation.py
# Results saved to retrieval_evaluation_results.json
```

**LLM Evaluation** (compare 4 prompt templates):
```bash
python llm_evaluation.py
# Results saved to llm_evaluation_results.json
```

## 🔬 Evaluation Results

### Retrieval Optimization

Tested 4 approaches on 5 diverse queries:

| Approach | Precision | Keyword Score | Combined |
|----------|-----------|---------------|----------|
| **Semantic (k=3)** ⭐ | **86.67%** | **2.53** | **1.37** |
| Semantic (k=5) | 84.00% | 2.36 | 1.30 |
| Semantic (k=10) | 84.00% | 2.16 | 1.24 |
| MMR (k=5) | 84.00% | 1.92 | 1.16 |

**Selected**: Semantic search with k=3 for highest precision and relevance.

**Scripts**: [`retrieval_evaluation.py`](retrieval_evaluation.py) | [`retrieval_evaluation_results.json`](retrieval_evaluation_results.json)

### LLM Prompt Optimization

Tested 4 prompt templates on 5 queries (including unknown topics):

| Prompt Template | Quality Score | Avg Words | Notes |
|----------------|---------------|-----------|-------|
| **Expert Technical** ⭐ | **8.00/10** | **701** | Comprehensive, technical |
| Detailed Context | 7.90/10 | 354 | Balanced detail |
| Structured | 7.00/10 | 129 | Organized, moderate |
| Concise | 6.20/10 | 50 | Brief, minimal |

**Selected**: Expert Technical Style for highest quality and proper handling of unknowns.

**Scripts**: [`llm_evaluation.py`](llm_evaluation.py) | [`llm_evaluation_results.json`](llm_evaluation_results.json)

### Monitoring Dashboard

Integrated dashboard with 7 interactive Plotly charts:

1. **Feedback Distribution** - Pie chart of positive/negative feedback
2. **Feedback Timeline** - Trend analysis over time
3. **Response Time** - Histogram of query processing speeds
4. **Query Volume** - Daily usage bar chart
5. **Sources Retrieved** - Distribution of retrieved documents
6. **Answer Length** - Box plot of response sizes
7. **Hourly Activity** - Heatmap of peak usage times

**Features**:
- User feedback collection (👍/👎 buttons)
- Key metrics: total queries, positive %, avg response time, avg sources
- CSV data export for external analysis

## 📁 Project Structure

```
rag_project/
├── app.py                              # Streamlit UI (integrated dashboard)
├── rag_pipeline.py                     # Core RAG implementation
├── document_processor.py               # PDF/text processing
├── feedback_storage.py                 # User feedback & interaction logging
├── retrieval_evaluation.py             # Retrieval evaluation script
├── llm_evaluation.py                   # LLM prompt evaluation script
├── requirements.txt                    # Python dependencies (178 packages)
├── .env                                # Environment configuration
├── Dockerfile                          # Container definition
├── docker-compose.yml                  # Container orchestration
├── data/                               # Document storage (your PDFs/TXTs)
├── chroma_db/                          # Vector database (auto-generated)
└── feedback_data.json                  # Analytics data (auto-generated)
```

## 🎓 DataTalksClub Project Implementation

### Complete RAG Pipeline
- **Document Ingestion**: Automated processing of PDFs and text files with batch handling (5000 chunks/batch)
- **Text Processing**: Intelligent chunking (1000 chars, 200 overlap)
- **Embeddings**: Google text-embedding-004 for semantic search
- **Vector Store**: ChromaDB with persistence
- **Retrieval**: Optimized semantic search (k=3)
- **Generation**: Google Gemini Pro with expert technical prompt
- **Sources**: Always cited with expandable previews

### Interface
- Streamlit web UI with two-tab design (Q&A + Dashboard)
- Real-time processing, chat history, source explorer
- Auto-initialization and persistence across refreshes

### Ingestion Pipeline
- Automated Python script with error handling
- Auto-detects existing vector store or creates new one
- Batch processing to avoid API limits
- Supports PDF and TXT formats

### Reproducibility
- Clear setup instructions for local and Docker deployment
- All dependencies pinned in requirements.txt
- Sample documents included
- Works on macOS, Linux, and Windows

### Containerization
- Complete docker-compose.yml with all services
- Named volumes for data persistence
- Health checks, resource limits, security hardening
- Production-ready configuration

## 🛠️ Technologies

- **LLM**: Google Gemini Pro
- **Embeddings**: text-embedding-004
- **Vector DB**: ChromaDB
- **Framework**: LangChain
- **Interface**: Streamlit
- **Containerization**: Docker & Docker Compose
- **Analytics**: Plotly for visualizations
- **Language**: Python 3.12

## 📊 System Performance

- **Processing Speed**: ~1,400 chunks/minute
- **Retrieval Speed**: < 1 second
- **Batch Capacity**: 5,000 docs/batch, 15,000+ total
- **Formats**: PDF, TXT
- **Sample Dataset**: 10+ technical books, 15,354 chunks

## 🔮 Future Enhancements

- Hybrid search combining vector and keyword approaches
- Query rewriting for improved retrieval
- Re-ranking with cross-encoder
- Multi-modal support for images and tables
- Cloud deployment (AWS/GCP/Azure)
- FastAPI REST API
- Multi-language document support
- Conversational memory for follow-up questions

---

**Built for DataTalksClub LLM Zoomcamp** | [Report Issues](https://github.com/yourusername/documind/issues)
