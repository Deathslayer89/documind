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
- 📊 **Advanced Monitoring**: 
  - **LangSmith Tracing**: Real-time LLM call tracking, token usage, latency analysis
  - **Grafana Dashboards**: Visual metrics for queries, response times, feedback
  - **ChromaDB Admin UI**: Vector database inspection and management
  - **Integrated Dashboard**: 7 Plotly charts for user feedback and performance
- 🎨 **Modern UI**: Streamlit interface with Q&A and dashboard tabs
- 🐳 **Docker Ready**: Full containerization with docker-compose (4 services)
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

### Option 2: Docker Deployment (Full Monitoring Stack)

```bash
# 1. Create .env file with your API keys
cat > .env << EOF
GOOGLE_API_KEY=your_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=documind-rag
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin
EOF

# 2. Build and run all services
docker-compose up -d

# 3. Access services:
# Main App:        http://localhost:8501
# Grafana:         http://localhost:3000
# ChromaDB UI:     http://localhost:8000
# Prometheus:      http://localhost:9090
# LangSmith:       https://smith.langchain.com

# View logs
docker-compose logs -f rag-app

# Stop all services
docker-compose down

# Full cleanup (removes volumes)
docker-compose down -v
```

**Docker Stack (4 Services):**
- ✅ **Main RAG App**: Streamlit UI on port 8501
- ✅ **Grafana**: Metrics visualization on port 3000
- ✅ **Prometheus**: Metrics collection on port 9090
- ✅ **ChromaDB Admin**: Vector DB UI on port 8000
- ✅ Named volumes for data persistence
- ✅ Resource limits (2 CPU cores, 4GB RAM)
- ✅ Health checks and security (non-root user)

### Option 3: Local Monitoring Stack

Run monitoring tools locally alongside your main app:

**1. LangSmith Tracing (LLM Monitoring)**

```bash
# Add to .env file
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=documind-rag

# Run your app - tracing is automatic
streamlit run app.py
```
View traces at: https://smith.langchain.com/projects

**2. Monitoring Stack with Docker (Recommended)**

Run Grafana and Prometheus for monitoring:

```bash
# Start monitoring stack (Grafana + Prometheus + ChromaDB Server)
docker-compose -f docker-compose-monitoring.yml up -d

# Access services:
# Grafana:            http://localhost:3000 (admin/admin)
# Prometheus:         http://localhost:9090
# ChromaDB Server:    http://localhost:8000

# Stop monitoring stack
docker-compose -f docker-compose-monitoring.yml down
```

**3. ChromaDB Viewer (Custom UI)**

For inspecting your vector database:

```bash
# Run alongside your app (separate terminal)
streamlit run chromadb_viewer.py --server.port 8001
```

Access at: http://localhost:8001

**Features:**
- 🔍 Search vector database with similarity scores
- 📄 Browse documents with full metadata
- 📊 Collection statistics and analytics
- 📥 Export data as CSV
- 🗂️ Multi-collection support

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

## 📊 Monitoring & Observability

### Overview: 4-Layer Monitoring Stack

This project implements comprehensive monitoring at multiple levels:

```
┌─────────────────────────────────────────────┐
│  1. LangSmith → LLM call tracing            │ (Primary)
│  2. Grafana → System metrics visualization  │
│  3. ChromaDB Admin → Vector DB inspection   │
│  4. Streamlit → User feedback analytics     │
└─────────────────────────────────────────────┘
```

### 1. LangSmith Tracing (Primary LLM Monitoring) ⭐

**Production-grade LLM observability** - the most important monitoring tool for RAG systems:

**What it monitors:**
- 🔍 **Request Tracing**: Every LLM call with full inputs/outputs
- ⏱️ **Latency Breakdown**: Embedding → Retrieval → Generation timing
- 💰 **Cost Monitoring**: Token usage (input/output) and API costs per query
- 🐛 **Debugging**: Detailed chain execution with intermediate steps
- 📈 **Analytics**: Query patterns, failure rates, performance trends
- 🧪 **A/B Testing**: Compare different prompts and retrieval strategies

**How to enable:**
```bash
# Add to .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_api_key_here
LANGCHAIN_PROJECT=documind-rag
```

**Access:** https://smith.langchain.com/projects

**Why it's essential:** Unlike basic metrics, LangSmith shows *why* your RAG system behaves the way it does - what context was retrieved, how the LLM interpreted it, and where bottlenecks occur.

### 2. Grafana Dashboards (System Metrics)

Visual monitoring with Prometheus + Grafana stack (auto-configured in Docker):

**Pre-built dashboard includes:**
- Query rate and volume trends
- Response time percentiles (p50, p95)
- Feedback distribution (positive/negative)
- Sources retrieved per query
- System health metrics

**Access:** http://localhost:3000 (default: admin/admin)

**Use case:** High-level system health and performance trends over time.

### 3. ChromaDB Viewer UI (Vector Database)

Custom Streamlit interface to inspect and manage your vector embeddings:
- View collections and document counts (15,354 chunks)
- **Search with similarity scores**: Query the vector DB directly
- Browse documents with metadata (source, page, content)
- View statistics: document lengths, source distribution
- Export data as CSV for analysis
- Debug retrieval issues (why certain docs are/aren't retrieved)

**Access:** http://localhost:8001 (run `streamlit run chromadb_viewer.py --server.port 8001`)

**Use case:** Debug retrieval problems, understand document distribution, verify embeddings, test queries.

### 4. Integrated Streamlit Dashboard

Built-in analytics with 7 interactive Plotly charts:

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

**Access:** Main app → "📊 Monitoring Dashboard" tab

**Use case:** End-user feedback analysis, understand query patterns, identify improvement areas.

## 📁 Project Structure

```
rag_project/
├── app.py                              # Streamlit UI (integrated dashboard)
├── rag_pipeline.py                     # Core RAG implementation
├── document_processor.py               # PDF/text processing
├── feedback_storage.py                 # User feedback & interaction logging
├── retrieval_evaluation.py             # Retrieval evaluation script
├── llm_evaluation.py                   # LLM prompt evaluation script
├── requirements.txt                    # Python dependencies
├── .env                                # Environment configuration
├── Dockerfile                          # Container definition
├── docker-compose.yml                  # Container orchestration (4 services)
├── prometheus.yml                      # Prometheus configuration
├── grafana-datasources.yml             # Grafana data sources
├── grafana-dashboards.yml              # Grafana dashboard provisioning
├── grafana-dashboard.json              # Pre-built Grafana dashboard
├── data/                               # Document storage (your PDFs/TXTs)
├── chroma_db/                          # Vector database (auto-generated)
└── feedback_data.json                  # Analytics data (auto-generated)
```

## 🛠️ Technologies

- **LLM**: Google Gemini 2.5 Pro
- **Embeddings**: text-embedding-004
- **Vector DB**: ChromaDB (with Admin UI)
- **Framework**: LangChain (with LangSmith tracing)
- **Interface**: Streamlit
- **Monitoring**: Prometheus + Grafana + LangSmith
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

## 📸 Screenshots

![Screenshot 1](images/Screenshot%20from%202025-10-20%2009-22-39.png)

![Screenshot 2](images/Screenshot%20from%202025-10-20%2009-23-01.png)

![Screenshot 3](images/Screenshot%20from%202025-10-20%2009-25-04.png)

![Screenshot 4](images/Screenshot%20from%202025-10-20%2009-27-55.png)

![Screenshot 5](images/Screenshot%20from%202025-10-20%2009-46-42.png)

![Screenshot 6](images/Screenshot%20from%202025-10-20%2011-16-20.png)

---