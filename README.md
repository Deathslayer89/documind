# 📚 DocuMind: Intelligent Document Q&A System

An end-to-end RAG (Retrieval-Augmented Generation) application for querying technical documents and textbooks. Built with Google Gemini, LangChain, Qdrant, and Streamlit for the **DataTalksClub LLM Zoomcamp**.

## 🎯 Problem Description

Technical professionals and students struggle with information overload when searching through hundreds of pages of documentation and textbooks. **DocuMind** solves this by:
- Creating a searchable AI-powered knowledge base from PDF documents and text files
- Using semantic search to find relevant information instantly
- Providing accurate, context-aware answers with source citations

## ✨ Features

- 🔍 **Intelligent Retrieval**: Semantic search using Google's text-embedding-004 model
- 🤖 **AI Answers**: Google Gemini Pro generates comprehensive responses with source attribution
- 📊 **Production-Grade Monitoring & Observability**: 
  - **LangSmith Tracing**: Complete LLM call tracking with token costs, latency, debugging
  - **Qdrant 3D Vector Visualizations**: Interactive t-SNE/UMAP plots of document embeddings
  - **Grafana + Prometheus**: Professional time-series dashboards for system metrics
  - **Integrated Analytics**: 7 Plotly charts for user feedback and performance trends
- 🎨 **Modern UI**: Streamlit interface with Q&A and dashboard tabs
- 🐳 **Docker Ready**: Full containerization with docker-compose (4 services)
- 📈 **Optimized**: Evaluated 4 retrieval strategies and 4 prompt templates to select best approaches

## 🏗️ Architecture

```
┌─────────────┐
│  Streamlit  │  User Interface + Integrated Dashboard
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LangChain  │  RAG Pipeline (with LangSmith tracing)
└──────┬──────┘
       │
   ┌───┴────┬─────────────┐
   ▼        ▼             ▼
┌────────┐ ┌──────┐  ┌──────────┐
│ Qdrant │ │Gemini│  │Prometheus│
│3D Viz  │ │ LLM  │  │+ Grafana │
└────────┘ └──────┘  └──────────┘
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
# Qdrant UI:       http://localhost:6333/dashboard  (3D visualizations!)
# Grafana:         http://localhost:3000
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
- ✅ **Qdrant**: Vector database with 3D visualization UI on port 6333
- ✅ **Grafana**: Metrics dashboards on port 3000
- ✅ **Prometheus**: Time-series metrics collection on port 9090
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

Run full monitoring stack (Qdrant, Grafana, Prometheus):

```bash
# Start monitoring stack
docker-compose -f docker-compose-monitoring.yml up -d

# Access services:
# Qdrant UI:          http://localhost:6333/dashboard (3D viz!)
# Grafana:            http://localhost:3000 (admin/admin)
# Prometheus:         http://localhost:9090

# Stop monitoring stack
docker-compose -f docker-compose-monitoring.yml down
```

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

# Delete Qdrant collection to force rebuild
curl -X DELETE http://localhost:6333/collections/cs_textbooks

# Restart app (it will auto-reinitialize with new documents)
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
┌──────────────────────────────────────────────────┐
│  1. LangSmith → LLM tracing (token costs, debug) │ ⭐ Primary
│  2. Qdrant UI → 3D vector visualization (t-SNE)  │ 
│  3. Grafana → Time-series dashboards             │
│  4. Prometheus → Metrics collection & queries    │
│  5. Streamlit → User feedback analytics          │
└──────────────────────────────────────────────────┘
```

### 1. LangSmith Tracing (Primary LLM Monitoring) ⭐

**Production-grade LLM observability** - the most critical monitoring tool for RAG systems:

**What it tracks:**
- 🔍 **Full Request Tracing**: Every LLM call with complete inputs/outputs/intermediate steps
- ⏱️ **Latency Breakdown**: Embedding (0.2s) → Retrieval (0.1s) → Generation (2.5s) timing
- 💰 **Cost Monitoring**: Token usage (input: 3.2K, output: 450) and real-time API costs
- 🐛 **Debugging**: Step-by-step chain execution showing retrieval → context → final answer
- 📈 **Analytics Dashboard**: Query patterns, failure rates, performance trends, token efficiency
- 🧪 **A/B Testing**: Compare different prompts, embeddings, and retrieval strategies side-by-side
- 🔔 **Alerts**: Set up notifications for errors, latency spikes, or cost thresholds

**How to enable:**
```bash
# Add to .env file
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_api_key_here  # Get from smith.langchain.com
LANGCHAIN_PROJECT=documind-rag
```

**Access:** https://smith.langchain.com/projects

**Why it's essential:** Unlike basic metrics, LangSmith shows *why* your RAG system behaves the way it does - what context was retrieved, how the LLM interpreted it, where bottlenecks occur, and exactly which prompts/strategies perform best. This is the #1 tool for debugging and optimizing RAG applications.

### 2. Qdrant 3D Vector Visualizations 🎨

**Interactive vector space exploration** with built-in Qdrant UI:

**Features:**
- **3D Scatter Plots**: Visualize 3,801 document embeddings using t-SNE or UMAP dimensionality reduction
- **Color Coding**: Color points by source document, chunk position, or similarity clusters
- **Interactive**: Rotate, zoom, pan through your vector space to see document relationships
- **Filtering**: Focus on specific documents, topics, or similarity ranges
- **Hover Tooltips**: See actual document text and metadata on mouseover

**Access:** http://localhost:6333/dashboard → Collections → `cs_textbooks` → **Visualize tab**

**Example visualizations:**

```json
// View all vectors colored by source document
{
  "limit": 3801,
  "color_by": { "payload": "source" },
  "algorithm": "UMAP"
}

// See document structure (beginning vs end chunks)
{
  "limit": 2000,
  "color_by": { "payload": "chunk_id" },
  "algorithm": "TSNE"
}

// Filter to specific document only
{
  "limit": 1000,
  "filter": {
    "must": [
      { "key": "source", "match": { "text": "Algorithm" } }
    ]
  },
  "algorithm": "UMAP"
}
```

**Use case:** Understand document clustering, verify embedding quality, identify semantic gaps, create impressive visualizations for presentations.

### 3. Grafana + Prometheus Dashboards (System Metrics)

**Professional time-series monitoring** with Prometheus + Grafana stack (auto-configured in Docker):

**Pre-built dashboard includes:**
- Query rate and volume trends over time
- Response time percentiles (p50, p95, p99)
- Memory and CPU usage graphs
- Qdrant collection statistics
- System health indicators

**Access:**
- **Grafana**: http://localhost:3000 (login: admin/admin)
- **Prometheus**: http://localhost:9090

**Useful Prometheus Queries to Try:**

| Query | Description | Panel Type |
|-------|-------------|------------|
| `process_resident_memory_bytes / 1024 / 1024` | Memory usage in MB | Time Series |
| `rate(process_cpu_seconds_total[5m]) * 100` | CPU usage % | Time Series |
| `go_goroutines` | Active threads | Gauge |
| `rate(qdrant_operations_total[1m])` | Qdrant ops/sec | Time Series |
| `histogram_quantile(0.95, qdrant_search_duration_seconds)` | Search latency p95 | Time Series |

**Creating Custom Panels in Grafana:**
1. Go to http://localhost:3000 → Dashboards → New → Add visualization
2. Select "Prometheus" as data source
3. Enter any query from the table above
4. Set panel title, unit, and visualization type
5. Click "Apply" and "Save dashboard"

**Use case:** Monitor system health, track performance trends, identify bottlenecks, create professional dashboards for stakeholders.


### 4. Integrated Streamlit Analytics Dashboard

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
├── rag_pipeline.py                     # Core RAG implementation (Qdrant)
├── document_processor.py               # PDF/text processing (PyMuPDF)
├── feedback_storage.py                 # User feedback & interaction logging
├── retrieval_evaluation.py             # Retrieval evaluation script
├── llm_evaluation.py                   # LLM prompt evaluation script
├── requirements.txt                    # Python dependencies
├── .env                                # Environment configuration
├── Dockerfile                          # Container definition
├── docker-compose.yml                  # Main stack: app + Qdrant + Grafana + Prometheus
├── docker-compose-monitoring.yml       # Standalone monitoring stack
├── prometheus.yml                      # Prometheus configuration
├── grafana-datasources.yml             # Grafana data sources
├── grafana-dashboards.yml              # Grafana dashboard provisioning
├── grafana-dashboard.json              # Pre-built Grafana dashboard
├── data/                               # Document storage (your PDFs/TXTs)
└── feedback_data.json                  # Analytics data (auto-generated)
```

## 🛠️ Technologies

- **LLM**: Google Gemini 2.5 Pro
- **Embeddings**: Google text-embedding-004 (768 dimensions)
- **Vector DB**: Qdrant (with 3D visualization UI)
- **Framework**: LangChain (with LangSmith tracing)
- **Interface**: Streamlit
- **Monitoring**: LangSmith + Prometheus + Grafana
- **Containerization**: Docker & Docker Compose
- **Analytics**: Plotly for charts, t-SNE/UMAP for embeddings
- **PDF Processing**: PyMuPDF (100x faster than PyPDF2)
- **Language**: Python 3.12

## 📊 System Performance

- **PDF Processing**: 2.6 seconds for 1,312 pages (PyMuPDF)
- **Embedding Speed**: ~500 chunks/minute (Google batch API)
- **Retrieval Speed**: < 100ms (Qdrant vector search)
- **LLM Response**: 2-5 seconds (Gemini Pro)
- **Batch Capacity**: 5,000 docs/batch, 15,000+ total
- **Formats**: PDF, TXT
- **Sample Dataset**: Technical textbooks, 3,801 chunks, 768-dim vectors

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

![Qdrant 3D Visualization](images/Screenshot%20from%202025-10-21%2000-24-48.png)
*Qdrant 3D Vector Space: Interactive UMAP visualization of 3,801 document embeddings colored by source*

![Grafana Dashboard](images/Screenshot%20from%202025-10-21%2000-33-23.png)
*Grafana + Prometheus: System metrics dashboard with CPU, memory, and Qdrant performance graphs*

![LangSmith Tracing](images/Screenshot%20from%202025-10-21%2000-40-52.png)
*LangSmith LLM Tracing: Complete request traces showing token usage, latency breakdown, and debugging info*

---