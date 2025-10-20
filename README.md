# 📚 DocuMind: Intelligent Document Q&A System

An end-to-end RAG (Retrieval-Augmented Generation) application that enables users to ask questions about their technical documents, textbooks, and PDFs. Built with Google Gemini, LangChain, ChromaDB, and Streamlit.

## 🎯 Problem Description

Technical professionals, students, and researchers often struggle with:
- **Information Overload**: Managing and searching through hundreds of pages of technical documentation, textbooks, and research papers
- **Time-Consuming Search**: Manually scanning through multiple documents to find specific information
- **Context Understanding**: Difficulty in getting quick, contextualized answers from large document collections

**DocuMind solves this by**:
- Creating a searchable knowledge base from your PDF documents and text files
- Using AI-powered semantic search to find relevant information instantly
- Providing accurate, context-aware answers with source citations
- Supporting multiple document types (algorithms, programming, NLP, cryptography, etc.)

## ✨ Key Features

### 🔍 Core Capabilities
- **Intelligent Document Processing**: Automatically extracts and chunks text from PDFs and text files
- **Semantic Search**: Uses Google's `text-embedding-004` model for accurate document retrieval
- **AI-Powered Answers**: Leverages Google Gemini Pro for generating comprehensive responses
- **Source Attribution**: Always cites the source documents for transparency and verification
- **Batch Processing**: Handles large document collections efficiently (15,000+ chunks)

### 🎨 User Interface
- **Modern Web Interface**: Built with Streamlit for easy interaction
- **Real-Time Processing**: Upload documents and get instant responses
- **Chat History**: Keeps track of previous questions and answers
- **Document Statistics**: View system status and document counts
- **Source Explorer**: Inspect retrieved document chunks with full content preview

### ⚙️ Technical Highlights
- **Vector Database**: ChromaDB for efficient similarity search
- **Automatic Batching**: Handles API rate limits intelligently
- **Persistent Storage**: Saves vector embeddings for fast reloading
- **Error Handling**: Graceful handling of corrupted or unsupported files
- **Configurable Chunking**: Optimized chunk size (1000) and overlap (200) for context preservation

## 🏗️ Architecture

```
┌─────────────────┐
│   User Input    │
│   (Streamlit)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │
│   (LangChain)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────┐
│ ChromaDB│ │  Gemini  │
│ Vector  │ │   LLM    │
│  Store  │ │          │
└─────────┘ └──────────┘
```

## 📋 Technologies Used

- **LLM**: Google Gemini Pro (via `google-generativeai`)
- **Embeddings**: Google Text-Embedding-004
- **Vector Database**: ChromaDB (persistent storage)
- **Framework**: LangChain for RAG orchestration
- **Interface**: Streamlit (Web UI)
- **Document Processing**: PyPDF2 for PDF text extraction
- **Language**: Python 3.12

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Step 1: Clone the Repository
   ```bash
git clone <your-repo-url>
   cd rag_project
   ```

### Step 2: Create Virtual Environment
   ```bash
python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

### Step 3: Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Configure API Key
Create a `.env` file in the project root:
   ```bash
GOOGLE_API_KEY=your_api_key_here

# Optional configurations
CHROMA_PERSIST_DIRECTORY=./chroma_db
COLLECTION_NAME=cs_textbooks
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Step 5: Prepare Your Documents
Place your PDF files and text documents in the `data/` folder:
```bash
mkdir -p data
# Copy your PDFs and .txt files to the data/ folder
```

### Step 6 (Optional): Enable LangSmith Tracing
To enable LangChain tracing for debugging and monitoring, add to your `.env`:
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=documind-rag
```
Get your LangSmith API key from: https://smith.langchain.com/

---

## 🐳 Docker Deployment (Alternative)

For containerized deployment using Docker:

### Quick Start with Docker Compose

1. **Ensure `.env` file exists** with your API key
2. **Build and run**:
```bash
docker-compose up -d
```

3. **Access the app** at http://localhost:8501

### Docker Commands

```bash
# Build the image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Full cleanup (removes volumes)
docker-compose down -v
```

### Features

- ✅ **Full containerization** - Everything runs in Docker
- ✅ **Data persistence** - Named volumes for vector DB and feedback
- ✅ **Resource limits** - CPU and memory constraints
- ✅ **Health checks** - Automatic monitoring
- ✅ **Security** - Non-root user, read-only mounts
- ✅ **Production-ready** - Optimized for deployment

**See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for complete Docker documentation.**

---

## 📖 Usage Guide

### Starting the Application

1. **Activate Virtual Environment** (if not already activated):
```bash
source venv/bin/activate
```

2. **Run the Streamlit App**:
```bash
streamlit run app.py
```

3. **Open in Browser**:
The app will automatically open at `http://localhost:8501`

The application includes an integrated monitoring dashboard accessible via tabs within the main app.

### Using the Application

#### 1. System Auto-Initialization
The system **automatically initializes** on startup:
- Loads existing vector database if available
- If no database exists, processes all documents in the `data/` folder
- Extracts text and creates semantic chunks
- Generates embeddings (first run may take a few minutes)
- Stores vectors in ChromaDB for future use

**Note:** The system persists across page refreshes. Once initialized, you can refresh the browser without losing your data.

#### 2. Ask Questions
- Type your question in the text input box
- Examples:
  - "What is the Floyd-Warshall algorithm?"
  - "Explain dynamic programming"
  - "How does gradient descent work?"

#### 3. Review Answers
- The system provides:
  - **Comprehensive Answer**: AI-generated response based on retrieved context
  - **Source Citations**: List of source documents with chunk IDs
  - **Content Preview**: Expandable sections showing the actual text used

#### 4. Provide Feedback
- After each answer, use the feedback buttons:
  - **👍 Yes**: Answer was helpful
  - **👎 No**: Answer needs improvement
- Your feedback is automatically logged for monitoring and analytics

#### 5. View Chat History
- Previous questions are shown in the **"💭 Previous Questions"** section
- Click to expand and review past interactions

#### 6. Monitor System Performance
- Click on the **"📊 Monitoring Dashboard"** tab in the main app
- View real-time analytics including:
  - User feedback distribution (pie chart)
  - Feedback timeline (line chart)
  - Response time metrics (histogram)
  - Query volume trends (bar chart)
  - Sources retrieved statistics (bar chart)
  - Answer length analysis (box plot)
  - Hourly activity patterns (heatmap)
  - Recent feedback table
  - Data export functionality (CSV downloads)

### Loading New Documents

#### Method 1: Using the Data Folder
1. Add new PDFs or text files to the `data/` folder
2. Delete the existing vector store:
```bash
rm -rf chroma_db
```
3. Click **"🚀 Initialize System"** again to reprocess all documents

#### Method 2: Programmatic Upload
```python
from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline

# Initialize
processor = DocumentProcessor()
rag = RAGPipeline(google_api_key="your_key")

# Process new documents
documents = processor.process_directory("path/to/new/docs")
rag.vector_store.add_documents(documents)
```

### Running Evaluations

#### Retrieval Evaluation

To reproduce the retrieval evaluation and compare different approaches:

```bash
python retrieval_evaluation.py
```

This will:
- Test 4 different retrieval approaches (semantic search with k=3, k=5, k=10, and MMR search)
- Evaluate on 5 test queries
- Generate metrics (precision, keyword relevance)
- Save results to `retrieval_evaluation_results.json`
- Display the best performing approach

#### LLM Evaluation

To reproduce the LLM prompt evaluation:

```bash
python llm_evaluation.py
```

This will:
- Test 4 different prompt templates (Expert Technical, Detailed Context, Structured, Concise)
- Evaluate on 5 test queries (including out-of-scope questions)
- Measure quality scores based on accuracy, honesty, technical depth, and clarity
- Save results to `llm_evaluation_results.json`
- Display the best performing prompt template

## 📊 System Performance

### Current Capabilities
- **Document Processing**: ~1,400 chunks per minute
- **Retrieval Speed**: < 1 second for semantic search
- **Batch Size**: 5,000 documents per batch (handles 15,000+ total)
- **Supported Formats**: PDF, TXT
- **Context Window**: 1,000 tokens per chunk with 200-token overlap

### Sample Dataset Statistics
- **Documents Processed**: 10+ technical books
- **Total Chunks**: 15,354 semantic chunks
- **Topics Covered**: Algorithms, Deep Learning, NLP, Cryptography, Python, Spring Framework

## 🔬 Evaluation & Performance

### Retrieval Evaluation

We evaluated **4 different retrieval approaches** to find the optimal configuration:

| Approach | Precision | Keyword Score | Combined Score |
|----------|-----------|---------------|----------------|
| **Semantic Search (k=3)** ⭐ | **86.67%** | **2.53** | **1.3667** |
| Semantic Search (k=5) | 84.00% | 2.36 | 1.2960 |
| Semantic Search (k=10) | 84.00% | 2.16 | 1.2360 |
| MMR Search (k=5) | 84.00% | 1.92 | 1.1640 |

**Evaluation Methodology:**
- Tested on 5 diverse queries across different topics (algorithms, ML, programming)
- Measured precision (relevant docs / total retrieved)
- Measured keyword relevance (expected keywords found in context)
- Combined score: 70% precision + 30% keyword relevance

**Best Approach Selected:** Semantic Search with k=3
- ✅ **Highest Precision**: 86.67% (best relevance)
- ✅ **Best Keyword Score**: 2.53 (most relevant content)
- ✅ **Optimal Performance**: Returns focused, highly relevant results

**Current Implementation:**
- **Search Method**: Vector similarity search using text-embedding-004
- **Top-K Retrieval**: Returns top 3 most similar chunks (optimized from evaluation)
- **Source Verification**: Always provides source document references

### LLM Evaluation

We evaluated **4 different prompt templates** to optimize answer quality:

| Prompt Template | Quality Score | Avg Word Count | Characteristics |
|----------------|---------------|----------------|-----------------|
| **Expert Technical Style** ⭐ | **8.00/10** | **701 words** | **Comprehensive, technical terminology** |
| Detailed Context-Based | 7.90/10 | 354 words | Helpful, balanced detail |
| Structured Response | 7.00/10 | 129 words | Organized, moderate length |
| Concise Direct | 6.20/10 | 50 words | Brief, minimal detail |

**Evaluation Methodology:**
- Tested on 5 diverse queries (algorithms, ML, cryptography, unknown topics)
- Measured quality score (0-10) based on:
  - Content accuracy and completeness
  - Proper handling of unknown topics (honesty)
  - Technical depth and terminology usage
  - Answer structure and clarity

**Best Prompt Selected:** Expert Technical Style
- ✅ **Highest Quality Score**: 8.00/10
- ✅ **Comprehensive Responses**: Average 701 words
- ✅ **Technical Depth**: Includes terminology and detailed explanations
- ✅ **Honest About Unknowns**: Perfect score (10/10) on out-of-scope questions

**Current Implementation:**
- **Model**: Google Gemini Pro (stable, production-ready)
- **Temperature**: 0 (deterministic responses)
- **Prompt**: Expert Technical Style template (optimized from evaluation)
- **Hallucination Prevention**: Explicitly states when information is not in context

### Monitoring & Analytics

Our comprehensive monitoring system tracks user interactions and system performance in real-time:

**User Feedback Collection:**
- 👍 **Positive Feedback**: "Was this helpful?" buttons after each answer
- 👎 **Negative Feedback**: Capture issues for improvement
- **Automatic Logging**: All feedback stored with timestamps, question context, and metadata

**Monitoring Dashboard** (7 Charts):

1. **Feedback Distribution** (Pie Chart)
   - Visual breakdown of positive vs negative feedback
   - Shows user satisfaction percentage

2. **Feedback Timeline** (Line Chart)
   - Tracks feedback trends over time
   - Helps identify improvement patterns

3. **Response Time Distribution** (Histogram)
   - Shows query processing speed distribution
   - Helps identify performance bottlenecks

4. **Query Volume Over Time** (Bar Chart)
   - Daily query counts
   - Tracks system usage patterns

5. **Sources Retrieved Distribution** (Bar Chart)
   - How many sources are typically retrieved
   - Validates retrieval strategy effectiveness

6. **Answer Length Distribution** (Box Plot)
   - Statistical distribution of answer lengths
   - Ensures consistent response quality

7. **Hourly Activity Pattern** (Heatmap)
   - Peak usage hours
   - Helps with resource planning

**Key Metrics Dashboard:**
- Total queries processed
- Positive feedback percentage
- Average response time
- Average sources per query

**Data Export:**
- Download all feedback data as CSV
- Export interaction logs for external analysis
- Timestamp-indexed for time-series analysis

### Example Query Results

**Query**: "What is the Floyd-Warshall algorithm?"
- ✅ **Retrieval**: Found 5 relevant chunks from algorithms textbook
- ✅ **Answer Quality**: Detailed explanation with pseudocode and complexity analysis
- ✅ **Sources**: Correctly cited "algorithms_introduction.pdf"

**Query**: "What is reinforcement learning?"
- ✅ **Retrieval**: Found chunks from Deep Learning book (but wrong topic)
- ✅ **Honesty**: System correctly stated "I cannot tell you about reinforcement learning based on the provided context"
- ✅ **No Hallucination**: Did not make up information

## 🐳 Docker Support (Optional)

### Using Docker Compose
```bash
# Build and run
docker-compose up -d

# Access the app at http://localhost:8501
```

### Dockerfile
The project includes a Dockerfile for containerization:
```bash
docker build -t documind .
docker run -p 8501:8501 --env-file .env documind
```

## 📁 Project Structure

```
rag_project/
├── app.py                              # Streamlit web interface (with integrated dashboard)
├── rag_pipeline.py                     # Core RAG implementation
├── document_processor.py               # PDF/text processing
├── feedback_storage.py                 # User feedback management
├── monitoring_dashboard.py             # Standalone dashboard (optional, backup)
├── retrieval_evaluation.py             # Retrieval evaluation script
├── retrieval_evaluation_results.json   # Retrieval evaluation results
├── llm_evaluation.py                   # LLM prompt evaluation script
├── llm_evaluation_results.json         # LLM evaluation results
├── feedback_data.json                  # User feedback & interactions (auto-generated)
├── requirements.txt                    # Python dependencies
├── .env                                # Environment configuration
├── data/                               # Document storage
│   ├── algorithms_introduction.pdf
│   └── sample_cs_notes.txt
├── chroma_db/                          # Vector database (auto-generated)
└── README.md                           # This file
```

## 🎓 DataTalksClub LLM Zoomcamp Project

This project was developed as part of the **DataTalksClub LLM Zoomcamp** course, implementing a complete RAG (Retrieval-Augmented Generation) system for querying computer science textbooks.

### Project Overview

#### Problem & Solution
This system addresses the challenge of efficiently searching and understanding technical content across multiple computer science textbooks. Instead of manually searching through hundreds of pages, users can ask natural language questions and receive accurate, context-aware answers with source citations.

#### Complete RAG Pipeline
The system implements a full end-to-end RAG flow:
- **Document Ingestion**: Automatic processing of PDF and text files from the data directory
- **Text Processing**: Intelligent chunking (1000 chars with 200 char overlap) for context preservation
- **Embedding Generation**: Using Google's text-embedding-004 model for semantic search
- **Vector Storage**: ChromaDB for efficient similarity search with persistence
- **Retrieval**: Semantic search with optimized parameters
- **Generation**: Google Gemini Pro for answer synthesis
- **Response**: Formatted answers with source citations and expandable previews

#### Retrieval Optimization
Multiple retrieval approaches were evaluated to optimize performance:
- **Approaches Tested**: Semantic search (k=3, k=5, k=10) and MMR search (k=5)
- **Evaluation Metrics**: Precision, keyword relevance, and combined scores
- **Test Set**: 5 diverse queries across algorithms, data structures, and ML topics
- **Results**: Semantic search with k=3 achieved best performance (86.67% precision, 2.53/3.0 keyword relevance)
- **Implementation**: Production system uses the optimized k=3 configuration
- **Evaluation Script**: [`retrieval_evaluation.py`](retrieval_evaluation.py) | **Results**: [`retrieval_evaluation_results.json`](retrieval_evaluation_results.json)

#### LLM Configuration Optimization  
Different prompt templates were tested to maximize answer quality:
- **Templates Evaluated**: Expert Technical Style, Detailed Context-Based, Structured Response, Concise Direct
- **Evaluation Criteria**: Accuracy, completeness, technical depth, honesty about unknowns
- **Test Set**: 5 queries including out-of-scope questions to test hallucination prevention
- **Results**: Expert Technical Style achieved highest quality (8.00/10 score, 701-word avg responses)
- **Key Features**: Comprehensive technical explanations, proper terminology, perfect handling of unknown topics
- **Evaluation Script**: [`llm_evaluation.py`](llm_evaluation.py) | **Results**: [`llm_evaluation_results.json`](llm_evaluation_results.json)

#### User Interface
Interactive web application built with Streamlit featuring:
- **Two-Tab Design**: Q&A System and Monitoring Dashboard
- **Question Answering**: Real-time processing with streaming responses
- **Source Citations**: Expandable document previews with chunk IDs
- **User Feedback**: Thumbs up/down buttons for answer quality
- **Chat History**: Recent questions for easy reference
- **Modern UI**: Custom CSS styling with proper contrast and readability
- **Auto-Initialization**: System loads automatically on startup

#### Data Ingestion Pipeline
Automated document processing system:
- **Auto-Initialization**: Detects and loads existing vector database on startup
- **Format Support**: PDF and TXT files with robust error handling
- **Batch Processing**: Handles large document sets (5000 chunks per batch to avoid API limits)
- **Progress Tracking**: Real-time feedback during initialization
- **Persistence**: Vector database survives application restarts
- **Error Handling**: Graceful handling of corrupted or unsupported files

#### Reproducibility
Complete setup with clear documentation:
- **Detailed Instructions**: Step-by-step installation and usage guide
- **Dependency Management**: requirements.txt with pinned versions (178 packages)
- **Environment Configuration**: Simple .env file for API keys and settings
- **Sample Documents**: Included for immediate testing
- **Multiple Deployment Options**: Local (venv) and containerized (Docker)
- **Troubleshooting Guide**: Common issues and solutions documented

#### Monitoring & Analytics
Comprehensive monitoring system with integrated dashboard:
- **User Feedback Collection**: Thumbs up/down after each answer with persistent storage
- **Real-Time Dashboard**: Integrated tab with 7 interactive Plotly charts:
  1. **Feedback Distribution**: Pie chart showing positive vs negative feedback
  2. **Feedback Timeline**: Line chart tracking feedback trends over time
  3. **Response Time Distribution**: Histogram of query processing times
  4. **Query Volume**: Bar chart showing daily usage patterns
  5. **Sources Retrieved**: Distribution of source documents per query
  6. **Answer Length**: Box plot of response sizes
  7. **Hourly Activity**: Heatmap showing peak usage hours
- **Key Metrics Display**: Total queries, positive feedback %, avg response time, avg sources
- **Data Export**: CSV download for feedback and interaction logs
- **Debug Tools**: File path verification and sample data preview

#### Containerization
Production-ready Docker deployment:
- **Dockerfile**: Optimized build with non-root user, health checks, and layer caching
- **Docker Compose**: Complete orchestration with named volumes and resource limits
- **Data Persistence**: Separate volumes for vector DB and feedback data
- **Security**: Non-root user execution, read-only mounts for config and data
- **Resource Management**: Configurable CPU (2 cores) and memory (4GB) limits
- **Health Monitoring**: Automatic health checks every 30 seconds
- **Network Isolation**: Custom bridge network for security
- **Comprehensive Guide**: DOCKER_GUIDE.md with commands, troubleshooting, and best practices

### Additional Features

#### LangChain Integration
- Full LangChain framework implementation for RAG pipeline
- Optional LangSmith tracing for debugging and monitoring
- Configurable via environment variables
- Complete trace visibility for development

#### Performance Optimizations
- Batch embedding generation to handle large document sets
- Vector database persistence and caching
- Efficient chunk processing with overlap
- Response time tracking and logging
- Memory-efficient document handling

### Technology Stack

- **LLM**: Google Gemini Pro (stable, production model)
- **Embeddings**: text-embedding-004 (latest Google embedding model)
- **Vector Database**: ChromaDB with SQLite backend
- **Framework**: LangChain for RAG orchestration
- **Interface**: Streamlit with custom CSS
- **Containerization**: Docker & Docker Compose
- **Language**: Python 3.11+
- **Key Libraries**: langchain, langchain-google-genai, chromadb, streamlit, plotly

## 🔮 Future Enhancements

Potential improvements for future iterations:

- **Hybrid Search**: Combine vector and keyword search for better recall
- **Query Rewriting**: Improve retrieval with automatic query reformulation
- **Re-ranking**: Implement cross-encoder for better result ordering
- **Multi-Modal Support**: Add support for images, tables, and diagrams from PDFs
- **Cloud Deployment**: Deploy to AWS/GCP/Azure with scalable infrastructure
- **API Endpoint**: Create FastAPI REST API for programmatic access
- **Multi-Language Support**: Extend beyond English-language documents
- **Conversational Memory**: Add chat history and follow-up question handling

---
