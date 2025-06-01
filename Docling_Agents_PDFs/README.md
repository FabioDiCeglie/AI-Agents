# Docling AI Document Processing & Q&A System

A comprehensive AI-powered system for extracting, processing, and querying information from PDFs and web documents using the Docling library, vector embeddings, and Google's Gemini AI.

## 🚀 Features

- **Multi-format Document Processing**: Extract and process PDFs and HTML content
- **Intelligent Document Chunking**: Smart document segmentation using HybridChunker
- **Vector Embeddings**: Store and search documents using semantic similarity
- **Interactive Q&A Interface**: Streamlit-powered chat interface for document queries
- **Sitemap Support**: Bulk process websites using sitemap parsing
- **Source Attribution**: Track document origins with page numbers and titles

## 📋 Project Structure

```
Docling_Agents_PDFs/
├── 1.extraction.py     # Document extraction from PDFs and websites
├── 2.chunking.py       # Document chunking with HybridChunker
├── 3.embedding.py      # Vector embeddings and LanceDB storage
├── 4.search.py         # Vector similarity search functionality
├── 5.chat.py          # Streamlit chat interface
├── utils/
│   └── sitemap.py     # Sitemap parsing utilities
├── data/
│   └── lancedb/       # Vector database storage
└── requirements.txt   # Project dependencies
```

## 🛠️ Installation

1. **Clone the repository and navigate to the project directory**

2. **Install uv (if not already installed)**:
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Alternative: using pip
   pip install uv
   ```

3. **Create a virtual environment with uv**:
   ```bash
   uv venv .venv
   ```

4. **Activate the virtual environment**:
   ```bash
   # On macOS/Linux
   source .venv/bin/activate
   
   # On Windows
   .venv\Scripts\activate
   ```

5. **Install dependencies with uv**:
   ```bash
   uv pip install -r requirements.txt
   ```

6. **Set up environment variables**:
   Create a `.env` file in the project root with your API keys:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 🔧 Usage

### Step 1: Document Extraction (`1.extraction.py`)

Extract content from PDFs and websites:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Extract from PDF URL
result = converter.convert("https://arxiv.org/pdf/2408.09869")

# Extract from HTML
result = converter.convert("https://docling-project.github.io/docling/")

# Bulk extract from sitemap
sitemap_urls = get_sitemap_urls("https://docling-project.github.io/docling/")
conv_results_iter = converter.convert_all(sitemap_urls)
```

### Step 2: Document Chunking (`2.chunking.py`)

Break documents into manageable chunks:

```python
from docling.chunking import HybridChunker

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)
```

### Step 3: Vector Embeddings (`3.embedding.py`)

Store documents in a vector database:

```python
import lancedb
from lancedb.embeddings import get_registry

# Create LanceDB database
db = lancedb.connect("data/lancedb")

# Set up Gemini embeddings
func = get_registry().get("gemini-text").create()

# Store processed chunks
table.add(processed_chunks)
```

### Step 4: Search Documents (`4.search.py`)

Query documents using vector similarity:

```python
# Search for relevant content
result = table.search(query="pdf", query_type="vector").limit(3)
```

### Step 5: Interactive Chat (`5.chat.py`)

Launch the Streamlit chat interface:

```bash
streamlit run 5.chat.py
```

## 💬 Chat Interface Features

- **Real-time Document Search**: Automatically finds relevant document sections
- **Source Attribution**: Shows document source, page numbers, and section titles
- **Streaming Responses**: Real-time AI responses using Gemini 2.0 Flash
- **Interactive Context**: Expandable sections showing retrieved document chunks
- **Chat History**: Maintains conversation context

## 🔧 Configuration

### Tokenizer Options

The system supports multiple tokenizer models in `2.chunking.py` and `3.embedding.py`:

- **General purpose**: `sentence-transformers/all-MiniLM-L6-v2`
- **Advanced**: `microsoft/DialoGPT-medium`
- **Multilingual**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### Chunking Parameters

- **MAX_TOKENS**: 8191 (configurable)
- **merge_peers**: True (merges related chunks)

### Search Parameters

- **num_results**: 5 (default number of context chunks)
- **query_type**: "vector" (semantic similarity search)

## 📊 Database Schema

The LanceDB stores chunks with the following structure:

```python
class Chunks(LanceModel):
    text: str                    # Document text content
    vector: Vector               # Embedding vector
    metadata: ChunkMetadata      # Source information

class ChunkMetadata(LanceModel):
    filename: str | None         # Source document name
    page_numbers: List[int] | None   # Page references
    title: str | None           # Section/heading title
```

## 🔍 Example Workflow

1. **Process Documents**: Run `3.embedding.py` to extract, chunk, and create vector embeddings
   ```bash
   python 3.embedding.py
   ```

2. **Start Chat Interface**: Launch the interactive Q&A system
   ```bash
   streamlit run 5.chat.py
   ```

**Optional**: Test search functionality with `4.search.py` before using the chat interface

### Individual Components (for development/testing)

If you want to run components separately:
- **Extract**: Run `1.extraction.py` to process a PDF from arXiv
- **Chunk**: Run `2.chunking.py` to segment the document  
- **Embed**: Run `3.embedding.py` to create and store vector embeddings
- **Search**: Test queries with `4.search.py`

## 🛡️ Requirements

- Python 3.8+
- Google API Key for Gemini AI
- Internet connection for document downloads
- ~2GB disk space for models and embeddings

## 📚 Dependencies

- **docling**: Document processing and conversion
- **lancedb**: Vector database for embeddings
- **streamlit**: Web interface framework
- **google-genai**: Google's Gemini AI API
- **transformers**: HuggingFace tokenizers
- **requests**: HTTP requests for document fetching

## 📧 Support

For issues and questions, please check the documentation of the individual libraries:
- [Docling Documentation](https://docling-project.github.io/docling/)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [Streamlit Documentation](https://docs.streamlit.io/) 