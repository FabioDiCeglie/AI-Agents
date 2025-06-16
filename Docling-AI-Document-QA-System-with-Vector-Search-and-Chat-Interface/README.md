# Docling AI Document Processing & Q&A System

An intelligent document processing pipeline that extracts content from PDFs and websites, creates vector embeddings, and provides an interactive chat interface for document queries using AI.

## What This System Does

🔄 **Complete Document Processing Pipeline:**
1. **Extract** content from PDFs and websites using Docling
2. **Chunk** documents intelligently with HybridChunker
3. **Embed** text into vector database (LanceDB) for semantic search
4. **Search** documents using vector similarity
5. **Chat** with documents via Streamlit interface powered by Google Gemini AI

## Project Structure

```
├── 1.extraction.py      # Extract from PDFs and websites (including sitemaps)
├── 2.chunking.py        # Smart document segmentation  
├── 3.embedding.py       # Vector embeddings + LanceDB storage
├── 4.search.py          # Vector similarity search
├── 5.chat.py           # Streamlit chat interface
├── utils/sitemap.py    # Bulk website processing
└── data/lancedb/       # Vector database storage
```

## Key Features

- **Multi-format Support**: PDFs, HTML, bulk sitemap processing
- **Semantic Search**: Vector embeddings for intelligent document retrieval
- **Interactive Chat**: Streamlit-powered Q&A with source attribution
- **Real-time AI**: Google Gemini 2.0 Flash integration
- **Source Tracking**: Page numbers, titles, and document origins

## Technologies

- **Docling** - Document extraction and conversion
- **LanceDB** - Vector database for embeddings
- **Google Gemini AI** - Language model for responses
- **Streamlit** - Interactive chat interface
- **HuggingFace Transformers** - Text tokenization

## Quick Start

1. **Setup Environment**:
   ```bash
   uv venv .venv && source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. **Add API Key**: Create `.env` with `GOOGLE_API_KEY=your_key`

3. **Process Documents**: `python 3.embedding.py`

4. **Start Chat Interface**: `streamlit run 5.chat.py`

## Use Cases

- **Research**: Query academic papers and documentation
- **Knowledge Base**: Build searchable document collections  
- **Content Analysis**: Extract insights from large document sets
- **Technical Documentation**: Interactive help systems

This system transforms static documents into an intelligent, searchable knowledge base with conversational AI interface. 