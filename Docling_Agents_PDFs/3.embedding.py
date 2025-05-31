from typing import List

import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from google import genai
import os
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

load_dotenv()

# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  

# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------

converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")

# --------------------------------------------------------------
# Apply hybrid chunking with HuggingFace tokenizer
# --------------------------------------------------------------

# Choose a tokenizer - here are some good options:
# For general purpose: "sentence-transformers/all-MiniLM-L6-v2"
# For more advanced: "microsoft/DialoGPT-medium" 
# For multilingual: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 8191

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_MODEL),
    max_tokens=MAX_TOKENS,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

# --------------------------------------------------------------
# Create a LanceDB database and table
# --------------------------------------------------------------

# Create a LanceDB database
db = lancedb.connect("data/lancedb")

# Get the Gemini AI embedding function
func = get_registry().get("gemini-text").create()

# Define a simplified metadata schema
class ChunkMetadata(LanceModel):
    """
    You must order the fields in alphabetical order.
    This is a requirement of the Pydantic implementation.
    """

    filename: str | None
    page_numbers: List[int] | None
    title: str | None


# Define the main Schema
class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata

table = db.create_table("docling", schema=Chunks, mode="overwrite")

# --------------------------------------------------------------
# Prepare the chunks for the table
# --------------------------------------------------------------

processed_chunks = [
    {
        "text": chunk.text,
        "metadata": {
            "filename": chunk.meta.origin.filename,
            "page_numbers": [
                page_no for page_no in sorted(set(prov.page_no for item in chunk.meta.doc_items for prov in item.prov))
            ] or None,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None
        }
    }
    for chunk in chunks
]

# --------------------------------------------------------------
# Add the chunks to the table (automatically embeds the text)
# --------------------------------------------------------------

table.add(processed_chunks)

# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

print(table.to_pandas())
# print(table.count_rows())