from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from google import genai
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Test the client connection
# response = client.models.generate_content(
#     model="gemini-2.0-flash", contents="Explain how AI works in a few words"
# )
# print(response.text)

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
# print(f"Number of chunks: {len(chunks)}")
# print(chunks)

# Print first few chunks to verify
# for i, chunk in enumerate(chunks[:3]):
#     print(f"\n=== Chunk {i+1} ===")
#     print(f"Text: {chunk.text[:200]}...")
#     print(f"Metadata: {chunk.meta}")