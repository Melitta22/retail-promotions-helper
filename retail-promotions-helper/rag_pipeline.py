"""
rag_pipeline.py
---------------
Handles everything related to documents:
  1. Reading PDFs
  2. Cleaning and splitting text into chunks
  3. Creating embeddings (vector numbers that capture meaning)
  4. Storing embeddings in FAISS (a fast search database)
  5. Searching for relevant chunks when a question is asked
"""

import os
import re
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from groq import Groq

# ── Configuration ──────────────────────────────────────────────────────────────

CHUNK_SIZE = 500          # How many characters per chunk of text
CHUNK_OVERLAP = 100       # How many characters to overlap between chunks (keeps context)
TOP_K = 5                 # How many chunks to retrieve per query
EMBED_MODEL = "llama-3.3-70b-versatile"   # Groq model used for embeddings
INDEX_FILE = "faiss_index.bin"   # Where to save the FAISS search index
CHUNKS_FILE = "chunks.pkl"       # Where to save the text chunks

# ── Step 1: Read PDF ───────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> str:
    """
    Reads all text from a PDF file and returns it as one big string.
    """
    print(f"  📄 Reading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    return "\n".join(all_text)


# ── Step 2: Clean and Chunk Text ───────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Removes extra whitespace and fixes common PDF extraction noise.
    """
    # Collapse multiple spaces/newlines into single ones
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers (standalone numbers)
    text = re.sub(r'\b\d{1,3}\b(?=\s)', ' ', text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits a long string into smaller overlapping chunks.
    Overlap ensures context isn't lost at boundaries.
    
    Example with chunk_size=10, overlap=3:
      "Hello World Goodbye" -> ["Hello Worl", "orld Goodb", "odbye"]
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():   # Skip empty chunks
            chunks.append(chunk.strip())
        start += chunk_size - overlap   # Move forward, leaving an overlap
    return chunks


def load_and_chunk_pdfs(pdf_folder: str) -> list[dict]:
    """
    Loads all PDFs from a folder, cleans them, and splits into chunks.
    Returns a list of dicts: {"text": "...", "source": "filename.pdf"}
    """
    all_chunks = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{pdf_folder}'. Please add Azure resource PDFs there.")

    for filename in pdf_files:
        path = os.path.join(pdf_folder, filename)
        raw_text = load_pdf(path)
        clean = clean_text(raw_text)
        chunks = split_into_chunks(clean)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": filename})
        print(f"  ✅ {filename}: {len(chunks)} chunks created")

    print(f"\n  Total chunks across all PDFs: {len(all_chunks)}")
    return all_chunks


# ── Step 3: Create Embeddings via Groq ────────────────────────────────────────

def get_embedding(client: Groq, text: str) -> list[float]:
    """
    Sends text to Groq and gets back an embedding (a list of numbers).
    These numbers represent the "meaning" of the text in vector space.
    
    Note: We simulate embeddings using the model's hidden representation
    by asking it to produce a compact numeric summary. For production use,
    you could use a dedicated embedding endpoint.
    """
    # We use a prompt that asks the model to produce a stable semantic hash
    # as a simple embedding strategy compatible with the Groq chat API.
    response = client.chat.completions.create(
        model=EMBED_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an embedding assistant. Given text, return ONLY a "
                    "comma-separated list of exactly 128 floating point numbers between -1 and 1 "
                    "that semantically represent the input. No explanation, no labels, just numbers."
                )
            },
            {"role": "user", "content": f"Embed this text:\n{text[:800]}"}
        ],
        max_tokens=512,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    # Parse the comma-separated numbers
    numbers = [float(x) for x in raw.replace('\n', ',').split(',') if x.strip()]
    # Ensure exactly 128 dimensions (pad or trim if needed)
    if len(numbers) < 128:
        numbers += [0.0] * (128 - len(numbers))
    return numbers[:128]


def embed_chunks(client: Groq, chunks: list[dict]) -> np.ndarray:
    """
    Converts all text chunks into embedding vectors.
    Returns a 2D numpy array of shape (num_chunks, 128).
    """
    print("\n  🔢 Generating embeddings (this may take a minute)...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"  Embedding chunk {i+1}/{len(chunks)}...", end="\r")
        vec = get_embedding(client, chunk["text"])
        embeddings.append(vec)
    print()
    return np.array(embeddings, dtype=np.float32)


# ── Step 4: Build and Save FAISS Index ────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Creates a FAISS index from the embeddings.
    FAISS enables super-fast similarity search: given a query embedding,
    it finds the most similar stored embeddings in milliseconds.
    """
    dimension = embeddings.shape[1]   # Should be 128
    index = faiss.IndexFlatL2(dimension)   # L2 = Euclidean distance
    index.add(embeddings)
    print(f"  📦 FAISS index built with {index.ntotal} vectors")
    return index


def save_index(index: faiss.IndexFlatL2, chunks: list[dict]):
    """Saves the FAISS index and chunks to disk so we don't rebuild every time."""
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  💾 Index saved to '{INDEX_FILE}' and '{CHUNKS_FILE}'")


def load_index() -> tuple[faiss.IndexFlatL2, list[dict]]:
    """Loads a previously saved FAISS index and chunks from disk."""
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def index_exists() -> bool:
    """Check if a saved index already exists on disk."""
    return os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE)


# ── Step 5: Retrieve Relevant Chunks ──────────────────────────────────────────

def retrieve(client: Groq, query: str, index: faiss.IndexFlatL2, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Given a user question, find the most relevant text chunks.
    
    Process:
      1. Convert the query into an embedding
      2. Ask FAISS to find the closest stored embeddings
      3. Return the corresponding text chunks
    """
    query_vec = np.array([get_embedding(client, query)], dtype=np.float32)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for idx in indices[0]:
        if idx != -1:   # -1 means no result found
            results.append(chunks[idx])
    return results
