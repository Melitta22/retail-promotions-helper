"""
chat_app.py
-----------
Retail Promotions Bot — Terminal Chat Interface

This is the main file you run. It:
  1. Asks if you want to rebuild the knowledge base (index PDFs) or use existing one
  2. Starts a chat loop where you can ask questions about your Azure resources
  3. Uses RAG: retrieves relevant document chunks, then generates an answer with Llama via Groq

HOW TO RUN:
  python chat_app.py

COMMANDS DURING CHAT:
  Type your question and press Enter
  Type 'quit' or 'exit' to stop
  Type 'rebuild' to re-index your PDFs
"""

import os
import sys
from groq import Groq
from rag_pipeline import (
    load_and_chunk_pdfs,
    embed_chunks,
    build_faiss_index,
    save_index,
    load_index,
    retrieve,
    index_exists,
)

# ── Configuration ──────────────────────────────────────────────────────────────

PDF_FOLDER = "docs"          # Put your Azure resource PDFs in this folder
GROQ_API_KEY_ENV = "GROQ_API_KEY"  # Set this environment variable with your Groq key
CHAT_MODEL = "llama-3.3-70b-versatile"      # Groq model used for answering questions

# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are RetailBot, an AI assistant specializing in retail promotions and marketing offers.

You help retail staff, marketing teams, and store managers understand current and upcoming promotions,
discount rules, eligibility conditions, product inclusions/exclusions, and campaign dates.

Use ONLY the provided promotion documents to answer questions. If the information is not in the 
documents, say so clearly — do not invent offers or discounts.

Be concise and friendly. Format promotions clearly with bullet points when listing multiple offers.
"""

# ── Build / Load Index ─────────────────────────────────────────────────────────

def setup_knowledge_base(client: Groq, force_rebuild: bool = False):
    """
    Either loads an existing FAISS index from disk, or builds a new one
    by reading all PDFs in the docs/ folder.
    """
    if not force_rebuild and index_exists():
        print("✅ Found existing knowledge base. Loading...")
        index, chunks = load_index()
        print(f"   Loaded {len(chunks)} chunks from previous indexing.\n")
        return index, chunks

    # Need to build the index
    print("🔨 Building knowledge base from PDFs...")

    # Make sure the PDF folder exists
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"\n⚠️  Created folder '{PDF_FOLDER}/'.")
        print(f"   Please add your Azure resource PDF files to that folder, then run again.\n")
        sys.exit(0)

    chunks = load_and_chunk_pdfs(PDF_FOLDER)
    embeddings = embed_chunks(client, chunks)
    index = build_faiss_index(embeddings)
    save_index(index, chunks)
    print("✅ Knowledge base ready!\n")
    return index, chunks


# ── Answer Generation ──────────────────────────────────────────────────────────

def generate_answer(client: Groq, query: str, context_chunks: list[dict]) -> str:
    """
    Builds a prompt from retrieved context chunks and sends it to Llama via Groq.
    Returns the generated answer as a string.
    """
    # Build the context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    context_text = "\n\n---\n\n".join(context_parts)

    user_message = f"""Based on the following Azure resource documentation excerpts, answer the question.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
        temperature=0.2,   # Low temperature = more factual, less creative
    )

    return response.choices[0].message.content.strip()


# ── Main Chat Loop ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ☁️   Retail Promotions Bot — Powered by Llama via Groq   ☁️")
    print("=" * 60)
    print()

    # ── Check API Key ──────────────────────────────────────────────────────────
    api_key = os.environ.get(GROQ_API_KEY_ENV)
    if not api_key:
        print("❌ ERROR: Groq API key not found.")
        print(f"   Please set the environment variable '{GROQ_API_KEY_ENV}'.")
        print()
        print("   On Mac/Linux:  export GROQ_API_KEY=your_key_here")
        print("   On Windows:    set GROQ_API_KEY=your_key_here")
        print()
        print("   Get your free key at: https://console.groq.com")
        sys.exit(1)

    client = Groq(api_key=api_key)

    # ── Setup Knowledge Base ───────────────────────────────────────────────────
    index, chunks = setup_knowledge_base(client)

    # ── Chat Loop ──────────────────────────────────────────────────────────────
    print("💬 Chat started! Ask questions about your Azure resources.")
    print("   Commands: 'quit' to exit | 'rebuild' to re-index PDFs")
    print("-" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye! 👋")
            break

        if user_input.lower() == "rebuild":
            index, chunks = setup_knowledge_base(client, force_rebuild=True)
            continue

        # Retrieve relevant chunks
        print("🔍 Searching knowledge base...")
        relevant_chunks = retrieve(client, user_input, index, chunks)

        if not relevant_chunks:
            print("\nAssistant: I couldn't find relevant information in the documents.\n")
            continue

        # Generate answer
        print("🤖 Generating answer...\n")
        answer = generate_answer(client, user_input, relevant_chunks)

        print(f"Assistant: {answer}")
        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
