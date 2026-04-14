# Retail Promotions Bot

Retail Promotions Q&A — RAG-powered chat using Llama via Groq

---

## What This Does

You provide PDF documents describing the different Retail Promotions.
The app reads them, builds a searchable knowledge base, and lets you ask questions in plain English.

Example questions:

- "What are promotions available for Butter?"
- "Does Fruits have the Buy X Get Y offer"
- "Which promotions end this month?"
- "What are the restrictions for Baby product promotions?"

---

## Project Structure

```
cloudops_insight_hub/
|
|-- chat_app.py          <- MAIN FILE: run this to start chatting
|-- rag_pipeline.py      <- Core RAG logic (PDF reading, embeddings, search)
|-- requirements.txt     <- Python packages to install
|
|-- docs/          <- PUT YOUR AZURE RESOURCE PDFs IN HERE
|
|-- faiss_index.bin      <- Auto-created after first run
+-- chunks.pkl           <- Auto-created after first run
```

---

## Setup (Step by Step)

### Step 1 — Install Python

Make sure Python 3.10 or newer is installed.
Check: `python --version`

### Step 2 — Install Dependencies

Open a terminal in this project folder and run:

    pip install -r requirements.txt

### Step 3 — Get Your Free Groq API Key

1. Go to https://console.groq.com
2. Sign up (free account)
3. Click "API Keys" -> "Create API Key"
4. Copy the key (it starts with gsk\_...)

### Step 4 — Set the API Key as an Environment Variable

Mac/Linux:
export GROQ_API_KEY=gsk_your_key_here

Windows (Command Prompt):
set GROQ_API_KEY=gsk_your_key_here

Windows (PowerShell):
$env:GROQ_API_KEY="gsk_your_key_here"

TIP: To make this permanent on Mac/Linux, add the export line to your ~/.bashrc or ~/.zshrc file.

### Step 5 — Add Your PDFs

Copy your Azure resource PDF files into the docs/ folder.
These should describe your VMs, Storage Accounts, Databricks clusters, and costs.

### Step 6 — Run the App

    python chat_app.py

The FIRST run will read all PDFs and build the knowledge base (takes a few minutes).
All later runs load instantly from the saved index.

---

## Chat Commands

| What you type | What it does                       |
| ------------- | ---------------------------------- |
| Any question  | Asks about your Azure resources    |
| rebuild       | Re-reads your PDFs (after updates) |
| quit / exit   | Stops the app                      |

---

## How It Works (Simple Explanation)

1. READ: Your PDFs are read and cleaned
2. CHUNK: Text is split into small overlapping pieces (~500 chars each)
3. EMBED: Each chunk is converted to numbers (a "vector") that capture its meaning
4. INDEX: All vectors are stored in FAISS (a fast search database)
5. QUERY: Your question is also converted to a vector
6. SEARCH: FAISS finds the 5 most similar chunks to your question
7. ANSWER: Those chunks + your question are sent to Llama (via Groq), which writes the answer

This is called RAG — Retrieval Augmented Generation. It prevents the AI from making
things up because it can only answer based on your actual documents.

---

## Troubleshooting

Problem: "GROQ_API_KEY not found"
Fix: Make sure you set the environment variable (Step 4) in the SAME terminal window you run the app from.

Problem: "No PDF files found in docs/"
Fix: Add .pdf files to the docs/ folder. The folder must contain at least one PDF.

Problem: Slow embedding on first run
Fix: This is normal — each chunk needs one API call. With 100 chunks it may take 2-3 minutes.
Subsequent runs skip this step entirely.

Problem: Answers seem wrong or irrelevant
Fix: Type 'rebuild' to re-index. Also ensure your PDFs have searchable text (not scanned images).
