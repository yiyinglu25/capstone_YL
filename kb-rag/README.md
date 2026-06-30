# TruCare KB Assistant — RAG Version

Hybrid RAG pipeline: semantic vector search (Voyage AI + ChromaDB) with full-page verification via pdfplumber.

## Architecture

```
User question
     ↓
Query rewrite (Haiku) — resolve follow-ups, extract version(s)
     ↓
Agent loop (Opus + tool calling, cap 8 turns)
     ↓
  search_chunks   — semantic search in ChromaDB, version hard-filtered
  read_pdf_page   — read full page verbatim before citing
  get_toc         — browse document structure if needed
     ↓
Answer with citations (doc path + page number)
     or ABSTAIN if evidence not found
```

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add API keys**
```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY and VOYAGE_API_KEY
```

**3. Build the vector index (run once)**
```bash
python embed.py                  # embed all versions
python embed.py --version v25.2  # or one version only
```

**4. Run the app**
```bash
streamlit run app.py
```

## Adding a new PDF version

```bash
# 1. Add the new version folder to kb-qa/documents/
# 2. Re-run embed for just that version
python embed.py --version v26.1
```

## Project structure

```
kb-rag/
  embed.py        # One-time PDF chunking + embedding pipeline
  tools.py        # search_chunks, read_pdf_page, get_toc + tool schemas
  agent.py        # Query rewrite + agent loop + logging
  app.py          # Streamlit chat UI
  chroma_db/      # Auto-created — persisted vector index
  logs/           # Auto-created — one JSON log per question
```

## Tech stack

- Anthropic Claude Opus — agent reasoning + answer generation
- Anthropic Claude Haiku — query rewriting (fast + cheap)
- Voyage AI voyage-3 — document + query embeddings
- ChromaDB — local vector store with metadata filtering
- pdfplumber — full-page text extraction for citation verification
- Streamlit — chat UI
