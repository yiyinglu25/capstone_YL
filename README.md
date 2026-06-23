## Knowledge Base Q&A Agent

An agentic AI assistant that answers questions about TruCare product documentation by reading PDFs directly using tools — no fixed retrieval chain.

## What it does

- Answers questions about TruCare release notes, user guides, and technical documentation
- Compares changes across versions (e.g. 24.1 vs 25.1)
- Cites the exact document and page number for every answer
- Skips re-retrieval when the answer is already in the conversation context
- Declines out-of-scope questions that aren't in the documentation
- Logs every retrieval decision to a JSON file for evaluation

## Architecture

The agent receives a question and decides on its own what to look up:

```
User question
     ↓
Agent loop (Claude claude-opus-4-8 + tool calling)
     ↓
Decides which tool(s) to call:
  • list_documents   — see what PDFs are available
  • get_toc          — browse a document's table of contents
  • search_pdf       — keyword search across all pages
  • read_pdf_page    — read a specific page in full
     ↓
Feeds results back, repeats until it has enough to answer
     ↓
Returns answer with citations + logs the retrieval decisions
```

This is an **agentic pipeline** — the model plans its own retrieval rather than following a fixed embed-retrieve-generate chain.

## Project structure

```
kb-qa/
  agent.py          # Agent loop — tool calling, conversation history, logging
  tools.py          # PDF tools (list, search, read) + Claude tool schemas
  app.py            # Streamlit chat UI
  main.py           # Terminal CLI (for testing)
  requirements.txt
  .env.example      # Copy to .env and add your API key
  docs/             # Place TruCare PDF documents here (not committed)
  logs/             # Auto-created — one JSON log file per question
```

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/yiyinglu25/capstone_YL.git
cd capstone_YL
pip install -r requirements.txt
```

**2. Add your Anthropic API key**
```bash
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-...
```

**3. Add TruCare PDF documents**

Place your TruCare PDF files in the `docs/` folder. Name them with version numbers for clarity:
```
docs/
  TruCare_Release_Notes_24.1.pdf
  TruCare_Release_Notes_25.1.pdf
  TruCare_User_Guide_24.1.pdf
  ...
```

**4. Run the app**
```bash
# Chat UI (browser)
streamlit run app.py

# Terminal CLI
python3 main.py
```

## Example questions

- `What changed between version 24.1 and 25.1?`
- `What are the new features in TruCare 25.1?`
- `What is bug CP-353335 and is it fixed?`
- `What Java version does TruCare 24.1 support?`
- `How do I install TruCare?`

## Logs

Every question generates a JSON log in `logs/` recording:
- The question asked
- Every tool call the agent made (name, inputs, result preview)
- Whether retrieval was skipped (answer came from conversation history)
- The final answer

Used for evaluating retrieval accuracy against the golden Q&A set.

## Tech stack

- [Anthropic Claude](https://www.anthropic.com) — claude-opus-4-8 with native tool calling
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF text and table extraction
- [Streamlit](https://streamlit.io) — chat UI
- Python 3.10+
