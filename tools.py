import os
import json
import pdfplumber

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


# ── Python functions (the actual tool implementations) ──────────────────────

def list_documents() -> str:
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    if not files:
        return "No documents found."
    return json.dumps(files)


def get_toc(doc_name: str) -> str:
    """Read the first 10 pages to capture the table of contents."""
    path = os.path.join(DOCS_DIR, doc_name)
    if not os.path.exists(path):
        return f"Document '{doc_name}' not found."

    pages_text = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages[:10], start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"[Page {i}]\n{text}")

    return "\n\n".join(pages_text)


def read_pdf_page(doc_name: str, page_number: int) -> str:
    """Read the full text of one page (1-indexed)."""
    path = os.path.join(DOCS_DIR, doc_name)
    if not os.path.exists(path):
        return f"Document '{doc_name}' not found."

    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        if page_number < 1 or page_number > total:
            return f"Page {page_number} out of range. Document has {total} pages."
        page = pdf.pages[page_number - 1]
        text = page.extract_text() or ""
        tables = page.extract_tables()

        result = f"[{doc_name} — Page {page_number} of {total}]\n\n{text}"

        if tables:
            result += "\n\n[TABLES ON THIS PAGE]\n"
            for t in tables:
                for row in t:
                    cleaned = [cell or "" for cell in row]
                    result += " | ".join(cleaned) + "\n"

        return result


def search_pdf(doc_name: str, query: str, max_results: int = 6) -> str:
    """Case-insensitive keyword search. Returns up to max_results pages with snippets."""
    path = os.path.join(DOCS_DIR, doc_name)
    if not os.path.exists(path):
        return f"Document '{doc_name}' not found."

    query_lower = query.lower()
    matches = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if query_lower in text.lower():
                # Find the first occurrence and grab surrounding context
                idx = text.lower().find(query_lower)
                start = max(0, idx - 150)
                end = min(len(text), idx + 300)
                snippet = text[start:end].replace("\n", " ").strip()
                matches.append({"page": i, "snippet": f"...{snippet}..."})
                if len(matches) >= max_results:
                    break

    if not matches:
        return f"No matches found for '{query}' in {doc_name}."

    return json.dumps(matches, indent=2)


# ── Tool dispatcher ──────────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict) -> str:
    if name == "list_documents":
        return list_documents()
    elif name == "get_toc":
        return get_toc(inputs["doc_name"])
    elif name == "read_pdf_page":
        return read_pdf_page(inputs["doc_name"], inputs["page_number"])
    elif name == "search_pdf":
        return search_pdf(inputs["doc_name"], inputs["query"])
    else:
        return f"Unknown tool: {name}"


# ── Claude tool schemas (what we pass to the API) ───────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "list_documents",
        "description": "List all available PDF documents in the knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_toc",
        "description": (
            "Get the table of contents from a document. Use this first to understand "
            "the document's structure and find which page numbers cover a topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "Exact filename of the PDF (e.g. 'Release_Notes.pdf').",
                }
            },
            "required": ["doc_name"],
        },
    },
    {
        "name": "read_pdf_page",
        "description": (
            "Read the full text content of a specific page in a PDF. "
            "Use after get_toc or search_pdf to read the relevant page in full."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "Exact filename of the PDF.",
                },
                "page_number": {
                    "type": "integer",
                    "description": "Page number to read (1-indexed).",
                },
            },
            "required": ["doc_name", "page_number"],
        },
    },
    {
        "name": "search_pdf",
        "description": (
            "Search for a keyword or phrase across all pages of a PDF. "
            "Returns matching page numbers and text snippets. "
            "Use when you know a term but not the page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "Exact filename of the PDF.",
                },
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for.",
                },
            },
            "required": ["doc_name", "query"],
        },
    },
]
