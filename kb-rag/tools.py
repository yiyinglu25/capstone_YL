import os
import json
import pdfplumber
import chromadb
import voyageai

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "kb-qa", "documents")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "trucare_docs"
TICKETS_COLLECTION = "trucare_tickets"

_chroma_client = None
_collection = None
_tickets_collection = None
_voyage_client = None


def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def _get_collection():
    global _collection
    if _collection is None:
        _collection = _get_chroma().get_or_create_collection(COLLECTION_NAME)
    return _collection


def _get_tickets_collection():
    global _tickets_collection
    if _tickets_collection is None:
        _tickets_collection = _get_chroma().get_or_create_collection(TICKETS_COLLECTION)
    return _tickets_collection


def _get_voyage():
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _voyage_client


# ── Tool implementations ──────────────────────────────────────────────────────

def list_documents() -> str:
    """Return all PDFs grouped by version."""
    if not os.path.isdir(DOCS_DIR):
        return "No documents folder found."
    result = {}
    for version in sorted(os.listdir(DOCS_DIR)):
        version_dir = os.path.join(DOCS_DIR, version)
        if not os.path.isdir(version_dir):
            continue
        files = sorted(f for f in os.listdir(version_dir) if f.endswith(".pdf"))
        if files:
            result[version] = files
    return json.dumps(result, indent=2) if result else "No documents found."


def search_chunks(query: str, version: str, n_results: int = 8) -> str:
    """Semantic search over pre-embedded chunks, filtered to a specific version."""
    collection = _get_collection()
    voyage = _get_voyage()

    result = voyage.embed([query], model="voyage-3", input_type="query")
    query_embedding = result.embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"version": version},
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"][0]:
        return f"No results found for '{query}' in version {version}."

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "doc_name": meta["doc_name"],
            "page_number": meta["page_number"],
            "relevance_score": round(1 - dist, 3),
            "snippet": doc[:300],
        })

    return json.dumps(output, indent=2)


def search_tickets(query: str, n_results: int = 6) -> str:
    """Semantic search over Jira tickets — no version filter, spans all versions."""
    collection = _get_tickets_collection()
    voyage = _get_voyage()

    result = voyage.embed([query], model="voyage-3", input_type="query")
    query_embedding = result.embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"][0]:
        return f"No tickets found for '{query}'."

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "relevance_score": round(1 - dist, 3),
            "impacted_versions": meta.get("impacted_versions", ""),
            "resolved_version": meta.get("resolved_version", ""),
            "platform": meta.get("platform", ""),
            "ticket": doc[:400],
        })

    return json.dumps(output, indent=2)


def get_ticket_by_id(ticket_id: str) -> str:
    """Direct lookup of a Jira ticket by its exact ID (e.g. CP-389405)."""
    collection = _get_tickets_collection()
    results = collection.get(
        where={"ticket_id": ticket_id.upper()},
        include=["documents", "metadatas"],
    )
    if not results["documents"]:
        return f"No ticket found with ID {ticket_id}."
    doc = results["documents"][0]
    meta = results["metadatas"][0]
    return json.dumps({
        "ticket_id": meta.get("ticket_id"),
        "impacted_versions": meta.get("impacted_versions"),
        "resolved_version": meta.get("resolved_version"),
        "platform": meta.get("platform"),
        "description": doc,
    }, indent=2)


def read_pdf_page(doc_name: str, page_number: int) -> str:
    """Read the full text of one page (1-indexed). doc_name is version-prefixed e.g. 'v25.1/TruCare_Release_Notes.pdf'."""
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
                    result += " | ".join(cell or "" for cell in row) + "\n"
        return result


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


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict) -> str:
    if name == "list_documents":
        return list_documents()
    elif name == "search_chunks":
        return search_chunks(inputs["query"], inputs["version"])
    elif name == "search_tickets":
        return search_tickets(inputs["query"])
    elif name == "get_ticket_by_id":
        return get_ticket_by_id(inputs["ticket_id"])
    elif name == "read_pdf_page":
        return read_pdf_page(inputs["doc_name"], inputs["page_number"])
    elif name == "get_toc":
        return get_toc(inputs["doc_name"])
    else:
        return f"Unknown tool: {name}"


# ── Claude tool schemas ───────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "list_documents",
        "description": "List all available PDF documents grouped by version (v24.1, v24.2, v25.1, v25.2).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_chunks",
        "description": (
            "Semantic search over pre-embedded document chunks for a specific version. "
            "Returns the most relevant passages with doc name, page number, and a relevance score. "
            "Use this as your primary search tool — it understands meaning, not just keywords."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or topic to search for.",
                },
                "version": {
                    "type": "string",
                    "description": "Version to search within. Must be one of: v24.1, v24.2, v25.1, v25.2.",
                },
            },
            "required": ["query", "version"],
        },
    },
    {
        "name": "read_pdf_page",
        "description": (
            "Read the full text of a specific page in a PDF. "
            "Use after search_chunks to verify and expand on a retrieved passage before citing it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "Version-prefixed path (e.g. 'v25.1/TruCare_Release_Notes.pdf').",
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
        "name": "get_ticket_by_id",
        "description": (
            "Look up a specific Jira ticket by its exact ID (e.g. CP-389405). "
            "Always use this first when the user provides a specific ticket ID — "
            "it does a direct lookup, not a semantic search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The exact Jira ticket ID, e.g. 'CP-389405'.",
                },
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "search_tickets",
        "description": (
            "Search Jira tickets for known issues, bugs, and fixes across all TruCare versions. "
            "Use this when the question is about a specific bug, issue ID (e.g. CP-389405), "
            "known problem, or whether something has been fixed and in which version."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The bug, issue, or problem to search for.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_toc",
        "description": "Get the table of contents from a document to understand its structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "Version-prefixed path (e.g. 'v25.1/TruCare_Installation_Guide.pdf').",
                },
            },
            "required": ["doc_name"],
        },
    },
]
