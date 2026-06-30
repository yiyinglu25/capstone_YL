"""
One-time script to chunk all PDFs and embed them into ChromaDB.
Run this once before starting the app, and re-run when new PDF versions are added.

Usage:
    python embed.py                        # embed all versions
    python embed.py --version v25.2        # embed one version only
"""

import os
import argparse
import pdfplumber
import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "kb-qa", "documents")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "trucare_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, doc_name: str, version: str, page_number: int) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append({
                "text": chunk,
                "version": version,
                "doc_name": doc_name,
                "page_number": page_number,
            })
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_version(version: str, collection, voyage_client):
    version_dir = os.path.join(DOCS_DIR, version)
    if not os.path.isdir(version_dir):
        print(f"  Version folder not found: {version_dir}")
        return

    pdfs = sorted(f for f in os.listdir(version_dir) if f.endswith(".pdf"))
    print(f"\n{version}: {len(pdfs)} PDFs")

    for pdf_name in pdfs:
        path = os.path.join(version_dir, pdf_name)
        doc_path = f"{version}/{pdf_name}"
        all_chunks = []

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    all_chunks.extend(chunk_text(text, doc_path, version, i))

        if not all_chunks:
            print(f"  Skipped (no text): {pdf_name}")
            continue

        # Embed in batches of 128 (Voyage API limit)
        batch_size = 128
        for batch_start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[batch_start: batch_start + batch_size]
            texts = [c["text"] for c in batch]
            result = voyage_client.embed(texts, model="voyage-3", input_type="document")
            embeddings = result.embeddings

            ids = [f"{doc_path}::p{c['page_number']}::c{batch_start + j}" for j, c in enumerate(batch)]
            metadatas = [{
                "version": c["version"],
                "doc_name": c["doc_name"],
                "page_number": c["page_number"],
            } for c in batch]

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        print(f"  Embedded {len(all_chunks)} chunks: {pdf_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="Embed a specific version only (e.g. v25.2)")
    args = parser.parse_args()

    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if args.version:
        versions = [args.version]
    else:
        versions = sorted(
            d for d in os.listdir(DOCS_DIR)
            if os.path.isdir(os.path.join(DOCS_DIR, d))
        )

    print(f"Embedding versions: {versions}")
    for version in versions:
        embed_version(version, collection, voyage_client)

    print(f"\nDone. Total chunks in DB: {collection.count()}")


if __name__ == "__main__":
    main()
