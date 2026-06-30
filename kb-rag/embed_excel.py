"""
Embeds the Jira tickets Excel sheet into a separate ChromaDB collection.
Run once after embed.py, or re-run anytime the Excel file is updated.

Usage:
    python embed_excel.py
"""

import os
import re
import openpyxl
import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "documents", "Knowledge base breakdown.xlsx")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
TICKETS_COLLECTION = "trucare_tickets"

ISSUES_SHEET = "TruCare Issues Search list"


def parse_tickets(path: str) -> list[dict]:
    wb = openpyxl.load_workbook(path)
    ws = wb[ISSUES_SHEET]

    tickets = []
    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if i == 1:
            continue  # skip header
        description, impacted, resolved, platform = row[0], row[1], row[2], row[3]
        if not description:
            continue

        # Extract ticket ID (e.g. CP-389405) from the start of the description
        desc_str = str(description).strip()
        match = re.match(r'(CP-\d+)', desc_str)
        ticket_id = match.group(1) if match else ""

        # Build a clean text block for embedding
        text = desc_str
        if impacted:
            text += f"\nImpacted versions: {impacted}"
        if resolved:
            text += f"\nResolved in: {resolved}"
        if platform:
            text += f"\nPlatform: {platform}"

        tickets.append({
            "text": text,
            "row": i,
            "ticket_id": ticket_id,
            "impacted_versions": str(impacted) if impacted else "",
            "resolved_version": str(resolved) if resolved else "",
            "platform": str(platform) if platform else "",
        })

    return tickets


def main():
    print(f"Reading: {EXCEL_PATH}")
    tickets = parse_tickets(EXCEL_PATH)
    print(f"Found {len(tickets)} tickets")

    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = chroma_client.get_or_create_collection(
        name=TICKETS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed in batches of 128
    batch_size = 128
    for batch_start in range(0, len(tickets), batch_size):
        batch = tickets[batch_start: batch_start + batch_size]
        texts = [t["text"] for t in batch]
        result = voyage_client.embed(texts, model="voyage-3", input_type="document")
        embeddings = result.embeddings

        ids = [f"ticket::row{t['row']}" for t in batch]
        metadatas = [{
            "ticket_id": t["ticket_id"],
            "impacted_versions": t["impacted_versions"],
            "resolved_version": t["resolved_version"],
            "platform": t["platform"],
            "row": t["row"],
        } for t in batch]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"  Embedded rows {batch_start + 2}–{batch_start + len(batch) + 1}")

    print(f"\nDone. Total tickets in DB: {collection.count()}")


if __name__ == "__main__":
    main()
