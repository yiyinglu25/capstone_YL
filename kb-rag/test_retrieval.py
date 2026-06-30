"""
Quick retrieval test — run before starting the app.
Checks: semantic search, version filtering, and page read.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from tools import search_chunks, read_pdf_page

def test(label, query, version):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Query: '{query}'  |  Version: {version}")
    print('='*60)
    result = search_chunks(query, version)
    print(result[:800])
    return result

# Test 1: basic semantic search
test("Basic semantic search", "system requirements for installation", "v25.2")

# Test 2: same query different version — results should differ
test("Version filter check (v24.1)", "system requirements for installation", "v24.1")

# Test 3: specific feature query
test("Specific feature", "single sign-on configuration", "v25.1")

# Test 4: read a full page from a result
print(f"\n{'='*60}")
print("TEST: Full page read (v25.2 Installation Guide page 1)")
print('='*60)
print(read_pdf_page("v25.2/TruCare_Installation_Guide.pdf", 1)[:800])
