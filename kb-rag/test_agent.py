"""
End-to-end agent tests. Runs 4 questions and prints answer + tool calls.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from agent import run

tests = [
    # (label, question, history)
    ("Basic factual", "What are the system requirements for TruCare 25.2?", []),
    ("Specific bug/keyword", "What is new in TruCare 25.1 release notes?", []),
    ("Version comparison", "What changed in the installation guide between v24.1 and v25.2?", []),
    ("Out of scope — should ABSTAIN", "What is the stock price of TruCare?", []),
]

for label, question, history in tests:
    print(f"\n{'='*65}")
    print(f"TEST: {label}")
    print(f"Q: {question}")
    print('='*65)
    answer, tool_calls, rewritten, versions, confidence_score, confidence_detail, cost_info = run(question, history)
    print(f"Rewritten: {rewritten}")
    print(f"Versions detected: {versions}")
    print(f"Tool calls: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"  [{tc['turn']}] {tc['name']} | inputs: {tc['inputs']}")
    print(f"\nANSWER:\n{answer}")
    print(f"Cost: ${cost_info['total_cost']:.4f} (Haiku ${cost_info['haiku_cost']:.4f} | Opus ${cost_info['opus_cost']:.4f})")
