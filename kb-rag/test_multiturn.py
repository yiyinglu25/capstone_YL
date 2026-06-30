"""
Multi-turn conversation test.
Simulates a real back-and-forth to check if session memory and query rewriting work.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from agent import run

history = []

def ask(question):
    print(f"\n{'='*65}")
    print(f"USER: {question}")
    print('='*65)
    answer, tool_calls, rewritten, versions, confidence_score, confidence_detail, cost_info = run(question, history)
    print(f"Rewritten: {rewritten}")
    print(f"Versions: {versions}")
    print(f"Tool calls: {len(tool_calls)} — {[tc['name'] for tc in tool_calls]}")
    print(f"Cost: ${cost_info['total_cost']:.4f} (Haiku ${cost_info['haiku_cost']:.4f} | Opus ${cost_info['opus_cost']:.4f})")
    print(f"\nASSISTANT: {answer[:600]}...")

    # Add to history for next turn
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})


# Turn 1: establish topic and version
ask("What are the installation requirements for TruCare 25.2?")

# Turn 2: version follow-up — should carry v25.2 context
ask("What about for 24.1? How is it different?")

# Turn 3: vague pronoun reference — should resolve "it" from history
ask("Is it harder to install?")

# Turn 4: completely new topic — should not bleed prior context
ask("What is ticket CP-389405 about?")
