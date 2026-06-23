import os
from dotenv import load_dotenv
from agent import run

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
    exit(1)

print("TruCare KB Q&A Agent")
print("Type your question, or 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() in ("quit", "exit", "q"):
        break

    answer, tool_calls = run(question)

    print(f"\n[Retrieved from {len(tool_calls)} tool calls]")
    for tc in tool_calls:
        print(f"  turn {tc['turn']}: {tc['name']}({tc['inputs']})")

    print(f"\nAgent: {answer}\n")
    print("-" * 60 + "\n")
