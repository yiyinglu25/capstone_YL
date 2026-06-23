import os
import json
import anthropic
from datetime import datetime
from tools import execute_tool, TOOL_SCHEMAS

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are a Q&A assistant for TruCare product documentation.
You answer questions by reading the PDF documents in the knowledge base using your tools.

Conversation memory rules (check these FIRST before calling any tool):
- If the conversation history already contains the answer to the current question, answer directly from history — do NOT call any tools.
- If the current question is a rephrasing or slight variation of a previous question, treat it as the same question and reuse the prior answer.
- Only call tools when the current question requires information not already present in the conversation history.

Retrieval strategy (only when tools are needed):
1. Start with list_documents to see what's available.
2. Use get_toc on the relevant document to find which pages cover the topic.
3. Use search_pdf when you know a keyword (e.g. a bug ID like CP-353335, a feature name).
4. Use read_pdf_page to read the full content of a specific page.
5. For version comparison questions, search and read pages from both versions.

Answer rules:
- Always cite your source: document name and page number.
- If the answer is not in the documents, say so clearly — do not guess.
- For version comparison questions, explicitly state what changed and between which versions.
- Decline questions that are outside the scope of the product documentation.
"""

MAX_TURNS = 10


def run(question: str, history: list = None) -> tuple[str, list]:
    """
    Run the agent on a question.

    history: list of prior {"role": "user"/"assistant", "content": str} exchanges.
             The agent sees this context and will skip retrieval if the answer
             is already there.

    Returns (answer, tool_calls).
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Build message list: prior exchanges + current question
    messages = list(history) if history else []
    messages.append({"role": "user", "content": question})

    tool_calls_log = []

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            answer = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "(No answer generated)"
            )
            _save_log(question, tool_calls_log, answer, used_history=bool(history))
            return answer, tool_calls_log

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_calls_log.append({
                        "turn": turn + 1,
                        "name": block.name,
                        "inputs": block.input,
                        "result_preview": result[:300],
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})

    answer = "Reached maximum turns without a final answer."
    _save_log(question, tool_calls_log, answer, used_history=bool(history))
    return answer, tool_calls_log


def _save_log(question: str, tool_calls: list, answer: str, used_history: bool = False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "used_conversation_history": used_history,
        "tool_calls": tool_calls,
        "total_tool_calls": len(tool_calls),
        "skipped_retrieval": len(tool_calls) == 0 and used_history,
        "answer": answer,
    }
    path = os.path.join(LOGS_DIR, f"{timestamp}.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
