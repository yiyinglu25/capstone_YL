import os
import json
import anthropic
from datetime import datetime
from tools import execute_tool, TOOL_SCHEMAS

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

KNOWN_VERSIONS = ["v24.1", "v24.2", "v25.1", "v25.2"]

REWRITE_PROMPT = """You are a query preprocessor for a TruCare documentation assistant.

Given a user question and the conversation history, extract:
1. The TruCare version(s) the question is about (v24.1, v24.2, v25.1, v25.2). If the user explicitly names a version, that always takes priority over history. If no version is mentioned, infer from context or return all versions.
2. A clean, standalone version of the question (resolve pronouns and follow-ups using history).

Return JSON only:
{
  "versions": ["v25.1"],
  "rewritten_question": "What are the new features in TruCare v25.1?"
}"""

SYSTEM_PROMPT = """You are a Q&A assistant for TruCare product documentation (versions v24.1, v24.2, v25.1, v25.2).

Retrieval strategy:
1. Use search_chunks to find relevant passages — it understands meaning, not just keywords.
2. Always call read_pdf_page to verify the full page before making any claim — never assert from a chunk snippet alone.
3. Use get_toc if you need to understand a document's structure first.
4. For version comparison questions, call search_chunks for each version separately.
5. When the user mentions a specific ticket ID (e.g. CP-389405), always use get_ticket_by_id first — it does a direct lookup. Only use search_tickets for general issue/bug queries without a specific ID.

Answer rules:
- Use inline citations immediately after each factual claim, in this format: [doc_name, p.X]
  Example: "TruCare 25.2 requires JDK 17 [v25.2/TruCare_Installation_Guide.pdf, p.8] and SQL Server 2022 [v25.2/TruCare_Installation_Guide.pdf, p.23]."
- Never make a claim without having read the full page via read_pdf_page — do not cite from chunk snippets alone.
- If the page read does not confirm the chunk, do not include that claim.
- If evidence is insufficient, respond with ABSTAIN and explain what was not found.
- Do not guess or infer beyond what the documents say.
- Decline questions outside the scope of TruCare product documentation.

Confidence score — end every answer (except ABSTAIN) with exactly this block:
---
Confidence: [0-100]
Detail: [one sentence — which page(s) confirmed the answer, or why confidence is lower]"""

MAX_TURNS = 8


def _rewrite_query(question: str, history: list, client: anthropic.Anthropic) -> tuple[str, list[str], int, int]:
    """Extract version(s) and rewrite the question. Returns (rewritten, versions, input_tokens, output_tokens)."""
    history_text = ""
    if history:
        for msg in history[-6:]:  # last 3 exchanges
            role = msg["role"].upper()
            content = msg["content"] if isinstance(msg["content"], str) else "[tool results]"
            history_text += f"{role}: {content}\n"

    user_content = f"Conversation history:\n{history_text}\nCurrent question: {question}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=REWRITE_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens

    try:
        parsed = json.loads(response.content[0].text)
        versions = [v for v in parsed.get("versions", KNOWN_VERSIONS) if v in KNOWN_VERSIONS]
        rewritten = parsed.get("rewritten_question", question)
        return rewritten, versions or KNOWN_VERSIONS, in_tok, out_tok
    except Exception:
        return question, KNOWN_VERSIONS, in_tok, out_tok


def _parse_confidence(answer: str) -> tuple[str, int, str]:
    """
    Split the agent's answer into (clean_answer, confidence_score, confidence_detail).
    Looks for the trailing block:
        ---
        Confidence: 85
        Detail: Confirmed on p.23 of the Installation Guide.
    """
    import re
    pattern = r"\n---\s*\nConfidence:\s*(\d+)\s*\nDetail:\s*(.+)"
    match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
    if match:
        score = min(100, max(0, int(match.group(1))))
        detail = match.group(2).strip()
        clean = answer[:match.start()].strip()
        return clean, score, detail
    return answer, 0, ""


_HAIKU_INPUT_PRICE  = 1.00 / 1_000_000   # $/token
_HAIKU_OUTPUT_PRICE = 5.00 / 1_000_000

_MODEL_PRICING = {
    "claude-haiku-4-5-20251001": (1.00 / 1_000_000,  5.00 / 1_000_000),
    "claude-sonnet-4-6":         (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-opus-4-8":           (5.00 / 1_000_000, 25.00 / 1_000_000),
}

DEFAULT_MODEL = "claude-opus-4-8"


def run(question: str, history: list = None, model: str = DEFAULT_MODEL) -> tuple[str, list, str, list[str], int, str, dict]:
    """
    Run the agent on a question.

    Returns (answer, tool_calls, rewritten_question, versions, confidence_score, confidence_detail, cost_info).
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    history = history or []

    # Step 1: rewrite query and extract versions (Haiku call)
    rewritten, versions, haiku_in, haiku_out = _rewrite_query(question, history, client)

    # Step 2: inject version context so the agent knows what to search
    version_note = f"[Detected versions for this question: {', '.join(versions)}]"
    messages = list(history)
    messages.append({"role": "user", "content": f"{version_note}\n\n{rewritten}"})

    tool_calls_log = []
    agent_in = 0
    agent_out = 0

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        agent_in += response.usage.input_tokens
        agent_out += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            raw_answer = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "(No answer generated)"
            )
            answer, confidence_score, confidence_detail = _parse_confidence(raw_answer)
            cost_info = _calc_cost(haiku_in, haiku_out, agent_in, agent_out, model)
            _save_log(question, rewritten, versions, tool_calls_log, answer, confidence_score, cost_info)
            return answer, tool_calls_log, rewritten, versions, confidence_score, confidence_detail, cost_info

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
    cost_info = _calc_cost(haiku_in, haiku_out, agent_in, agent_out, model)
    _save_log(question, rewritten, versions, tool_calls_log, answer, 0, cost_info)
    return answer, tool_calls_log, rewritten, versions, 0, "", cost_info


def _calc_cost(haiku_in: int, haiku_out: int, agent_in: int, agent_out: int, agent_model: str) -> dict:
    haiku_cost = haiku_in * _HAIKU_INPUT_PRICE + haiku_out * _HAIKU_OUTPUT_PRICE
    in_price, out_price = _MODEL_PRICING.get(agent_model, _MODEL_PRICING[DEFAULT_MODEL])
    agent_cost = agent_in * in_price + agent_out * out_price
    return {
        "haiku_input": haiku_in,
        "haiku_output": haiku_out,
        "haiku_cost": haiku_cost,
        "agent_model": agent_model,
        "agent_input": agent_in,
        "agent_output": agent_out,
        "agent_cost": agent_cost,
        "total_cost": haiku_cost + agent_cost,
    }


def _save_log(question, rewritten, versions, tool_calls, answer, confidence_score=0, cost_info=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.now().isoformat(),
        "original_question": question,
        "rewritten_question": rewritten,
        "detected_versions": versions,
        "tool_calls": tool_calls,
        "total_tool_calls": len(tool_calls),
        "confidence_score": confidence_score,
        "answer": answer,
        "cost_info": cost_info,
    }
    path = os.path.join(LOGS_DIR, f"{timestamp}.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
