import os
import streamlit as st
from dotenv import load_dotenv
from agent import run
from tools import DOCS_DIR

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TruCare KB Assistant",
    page_icon="📄",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("TruCare KB")
    st.caption("AI-powered product documentation assistant")
    st.divider()

    # Available documents
    st.subheader("📂 Knowledge Base")
    docs = sorted([f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")])
    for doc in docs:
        # Extract a clean display name: remove .pdf, replace _ with spaces
        label = doc.replace(".pdf", "").replace("_", " ")
        st.caption(f"• {label}")

    st.divider()

    # Session info + clear button
    exchanges = len(st.session_state.history) // 2
    if exchanges > 0:
        st.caption(f"💬 {exchanges} exchange(s) in this session")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

st.title("TruCare Knowledge Base Assistant")
st.caption("Ask questions about TruCare product documentation — versions 24.1 and 25.1.")
st.divider()

# ── Welcome screen (shown when no messages yet) ───────────────────────────────

SUGGESTED = [
    "What changed between version 24.1 and 25.1?",
    "What are the new features in TruCare 25.1?",
    "What Java version does TruCare 24.1 support?",
    "What is bug CP-353335 and is it fixed?",
]

if not st.session_state.messages:
    st.markdown("#### 👋 How can I help you today?")
    st.markdown("Try one of these questions or type your own:")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED):
        if cols[i % 2].button(q, use_container_width=True):
            st.session_state._prefill = q
            st.rerun()

# Handle prefilled question from suggestion button
prefill = st.session_state.pop("_prefill", None)

# ── Helper: human-readable tool call description ──────────────────────────────

def describe_tool(tc: dict) -> str:
    name = tc["name"]
    inputs = tc["inputs"]
    if name == "list_documents":
        return "📋 Listed available documents"
    if name == "get_toc":
        return f"📑 Browsed table of contents — {inputs.get('doc_name', '')}"
    if name == "search_pdf":
        return f"🔎 Searched **{inputs.get('doc_name', '')}** for *\"{inputs.get('query', '')}\"*"
    if name == "read_pdf_page":
        return f"📖 Read page {inputs.get('page_number')} of **{inputs.get('doc_name', '')}**"
    return f"`{name}`"

# ── Render existing chat history ──────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("from_history"):
                st.caption("💡 Answered from conversation context — no documents retrieved")
            elif msg.get("tool_calls"):
                with st.expander(f"🔍 {len(msg['tool_calls'])} source lookup(s)"):
                    for tc in msg["tool_calls"]:
                        st.markdown(describe_tool(tc))
                        if tc.get("result_preview"):
                            st.markdown(
                                f"<div style='background:#f8f9fa;border-left:3px solid #dee2e6;"
                                f"padding:8px 12px;border-radius:4px;font-size:0.82em;"
                                f"color:#555;margin:4px 0 12px 0'>{tc['result_preview'][:200]}…</div>",
                                unsafe_allow_html=True,
                            )

# ── Chat input ────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask a question about TruCare...") or prefill

if prompt:
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            answer, tool_calls = run(prompt, history=st.session_state.history)

        st.markdown(answer)

        from_history = len(tool_calls) == 0 and len(st.session_state.history) > 0
        if from_history:
            st.caption("💡 Answered from conversation context — no documents retrieved")
        elif tool_calls:
            with st.expander(f"🔍 {len(tool_calls)} source lookup(s)"):
                for tc in tool_calls:
                    st.markdown(describe_tool(tc))
                    if tc.get("result_preview"):
                        st.markdown(
                            f"<div style='background:#f8f9fa;border-left:3px solid #dee2e6;"
                            f"padding:8px 12px;border-radius:4px;font-size:0.82em;"
                            f"color:#555;margin:4px 0 12px 0'>{tc['result_preview'][:200]}…</div>",
                            unsafe_allow_html=True,
                        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_calls": tool_calls,
        "from_history": from_history,
    })

    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": answer})
