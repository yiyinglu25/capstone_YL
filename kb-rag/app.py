import os
import streamlit as st
from dotenv import load_dotenv
from agent import run, KNOWN_VERSIONS, DEFAULT_MODEL
from tools import DOCS_DIR

MODEL_OPTIONS = {
    "Haiku 4.5 — Fastest & cheapest": "claude-haiku-4-5-20251001",
    "Sonnet 4.6 — Balanced": "claude-sonnet-4-6",
    "Opus 4.8 — Most capable": "claude-opus-4-8",
}
MODEL_LABELS = {v: k.split(" — ")[0] for k, v in MODEL_OPTIONS.items()}
MODEL_DEFAULT_LABEL = "Sonnet 4.6 — Balanced"

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

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
if "selected_exchange" not in st.session_state:
    st.session_state.selected_exchange = None
if "past_sessions" not in st.session_state:
    st.session_state.past_sessions = []  # list of {"messages": [...], "history": [...]}
if "active_past_session" not in st.session_state:
    st.session_state.active_past_session = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_OPTIONS[MODEL_DEFAULT_LABEL]

# ── Check index is ready ──────────────────────────────────────────────────────

index_ready = os.path.isdir(CHROMA_DIR) and any(
    f for f in os.listdir(CHROMA_DIR) if not f.startswith(".")
) if os.path.isdir(CHROMA_DIR) else False

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("TruCare KB")
    st.caption("AI-powered product documentation assistant")

    if not index_ready:
        st.error("Vector index not built yet. Run `python embed.py` first.")
    else:
        st.success("Vector index ready")

    if st.button("✏️ New Chat", use_container_width=True, type="primary"):
        if st.session_state.messages:
            st.session_state.past_sessions.append({
                "messages": list(st.session_state.messages),
                "history": list(st.session_state.history),
            })
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.selected_exchange = None
        st.session_state.active_past_session = None
        st.rerun()

    st.divider()
    st.subheader("📂 Knowledge Base")
    for version in sorted(os.listdir(DOCS_DIR)):
        version_dir = os.path.join(DOCS_DIR, version)
        if not os.path.isdir(version_dir):
            continue
        docs = sorted(f for f in os.listdir(version_dir) if f.endswith(".pdf"))
        if docs:
            st.markdown(f"**{version}** ({len(docs)} docs)")
            with st.expander("View files"):
                for doc in docs:
                    st.caption(f"• {doc.replace('.pdf', '').replace('_', ' ')}")

    st.divider()
    st.subheader("🤖 Agent Model")
    model_label = st.radio(
        "Agent model",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(MODEL_DEFAULT_LABEL),
        label_visibility="collapsed",
    )
    st.session_state.selected_model = MODEL_OPTIONS[model_label]

    with st.expander("Which model for which task?"):
        st.markdown(
            "**Haiku 4.5** — $1 / $5 per M tokens  \n"
            "Best for simple factual questions with a clear answer in one section "
            "(e.g. \"What Java version does v24.1 require?\").\n\n"
            "**Sonnet 4.6** — $3 / $15 per M tokens  \n"
            "Best for most documentation questions: multi-step lookups, "
            "comparing two versions, or questions that may span several pages. "
            "Good balance of speed, quality, and cost.\n\n"
            "**Opus 4.8** — $5 / $25 per M tokens  \n"
            "Best for complex cross-version analysis, subtle reasoning "
            "(e.g. tracing a bug through multiple releases), or when accuracy "
            "is critical and cost is secondary."
        )

    st.divider()

    # ── Current chat history ──────────────────────────────────────────────────
    user_messages = [m for m in st.session_state.messages if m["role"] == "user"]

    if user_messages:
        st.subheader("💬 This Chat")
        for i, user_msg in enumerate(user_messages):
            label = user_msg["content"]
            label = label if len(label) <= 42 else label[:42] + "…"
            is_selected = st.session_state.selected_exchange == i and st.session_state.active_past_session is None
            btn_label = f"▶ {label}" if is_selected else label
            if st.button(btn_label, key=f"hist_{i}", use_container_width=True):
                if is_selected:
                    st.session_state.selected_exchange = None
                else:
                    st.session_state.selected_exchange = i
                    st.session_state.active_past_session = None
                st.rerun()

    # ── Previous sessions ─────────────────────────────────────────────────────
    if st.session_state.past_sessions:
        st.subheader("🕐 Previous Chats")
        for si in range(len(st.session_state.past_sessions) - 1, -1, -1):
            session = st.session_state.past_sessions[si]
            session_user_msgs = [m for m in session["messages"] if m["role"] == "user"]
            if not session_user_msgs:
                continue
            first_q = session_user_msgs[0]["content"]
            label = first_q if len(first_q) <= 38 else first_q[:38] + "…"
            n = len(session_user_msgs)
            caption = f"{n} Q" if n == 1 else f"{n} Qs"
            is_active = st.session_state.active_past_session == si
            btn_label = f"▶ {label} ({caption})" if is_active else f"{label} ({caption})"
            if st.button(btn_label, key=f"past_{si}", use_container_width=True):
                if is_active:
                    st.session_state.active_past_session = None
                    st.session_state.selected_exchange = None
                else:
                    st.session_state.active_past_session = si
                    st.session_state.selected_exchange = None
                st.rerun()

    st.divider()

# ── Header ────────────────────────────────────────────────────────────────────

st.title("TruCare Knowledge Base Assistant")
st.caption("Ask questions about TruCare product documentation — versions 24.1, 24.2, 25.1, and 25.2.")
st.divider()

# ── Welcome screen ────────────────────────────────────────────────────────────

SUGGESTED = [
    "What changed between version 24.1 and 25.1?",
    "What are the new features in TruCare 25.2?",
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

prefill = st.session_state.pop("_prefill", None)

# ── Helper: describe a tool call ──────────────────────────────────────────────

def _cost_html(ci: dict) -> str:
    agent_label = MODEL_LABELS.get(ci.get("agent_model", ""), "Agent")
    return (
        f"<div style='color:#999;font-size:0.8em;margin-top:6px'>"
        f"💰 Cost: <strong>${ci['total_cost']:.4f}</strong> &nbsp;|&nbsp; "
        f"Haiku rewrite: {ci['haiku_input']:,} in / {ci['haiku_output']:,} out "
        f"(${ci['haiku_cost']:.4f}) &nbsp;|&nbsp; "
        f"{agent_label} agent: {ci['agent_input']:,} in / {ci['agent_output']:,} out "
        f"(${ci['agent_cost']:.4f})</div>"
    )


def describe_tool(tc: dict) -> str:
    name = tc["name"]
    inputs = tc["inputs"]
    if name == "list_documents":
        return "📋 Listed available documents"
    if name == "get_toc":
        return f"📑 Browsed table of contents — {inputs.get('doc_name', '')}"
    if name == "search_chunks":
        return f"🔎 Semantic search in **{inputs.get('version', '')}** for *\"{inputs.get('query', '')}\"*"
    if name == "read_pdf_page":
        return f"📖 Read page {inputs.get('page_number')} of **{inputs.get('doc_name', '')}**"
    return f"`{name}`"

# ── Past session view ─────────────────────────────────────────────────────────

past_idx = st.session_state.active_past_session
if past_idx is not None and past_idx < len(st.session_state.past_sessions):
    session = st.session_state.past_sessions[past_idx]
    session_user_msgs = [m for m in session["messages"] if m["role"] == "user"]
    st.info(f"**Viewing previous chat — {len(session_user_msgs)} question(s)**  ·  Click the entry again in the sidebar to close.")
    for msg in session["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("confidence_score"):
                cs = msg["confidence_score"]
                cd = msg.get("confidence_detail", "")
                color, bg = ("#1a7f37", "#d4f5dc") if cs >= 80 else ("#9a6700", "#fff3cd") if cs >= 50 else ("#cf222e", "#ffd7d7")
                st.markdown(
                    f"<div style='background:{bg};border-radius:6px;padding:6px 12px;"
                    f"margin-bottom:8px;font-size:0.88em;color:{color}'>"
                    f"<strong>Confidence: {cs}/100</strong> — {cd}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                with st.expander(f"🔍 {len(msg['tool_calls'])} source lookup(s)"):
                    for tc in msg["tool_calls"]:
                        st.markdown(describe_tool(tc))
            if msg["role"] == "assistant" and msg.get("cost_info"):
                st.markdown(_cost_html(msg["cost_info"]), unsafe_allow_html=True)
    st.divider()

# ── Selected exchange view ────────────────────────────────────────────────────

sel = st.session_state.selected_exchange
if sel is not None and st.session_state.active_past_session is None:
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    asst_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
    if sel < len(user_msgs) and sel < len(asst_msgs):
        u = user_msgs[sel]
        a = asst_msgs[sel]
        st.info(f"**Viewing past question {sel + 1} of {len(user_msgs)}**")
        with st.chat_message("user"):
            st.markdown(u["content"])
        with st.chat_message("assistant"):
            if a.get("confidence_score"):
                cs, cd = a["confidence_score"], a.get("confidence_detail", "")
                color, bg = ("#1a7f37", "#d4f5dc") if cs >= 80 else ("#9a6700", "#fff3cd") if cs >= 50 else ("#cf222e", "#ffd7d7")
                st.markdown(
                    f"<div style='background:{bg};border-radius:6px;padding:6px 12px;"
                    f"margin-bottom:8px;font-size:0.88em;color:{color}'>"
                    f"<strong>Confidence: {cs}/100</strong> — {cd}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(a["content"])
            if a.get("tool_calls"):
                with st.expander(f"🔍 {len(a['tool_calls'])} source lookup(s)"):
                    for tc in a["tool_calls"]:
                        st.markdown(describe_tool(tc))
            if a.get("cost_info"):
                st.markdown(_cost_html(a["cost_info"]), unsafe_allow_html=True)
        st.divider()

# ── Render existing chat history ──────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("confidence_score"):
            cs = msg["confidence_score"]
            cd = msg.get("confidence_detail", "")
            if cs >= 80:
                color, bg = "#1a7f37", "#d4f5dc"
            elif cs >= 50:
                color, bg = "#9a6700", "#fff3cd"
            else:
                color, bg = "#cf222e", "#ffd7d7"
            st.markdown(
                f"<div style='background:{bg};border-radius:6px;padding:6px 12px;"
                f"margin-bottom:8px;font-size:0.88em;color:{color}'>"
                f"<strong>Confidence: {cs}/100</strong> — {cd}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("tool_calls"):
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
        if msg["role"] == "assistant" and msg.get("cost_info"):
            st.markdown(_cost_html(msg["cost_info"]), unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask a question about TruCare...") or prefill

if prompt:
    if not index_ready:
        st.error("Please run `python embed.py` to build the vector index before asking questions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            answer, tool_calls, rewritten, versions, confidence_score, confidence_detail, cost_info = run(
                prompt, history=st.session_state.history, model=st.session_state.selected_model
            )

        # Confidence score
        if confidence_score > 0:
            if confidence_score >= 80:
                color = "#1a7f37"
                bg = "#d4f5dc"
            elif confidence_score >= 50:
                color = "#9a6700"
                bg = "#fff3cd"
            else:
                color = "#cf222e"
                bg = "#ffd7d7"
            st.markdown(
                f"<div style='background:{bg};border-radius:6px;padding:6px 12px;"
                f"margin-bottom:8px;font-size:0.88em;color:{color}'>"
                f"<strong>Confidence: {confidence_score}/100</strong> — {confidence_detail}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(answer)

        if rewritten != prompt:
            st.caption(f"🔄 Interpreted as: *{rewritten}* (versions: {', '.join(versions)})")

        if tool_calls:
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

        st.markdown(_cost_html(cost_info), unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_calls": tool_calls,
        "confidence_score": confidence_score,
        "confidence_detail": confidence_detail,
        "cost_info": cost_info,
    })
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": answer})
