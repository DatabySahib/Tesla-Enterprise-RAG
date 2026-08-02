"""Streamlit chat dashboard for the Tesla Q4-2025 RAG system.

Run with::

    streamlit run app.py

Features: conversational Q&A, per-answer source previews with page numbers and
relevance bars, live retrieval parameter controls, index management, and the
blueprint's verification queries as one-click examples.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from src import config as config_module
from src.config import settings
from src.rag_engine import VERIFICATION_QUERIES, RAGEngine, RAGResponse
from src.vectorstore import build_index, index_stats, is_indexed

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="Tesla Q4-2025 | Enterprise RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIDENCE_COLORS = {
    "high": "#16a34a",
    "medium": "#ca8a04",
    "low": "#ea580c",
    "very low": "#dc2626",
    "unknown": "#6b7280",
}

st.markdown(
    """
    <style>
      .stChatMessage { border-radius: 10px; }
      .metric-row { display: flex; gap: 1.25rem; flex-wrap: wrap; font-size: 0.85rem;
                    color: #6b7280; margin-top: 0.35rem; }
      .conf-pill { display: inline-block; padding: 1px 9px; border-radius: 999px;
                   color: #fff; font-size: 0.75rem; font-weight: 600; }
      .cite-head { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.2rem; }
      .cite-body { font-size: 0.85rem; color: #374151; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_engine(k: int, search_type: str) -> RAGEngine:
    """Build the engine once per (k, search_type) combination.

    Cached as a resource because it holds the embedding model and the open
    ChromaDB handle -- Streamlit reruns this script on every widget change.
    """
    return RAGEngine(k=k, search_type=search_type, auto_build=False)


def render_confidence(response: RAGResponse) -> str:
    color = CONFIDENCE_COLORS.get(response.confidence_label, CONFIDENCE_COLORS["unknown"])
    return (
        f'<span class="conf-pill" style="background:{color}">'
        f"{response.confidence_label.upper()} · {response.confidence:.2f}</span>"
    )


def render_sources(response: RAGResponse) -> None:
    """Source document preview with highlighted citations."""
    if not response.sources:
        st.info("No source passages were retrieved for this question.")
        return

    pages = ", ".join(f"p. {p}" for p in response.pages_cited)
    with st.expander(f"📄 Sources — {len(response.sources)} passages from {pages}", expanded=False):
        for rank, citation in enumerate(response.sources, start=1):
            badge = "📊 table" if citation.is_tabular else "📝 prose"
            st.markdown(
                f'<div class="cite-head">{rank}. Page {citation.page_number} — '
                f"{citation.section} &nbsp;·&nbsp; {badge} &nbsp;·&nbsp; "
                f"relevance {citation.relevance:.3f}</div>",
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, citation.relevance))
            st.code(citation.full_text, language=None)
            st.caption(f"chunk_id: `{citation.chunk_id}`")
            if rank < len(response.sources):
                st.divider()


def render_response(response: RAGResponse) -> None:
    st.markdown(response.answer)
    st.markdown(
        f'<div class="metric-row">{render_confidence(response)}'
        f"<span>🔎 {response.chunks_retrieved} chunks</span>"
        f"<span>📑 pages {response.pages_cited}</span>"
        f"<span>⚙️ {response.backend} · {response.model}</span>"
        f"<span>⏱️ {response.latency_seconds:.2f}s</span></div>",
        unsafe_allow_html=True,
    )
    render_sources(response)


# --------------------------------------------------------------------------
# Sidebar -- configuration and index management
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("⚡ Control Panel")

    st.subheader("Retrieval parameters")
    top_k = st.slider(
        "Chunks retrieved (k)",
        min_value=1,
        max_value=15,
        value=settings.retriever_k,
        help="How many passages to feed the model. Higher = more context, slower.",
    )
    search_type = st.radio(
        "Search strategy",
        options=["mmr", "similarity"],
        index=0 if settings.search_type == "mmr" else 1,
        horizontal=True,
        help=(
            "MMR diversifies results — useful here because the quarterly and annual "
            "financial tables contain near-identical rows."
        ),
    )

    st.divider()
    st.subheader("Vector index")
    stats: dict[str, Any] = index_stats()

    if stats["indexed"]:
        col_a, col_b = st.columns(2)
        col_a.metric("Vectors", f"{stats['vector_count']:,}")
        col_b.metric("Pages", stats.get("pages_indexed", "—"))
        st.caption(f"Collection `{stats['collection']}` · {stats['distance_metric']} distance")
        if stats.get("sections"):
            st.caption("Sections: " + ", ".join(stats["sections"]))
    else:
        st.warning("No index found. Build it to start querying.")

    if st.button("🔄 Rebuild index", use_container_width=True):
        with st.spinner("Re-ingesting the PDF and embedding chunks…"):
            build_index(rebuild=True)
        st.cache_resource.clear()
        st.success("Index rebuilt.")
        st.rerun()

    st.divider()
    st.subheader("Pipeline configuration")
    st.json(config_module.describe(), expanded=False)

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# Main pane
# --------------------------------------------------------------------------

st.title("Tesla Q4 & FY 2025 — Document Intelligence")
st.caption(
    "Retrieval-augmented Q&A over the Tesla Q4-2025 Shareholder Update. "
    "Every answer is grounded in the source PDF and cited to the page."
)

if not is_indexed():
    st.error(
        "The vector index has not been built yet. Use **Rebuild index** in the sidebar, "
        "or run `python -m src.vectorstore --build` from a terminal."
    )
    st.stop()

if not settings.pdf_path.is_file():
    st.warning(f"Source PDF missing at `{settings.pdf_path}` — queries will still work "
               f"against the existing index, but rebuilding will fail.")

engine = load_engine(top_k, search_type)

if engine.backend == "extractive":
    st.info(
        "**Retrieval-only mode.** No generative backend is configured, so answers are the "
        "ranked source passages themselves. Set `OPENAI_API_KEY`, start a local Ollama "
        "daemon, or set `LLM_BACKEND=huggingface` in `.env` to enable synthesised answers.",
        icon="ℹ️",
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Verification queries from the blueprint, as one-click examples.
st.markdown("**Try a verification query:**")
example_cols = st.columns(len(VERIFICATION_QUERIES))
pending_question: str | None = None
for column, item in zip(example_cols, VERIFICATION_QUERIES):
    with column:
        if st.button(item["category"], use_container_width=True, help=item["question"]):
            pending_question = item["question"]
        st.caption(item["target"])

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("response") is not None:
            render_response(message["response"])
        else:
            st.markdown(message["content"])

typed_question = st.chat_input("Ask about Tesla's Q4 2025 financials, AI silicon, Optimus, Robotaxi…")
question = typed_question or pending_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and analysing…"):
            try:
                response = engine.answer(question)
            except Exception as exc:  # noqa: BLE001 - surface the error in the UI
                st.error(f"Query failed: {exc}")
                response = None
        if response is not None:
            render_response(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.answer, "response": response}
            )
