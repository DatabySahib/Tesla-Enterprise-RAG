"""Retrieval QA engine with source citations and confidence metadata.

Backend resolution order (``LLM_BACKEND=auto``):

1. **openai**   -- if ``OPENAI_API_KEY`` is set and ``langchain-openai`` is installed
2. **ollama**   -- if a local Ollama daemon answers on ``OLLAMA_BASE_URL``
3. **extractive** -- no generative model at all: return the retrieved passages
   verbatim, ranked. Always available, never hallucinates, and keeps the
   pipeline demonstrable on a machine with no LLM installed.

**huggingface** (a local seq2seq pipeline, ``google/flan-t5-base`` by default) is
deliberately excluded from ``auto``: selecting it triggers a ~1 GB model download,
which should be a deliberate choice rather than a surprise on first run. Opt in
with ``LLM_BACKEND=huggingface``.

Every answer carries the exact page numbers and the context snippets it was
derived from, plus a retrieval-similarity confidence score.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from .config import Settings, settings, use_utf8_stdout
from .embeddings import get_embeddings
from .vectorstore import get_retriever, get_vectorstore

logger = logging.getLogger(__name__)

Backend = Literal["openai", "ollama", "huggingface", "extractive"]

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior equity research analyst answering questions about \
Tesla's Q4 and FY 2025 Shareholder Update. You answer strictly from the supplied \
context excerpts.

RULES
1. Ground every claim in the CONTEXT. If the context does not contain the answer, \
say exactly: "The provided document does not contain this information." Do not \
speculate and do not use outside knowledge about Tesla.
2. Quote figures exactly as written, preserving units, scale and sign. The tables \
report dollars in millions unless the row says otherwise; percentages may be margins \
or year-over-year changes -- never conflate the two.
3. Table rows list periods in column order: Q4-2024, Q1-2025, Q2-2025, Q3-2025, \
Q4-2025, YoY (quarterly tables) or 2021, 2022, 2023, 2024, 2025, YoY (annual \
tables). Pick the value from the column the question asks about, and name the \
period you are quoting.
4. Distinguish GAAP from non-GAAP, quarterly from full-year, and production from \
deliveries. State which basis you are reporting.
5. Cite the page for each fact inline as [p. N]. Multiple pages: [p. 4, 8].
6. Be concise and factual. Lead with the number, then one or two sentences of \
context. No preamble, no disclaimers beyond rule 1.

CONTEXT
{context}"""

USER_PROMPT = """QUESTION: {question}

Answer using only the context above, with inline [p. N] citations."""

# Flat single-string variant for completion-style local models (FLAN-T5),
# which have no chat-role concept and a short context window.
FLAT_PROMPT = PromptTemplate.from_template(
    "Answer the question using only the excerpts from Tesla's Q4 2025 shareholder "
    "update below. Quote figures exactly. If the excerpts do not contain the answer, "
    'reply "The provided document does not contain this information."\n\n'
    "EXCERPTS:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
)


# --------------------------------------------------------------------------
# Response types
# --------------------------------------------------------------------------


@dataclass
class SourceCitation:
    """One retrieved passage, with everything the UI needs to render it."""

    page_number: int
    section: str
    chunk_id: str
    snippet: str
    full_text: str
    relevance: float
    is_tabular: bool

    def label(self) -> str:
        return f"p. {self.page_number} — {self.section}"


@dataclass
class RAGResponse:
    """A complete answer plus provenance and confidence metadata."""

    question: str
    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    confidence: float = 0.0
    confidence_label: str = "unknown"
    backend: str = "extractive"
    model: str = ""
    pages_cited: list[int] = field(default_factory=list)
    latency_seconds: float = 0.0
    chunks_retrieved: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# LLM backend resolution
# --------------------------------------------------------------------------


def _try_openai(config: Settings) -> tuple[Any, str] | None:
    import os

    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError:
        logger.debug("OPENAI_API_KEY set but langchain-openai is not installed.")
        return None
    logger.info("Using OpenAI backend (%s).", config.openai_model)
    return (
        ChatOpenAI(
            model=config.openai_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        ),
        config.openai_model,
    )


def _try_ollama(config: Settings) -> tuple[Any, str] | None:
    try:
        import urllib.request

        # Cheap liveness probe -- constructing the client alone does not verify
        # that a daemon is actually listening.
        with urllib.request.urlopen(f"{config.ollama_base_url}/api/tags", timeout=1.5):
            pass
    except Exception:  # noqa: BLE001 - any failure means "no local Ollama"
        return None

    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        except ImportError:
            logger.debug("Ollama daemon reachable but no langchain Ollama package installed.")
            return None

    logger.info("Using Ollama backend (%s).", config.ollama_model)
    return (
        ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=config.llm_temperature,
        ),
        config.ollama_model,
    )


def _try_huggingface(config: Settings) -> tuple[Any, str] | None:
    try:
        from langchain_huggingface import HuggingFacePipeline  # type: ignore
    except ImportError:
        try:
            from langchain_community.llms import HuggingFacePipeline  # type: ignore
        except ImportError:
            return None

    try:
        llm = HuggingFacePipeline.from_model_id(
            model_id=config.hf_model,
            task="text2text-generation",
            pipeline_kwargs={
                "max_new_tokens": min(config.llm_max_tokens, 512),
                "do_sample": False,
            },
        )
    except Exception as exc:  # noqa: BLE001 - model download/load can fail offline
        logger.warning("HuggingFace backend unavailable (%s).", exc)
        return None

    logger.info("Using local HuggingFace backend (%s).", config.hf_model)
    return llm, config.hf_model


_RESOLVERS = {
    "openai": _try_openai,
    "ollama": _try_ollama,
    "huggingface": _try_huggingface,
}


def resolve_llm(config: Settings = settings) -> tuple[Any | None, Backend, str]:
    """Pick the best available generative backend.

    Returns:
        ``(llm_or_None, backend_name, model_name)``. ``llm`` is ``None`` for
        the extractive backend, which needs no model.
    """
    requested = config.llm_backend.lower()

    if requested == "extractive":
        return None, "extractive", "retrieval-only"

    # huggingface is excluded from auto on purpose -- see the module docstring.
    order = ["openai", "ollama"] if requested == "auto" else [requested]
    for name in order:
        resolver = _RESOLVERS.get(name)
        if resolver is None:
            logger.warning("Unknown LLM_BACKEND %r; ignoring.", name)
            continue
        result = resolver(config)
        if result is not None:
            llm, model_name = result
            return llm, name, model_name  # type: ignore[return-value]
        if requested != "auto":
            logger.warning("Requested backend %r is unavailable; falling back to extractive.", name)

    logger.info("No generative backend available -- using extractive retrieval mode.")
    return None, "extractive", "retrieval-only"


# --------------------------------------------------------------------------
# Confidence scoring
# --------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def score_documents(question: str, docs: Sequence[Document]) -> list[float]:
    """Cosine relevance of each retrieved chunk to the question.

    Computed directly from the embedding model rather than read off the
    retriever, because MMR does not surface scores and different Chroma
    versions report distance on different scales.
    """
    if not docs:
        return []
    model = get_embeddings()
    query_vector = model.embed_query(question)
    doc_vectors = model.embed_documents([d.page_content for d in docs])
    return [round(max(0.0, _cosine(query_vector, dv)), 4) for dv in doc_vectors]


def _confidence_label(score: float) -> str:
    if score >= 0.60:
        return "high"
    if score >= 0.40:
        return "medium"
    if score >= 0.25:
        return "low"
    return "very low"


def aggregate_confidence(scores: Sequence[float]) -> tuple[float, str]:
    """Blend the best hit with the overall retrieval quality.

    A single strong match matters most, but a set where every chunk is weakly
    related signals the question may be out of scope, so the mean pulls the
    score down.
    """
    if not scores:
        return 0.0, "very low"
    top = max(scores)
    mean = sum(scores) / len(scores)
    blended = round(0.7 * top + 0.3 * mean, 4)
    return blended, _confidence_label(blended)


# --------------------------------------------------------------------------
# Context formatting
# --------------------------------------------------------------------------


def format_context(docs: Sequence[Document]) -> str:
    """Render retrieved chunks into a page-labelled context block."""
    blocks: list[str] = []
    for doc in docs:
        meta = doc.metadata
        header = f"[Page {meta.get('page_number', '?')} | Section: {meta.get('section', 'Unknown')}]"
        if meta.get("is_tabular"):
            header += " (financial table -- columns are period-ordered)"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


_STOPWORDS = frozenset(
    "the a an and or of for in on at to is are was were what which who whom how "
    "why when where did does do tesla new key with from by as its their our".split()
)


def _query_terms(question: str) -> list[str]:
    """Content words from the question, used to locate the relevant window."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


def _snippet(text: str, question: str = "", limit: int = 320) -> str:
    """Extract the passage window most relevant to ``question``.

    A chunk's opening characters are frequently the tail of the previous
    paragraph. Showing them hides the sentence that actually earned the match
    -- which matters most in extractive mode, where the snippet *is* the
    answer. This centres the window on the densest cluster of query terms.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed

    terms = _query_terms(question)
    if not terms:
        return collapsed[: limit - 1].rstrip() + "…"

    haystack = collapsed.lower()
    # Score each candidate window by how many distinct query terms it contains.
    step = 40
    best_start, best_score = 0, -1
    for start in range(0, max(1, len(collapsed) - limit + step), step):
        window = haystack[start : start + limit]
        score = sum(1 for term in terms if term in window)
        if score > best_score:
            best_start, best_score = start, score

    if best_score <= 0:
        return collapsed[: limit - 1].rstrip() + "…"

    # Snap to a word boundary so the window does not start mid-token.
    if best_start > 0:
        space = collapsed.find(" ", best_start)
        best_start = best_start if space == -1 else space + 1

    window = collapsed[best_start : best_start + limit].rstrip()
    prefix = "…" if best_start > 0 else ""
    suffix = "…" if best_start + limit < len(collapsed) else ""
    return f"{prefix}{window}{suffix}"


def _build_citations(
    docs: Sequence[Document], scores: Sequence[float], question: str = ""
) -> list[SourceCitation]:
    citations = [
        SourceCitation(
            page_number=int(doc.metadata.get("page_number", 0)),
            section=str(doc.metadata.get("section", "Unknown")),
            chunk_id=str(doc.metadata.get("chunk_id", "")),
            snippet=_snippet(doc.page_content, question),
            full_text=doc.page_content,
            relevance=score,
            is_tabular=bool(doc.metadata.get("is_tabular", False)),
        )
        for doc, score in zip(docs, scores)
    ]
    return sorted(citations, key=lambda c: c.relevance, reverse=True)


def _extractive_answer(citations: Sequence[SourceCitation]) -> str:
    """Compose an answer from the passages themselves, with no generation."""
    if not citations:
        return "The provided document does not contain this information."
    lines = [
        "**Retrieval-only mode** (no generative model configured). "
        "The most relevant passages from the document:",
        "",
    ]
    for rank, citation in enumerate(citations[:3], start=1):
        lines.append(f"**{rank}. [p. {citation.page_number}] {citation.section}** "
                     f"(relevance {citation.relevance:.2f})")
        lines.append(f"> {citation.snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class RAGEngine:
    """End-to-end question answering over the indexed Tesla document.

    Example:
        >>> engine = RAGEngine()
        >>> result = engine.answer("What was Tesla's Q4 2025 operating margin?")
        >>> result.pages_cited
        [4]
    """

    def __init__(
        self,
        *,
        config: Settings = settings,
        k: int | None = None,
        search_type: str | None = None,
        auto_build: bool = True,
    ) -> None:
        self.config = config
        self.k = k or config.retriever_k
        self.search_type = search_type or config.search_type
        self.vectorstore = get_vectorstore(auto_build=auto_build, config=config)
        self.retriever = get_retriever(
            self.vectorstore, k=self.k, search_type=self.search_type, config=config
        )
        self.llm, self.backend, self.model_name = resolve_llm(config)

    # -- retrieval ---------------------------------------------------------

    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        """Fetch the chunks most relevant to ``question``."""
        if k is not None and k != self.k:
            retriever = get_retriever(
                self.vectorstore, k=k, search_type=self.search_type, config=self.config
            )
        else:
            retriever = self.retriever
        return list(retriever.invoke(question))

    # -- generation --------------------------------------------------------

    def _generate(self, question: str, context: str) -> str:
        """Run the resolved LLM over the prompt and return plain text."""
        assert self.llm is not None  # extractive path never reaches here

        if self.backend == "huggingface":
            # FLAN-T5 has a 512-token window; trim context to fit.
            prompt_value = FLAT_PROMPT.format(question=question, context=context[:4000])
        else:
            prompt_value = CHAT_PROMPT.format_messages(question=question, context=context)

        raw = self.llm.invoke(prompt_value)
        text = getattr(raw, "content", raw)
        return str(text).strip()

    # -- public API --------------------------------------------------------

    def answer(self, question: str, *, k: int | None = None) -> RAGResponse:
        """Answer ``question`` with citations and confidence metadata.

        Args:
            question: Natural-language question about the document.
            k: Override the number of chunks to retrieve for this call.

        Returns:
            A :class:`RAGResponse`. Generation failures degrade to the
            extractive answer rather than raising, so the API stays up.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        started = time.perf_counter()
        docs = self.retrieve(question, k=k)
        scores = score_documents(question, docs)
        citations = _build_citations(docs, scores, question)
        confidence, label = aggregate_confidence(scores)

        backend_used = self.backend
        if self.llm is None:
            answer_text = _extractive_answer(citations)
        else:
            try:
                answer_text = self._generate(question, format_context(docs))
                if not answer_text:
                    raise ValueError("LLM returned an empty response.")
            except Exception as exc:  # noqa: BLE001 - degrade, don't crash the service
                logger.error("Generation failed on %s backend: %s", self.backend, exc)
                answer_text = _extractive_answer(citations)
                backend_used = f"{self.backend} (failed -> extractive)"

        return RAGResponse(
            question=question,
            answer=answer_text,
            sources=citations,
            confidence=confidence,
            confidence_label=label,
            backend=backend_used,
            model=self.model_name,
            pages_cited=sorted({c.page_number for c in citations}),
            latency_seconds=round(time.perf_counter() - started, 3),
            chunks_retrieved=len(docs),
        )


_ENGINE: RAGEngine | None = None


def get_engine(**kwargs: Any) -> RAGEngine:
    """Process-wide singleton engine (loads models and the index once)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RAGEngine(**kwargs)
    return _ENGINE


# Blueprint section 3 -- verification queries used by the CLI smoke test,
# the Streamlit example buttons and the README.
VERIFICATION_QUERIES: list[dict[str, str]] = [
    {
        "category": "Financial Metrics",
        "question": "What was Tesla's total gross profit and operating margin in Q4 2025?",
        "target": "Financial Summary (p. 4)",
    },
    {
        "category": "AI & Silicon",
        "question": "What are the key specifications and compute gains of the new AI5 inference chip?",
        "target": "AI & Software (p. 10)",
    },
    {
        "category": "Robotics & Fleet",
        "question": "What is the planned timeline and capacity for the Optimus Gen 3 production line?",
        "target": "Manufacturing & Hardware (p. 8)",
    },
]


def _main() -> None:  # pragma: no cover - CLI entry point
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Query the Tesla RAG engine.")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit to run the verification suite.")
    parser.add_argument("-k", type=int, default=None, help="Chunks to retrieve.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of formatted text.")
    args = parser.parse_args()

    use_utf8_stdout()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    engine = get_engine()

    questions = [args.question] if args.question else [q["question"] for q in VERIFICATION_QUERIES]

    for question in questions:
        response = engine.answer(question, k=args.k)
        if args.json:
            print(json.dumps(response.to_dict(), indent=2))
            continue
        print("=" * 78)
        print(f"Q: {response.question}")
        print("-" * 78)
        print(response.answer)
        print("-" * 78)
        print(
            f"backend={response.backend} model={response.model} "
            f"confidence={response.confidence:.2f} ({response.confidence_label}) "
            f"pages={response.pages_cited} latency={response.latency_seconds}s"
        )
        for citation in response.sources:
            print(f"  · [{citation.relevance:.3f}] {citation.label()} :: {citation.snippet[:160]}")
        print()


if __name__ == "__main__":  # pragma: no cover
    _main()
