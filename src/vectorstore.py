"""Persistent ChromaDB index and retriever factory.

The index lives on disk under ``./chroma_db`` so ingestion runs once and both
the Streamlit UI and the FastAPI service attach to the same collection
without re-embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from .config import Settings, settings, use_utf8_stdout
from .embeddings import get_embeddings
from .loader import load_and_split

logger = logging.getLogger(__name__)


def _import_chroma():
    """Prefer the maintained ``langchain-chroma`` package over the community copy."""
    try:
        from langchain_chroma import Chroma  # type: ignore

        return Chroma
    except ImportError:  # pragma: no cover - depends on installed extras
        from langchain_community.vectorstores import Chroma  # type: ignore

        logger.warning(
            "langchain-chroma not installed; falling back to the deprecated "
            "langchain_community implementation."
        )
        return Chroma


def _deterministic_id(document: Document) -> str:
    """Stable ID derived from content + location.

    Re-running ingestion upserts rather than duplicating, so the collection
    stays consistent when the pipeline is re-run against the same PDF.
    """
    payload = (
        f"{document.metadata.get('source', '')}"
        f"|{document.metadata.get('page_number', '')}"
        f"|{document.metadata.get('start_index', '')}"
        f"|{document.page_content}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _sanitize_metadata(document: Document) -> dict[str, Any]:
    """Chroma accepts only str/int/float/bool/None metadata values."""
    clean: dict[str, Any] = {}
    for key, value in document.metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def connect(
    embeddings: Embeddings | None = None, config: Settings = settings
) -> VectorStore:
    """Attach to the persistent collection, creating it if absent."""
    Chroma = _import_chroma()
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(config.chroma_dir),
        collection_metadata={"hnsw:space": config.distance_metric},
    )


def count(store: VectorStore | None = None, config: Settings = settings) -> int:
    """Number of vectors currently in the collection (0 if uninitialised)."""
    try:
        target = store if store is not None else connect(config=config)
        return target._collection.count()  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - a missing/corrupt index means "empty"
        logger.debug("Could not count collection: %s", exc)
        return 0


def is_indexed(config: Settings = settings) -> bool:
    """True when a usable index already exists on disk."""
    return config.chroma_dir.is_dir() and count(config=config) > 0


def build_index(
    documents: Sequence[Document] | None = None,
    *,
    rebuild: bool = False,
    config: Settings = settings,
) -> VectorStore:
    """Embed chunks and persist them to ChromaDB.

    Args:
        documents: Pre-split chunks. Loaded from the configured PDF if omitted.
        rebuild: Delete the existing persist directory before indexing.
        config: Settings instance.

    Returns:
        The populated vector store.
    """
    if rebuild and config.chroma_dir.exists():
        logger.warning("Rebuild requested -- removing %s", config.chroma_dir)
        shutil.rmtree(config.chroma_dir)

    chunks = list(documents) if documents is not None else load_and_split(config=config)
    if not chunks:
        raise ValueError("No chunks to index; check the source PDF and loader output.")

    store = connect(config=config)
    ids = [_deterministic_id(chunk) for chunk in chunks]
    payload = [
        Document(page_content=chunk.page_content, metadata=_sanitize_metadata(chunk))
        for chunk in chunks
    ]

    logger.info("Embedding and indexing %d chunks into '%s'...", len(payload), config.collection_name)
    # Batch the writes: Chroma's SQLite backend has a parameter limit that
    # large single add_documents calls can exceed.
    batch_size = 128
    for start in range(0, len(payload), batch_size):
        store.add_documents(
            documents=payload[start : start + batch_size],
            ids=ids[start : start + batch_size],
        )

    total = count(store, config)
    logger.info("Index ready: %d vectors in %s", total, config.chroma_dir)
    return store


def get_vectorstore(
    *, auto_build: bool = True, config: Settings = settings
) -> VectorStore:
    """Return a queryable store, ingesting the PDF first if the index is empty."""
    if not is_indexed(config):
        if not auto_build:
            raise RuntimeError(
                f"No index found at {config.chroma_dir}. Run `python -m src.vectorstore --build` first."
            )
        logger.info("No existing index found -- running ingestion.")
        return build_index(config=config)
    return connect(config=config)


def get_retriever(
    store: VectorStore | None = None,
    *,
    k: int | None = None,
    search_type: str | None = None,
    config: Settings = settings,
) -> VectorStoreRetriever:
    """Build a retriever with MMR or plain similarity search.

    MMR (maximal marginal relevance) is the default: the financial summary
    repeats near-identical table rows across quarterly and annual pages, and
    plain top-k similarity tends to return five copies of the same row.
    """
    target = store if store is not None else get_vectorstore(config=config)
    effective_k = k or config.retriever_k
    effective_type = search_type or config.search_type

    if effective_type == "mmr":
        search_kwargs: dict[str, Any] = {
            "k": effective_k,
            "fetch_k": max(config.mmr_fetch_k, effective_k * 2),
            "lambda_mult": config.mmr_lambda,
        }
    else:
        search_kwargs = {"k": effective_k}

    return target.as_retriever(search_type=effective_type, search_kwargs=search_kwargs)


def index_stats(config: Settings = settings) -> dict[str, Any]:
    """Summary of the persisted index, for /health and the UI sidebar."""
    total = count(config=config)
    stats: dict[str, Any] = {
        "indexed": total > 0,
        "vector_count": total,
        "collection": config.collection_name,
        "persist_directory": str(config.chroma_dir),
        "distance_metric": config.distance_metric,
    }
    if total:
        try:
            store = connect(config=config)
            metadatas = store.get(include=["metadatas"])["metadatas"]  # type: ignore[attr-defined]
            pages = {m.get("page_number") for m in metadatas if m and m.get("page_number")}
            sections = sorted({m.get("section") for m in metadatas if m and m.get("section")})
            stats["pages_indexed"] = len(pages)
            stats["sections"] = sections
            stats["tabular_chunks"] = sum(1 for m in metadatas if m and m.get("is_tabular"))
        except Exception as exc:  # noqa: BLE001 - stats are best-effort
            logger.debug("Could not compute detailed stats: %s", exc)
    return stats


def _main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="Manage the Tesla RAG ChromaDB index.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="Index the PDF (upserts into any existing index).")
    group.add_argument("--rebuild", action="store_true", help="Delete the index and re-ingest from scratch.")
    group.add_argument("--stats", action="store_true", help="Print index statistics and exit.")
    group.add_argument("--query", type=str, help="Run a retrieval-only smoke test.")
    parser.add_argument("-k", type=int, default=None, help="Number of chunks to retrieve with --query.")
    args = parser.parse_args()

    use_utf8_stdout()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.stats:
        import json

        print(json.dumps(index_stats(), indent=2))
        return

    if args.build or args.rebuild:
        build_index(rebuild=args.rebuild)
        import json

        print(json.dumps(index_stats(), indent=2))
        return

    if args.query:
        retriever = get_retriever(k=args.k)
        for doc in retriever.invoke(args.query):
            meta = doc.metadata
            print(f"\n[page {meta.get('page_number')} | {meta.get('section')} | {meta.get('chunk_id')}]")
            print(doc.page_content[:500])


if __name__ == "__main__":  # pragma: no cover
    _main()
