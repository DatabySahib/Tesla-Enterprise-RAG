"""FastAPI service exposing the Tesla RAG engine for external integration.

Run with::

    uvicorn api:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health   -- liveness, index statistics and active backend
    POST /query    -- answer a question with citations and confidence
    GET  /config   -- effective pipeline configuration
    POST /reindex  -- rebuild the vector index from the source PDF
    GET  /docs     -- interactive OpenAPI documentation
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config as config_module
from src.config import settings
from src.rag_engine import VERIFICATION_QUERIES, RAGEngine
from src.vectorstore import build_index, index_stats, is_indexed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tesla_rag.api")

_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    """Return the loaded engine or fail with a clear 503."""
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine is not initialised. Build the index with POST /reindex.",
        )
    return _engine


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model and vector index once at startup.

    Doing this eagerly means the first real request does not pay the
    model-loading cost, and a broken index surfaces in the logs immediately.
    """
    global _engine
    try:
        logger.info("Initialising RAG engine…")
        _engine = RAGEngine(auto_build=True)
        logger.info(
            "Engine ready: backend=%s model=%s vectors=%s",
            _engine.backend,
            _engine.model_name,
            index_stats().get("vector_count"),
        )
    except Exception as exc:  # noqa: BLE001 - stay up so /health can report the fault
        logger.error("Engine initialisation failed: %s", exc)
        _engine = None
    yield
    _engine = None


app = FastAPI(
    title="Tesla Q4-2025 Enterprise RAG API",
    description=(
        "Retrieval-augmented question answering over Tesla's Q4 & FY 2025 Shareholder "
        "Update. Answers are grounded in the source PDF and cited to exact page numbers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Tighten to your own origins before exposing this beyond localhost.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural-language question about the Tesla Q4-2025 update.",
        examples=["What was Tesla's total gross profit and operating margin in Q4 2025?"],
    )
    k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of passages to retrieve for this query.",
    )
    include_full_text: bool = Field(
        default=False,
        description="Return each source passage in full rather than a truncated snippet.",
    )


class SourceResponse(BaseModel):
    page_number: int
    section: str
    chunk_id: str
    snippet: str
    relevance: float
    is_tabular: bool
    full_text: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    confidence: float
    confidence_label: str
    pages_cited: list[int]
    backend: str
    model: str
    chunks_retrieved: int
    latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    index: dict[str, Any]
    backend: str | None = None
    model: str | None = None
    version: str = "1.0.0"


class ReindexResponse(BaseModel):
    status: str
    index: dict[str, Any]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness and readiness probe.

    Returns 200 even when the engine failed to load, so orchestrators can read
    the fault detail from the body rather than a bare connection error.
    """
    ready = _engine is not None
    return HealthResponse(
        status="ok" if ready and is_indexed() else "degraded",
        engine_ready=ready,
        index=index_stats(),
        backend=_engine.backend if _engine else None,
        model=_engine.model_name if _engine else None,
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question against the indexed document, with page citations."""
    engine = get_engine()
    try:
        result = engine.answer(request.question, k=request.k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - unexpected retrieval/store failure
        logger.exception("Query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query failed: {exc}"
        ) from exc

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceResponse(
                page_number=citation.page_number,
                section=citation.section,
                chunk_id=citation.chunk_id,
                snippet=citation.snippet,
                relevance=citation.relevance,
                is_tabular=citation.is_tabular,
                full_text=citation.full_text if request.include_full_text else None,
            )
            for citation in result.sources
        ],
        confidence=result.confidence,
        confidence_label=result.confidence_label,
        pages_cited=result.pages_cited,
        backend=result.backend,
        model=result.model,
        chunks_retrieved=result.chunks_retrieved,
        latency_seconds=result.latency_seconds,
    )


@app.get("/config", tags=["ops"])
def get_config() -> dict[str, Any]:
    """Effective pipeline configuration (no secrets are included)."""
    return config_module.describe()


@app.get("/examples", tags=["rag"])
def examples() -> list[dict[str, str]]:
    """Verification queries from the project blueprint."""
    return VERIFICATION_QUERIES


@app.post("/reindex", response_model=ReindexResponse, tags=["ops"])
def reindex(rebuild: bool = False) -> ReindexResponse:
    """Re-ingest the source PDF.

    Args:
        rebuild: Delete the existing index first instead of upserting.
    """
    global _engine
    try:
        build_index(rebuild=rebuild)
        _engine = RAGEngine(auto_build=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reindex failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Reindex failed: {exc}"
        ) from exc
    return ReindexResponse(status="ok", index=index_stats())


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api:app", host=settings.api_host, port=settings.api_port, reload=False)
