"""Central configuration for the Tesla Enterprise RAG pipeline.

Every value can be overridden through environment variables (or a local
``.env`` file) so the same codebase runs unchanged from a laptop to a
container in a CI/CD pipeline. See ``.env.example`` for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from dotenv import load_dotenv

# Load .env from the project root (parent of src/) if present.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

LLMBackend = Literal["huggingface", "openai", "ollama", "extractive"]


def use_utf8_stdout() -> None:
    """Make CLI output safe on legacy consoles.

    The source document contains characters (``≠``, ``·``, curly quotes) that
    a Windows cp1252 console cannot encode, which otherwise aborts a CLI run
    mid-print with UnicodeEncodeError.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - detached stream
                pass


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    """Read an int from the environment, falling back on malformed input."""
    raw = os.getenv(key)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key)
    if raw in (None, ""):
        return default
    candidate = Path(raw).expanduser()
    # Relative paths resolve against the project root, not the shell's cwd,
    # so `streamlit run app.py` and `uvicorn api:app` agree on the index location.
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the pipeline configuration."""

    # ---- Paths -----------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _env_path("DATA_DIR", PROJECT_ROOT / "data"))
    pdf_filename: str = field(default_factory=lambda: _env_str("PDF_FILENAME", "TSLA-Q4-2025-Update.pdf"))
    chroma_dir: Path = field(default_factory=lambda: _env_path("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))

    # ---- Chunking (blueprint spec: 800 / 150) ----------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))

    # ---- Embeddings ------------------------------------------------------
    embedding_model_name: str = field(
        default_factory=lambda: _env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_device: str = field(default_factory=lambda: _env_str("EMBEDDING_DEVICE", "cpu"))
    normalize_embeddings: bool = field(
        default_factory=lambda: _env_str("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    )
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 32))

    # ---- Vector store ----------------------------------------------------
    collection_name: str = field(default_factory=lambda: _env_str("COLLECTION_NAME", "tesla_q4_2025"))
    # Cosine matches the normalised MiniLM embedding space; Chroma defaults to L2.
    distance_metric: str = field(default_factory=lambda: _env_str("DISTANCE_METRIC", "cosine"))

    # ---- Retrieval -------------------------------------------------------
    retriever_k: int = field(default_factory=lambda: _env_int("RETRIEVER_K", 5))
    search_type: str = field(default_factory=lambda: _env_str("SEARCH_TYPE", "mmr"))
    mmr_fetch_k: int = field(default_factory=lambda: _env_int("MMR_FETCH_K", 20))
    # 0.7 favours relevance over diversity: financial lookups target a specific
    # row, and aggressive diversification pulls in off-topic pages.
    mmr_lambda: float = field(default_factory=lambda: _env_float("MMR_LAMBDA", 0.7))

    # ---- LLM backend -----------------------------------------------------
    llm_backend: str = field(default_factory=lambda: _env_str("LLM_BACKEND", "auto"))
    openai_model: str = field(default_factory=lambda: _env_str("OPENAI_MODEL", "gpt-4o-mini"))
    ollama_model: str = field(default_factory=lambda: _env_str("OLLAMA_MODEL", "llama3.1"))
    ollama_base_url: str = field(default_factory=lambda: _env_str("OLLAMA_BASE_URL", "http://localhost:11434"))
    hf_model: str = field(default_factory=lambda: _env_str("HF_MODEL", "google/flan-t5-base"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 768))

    # ---- API -------------------------------------------------------------
    api_host: str = field(default_factory=lambda: _env_str("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))

    @property
    def pdf_path(self) -> Path:
        """Absolute path to the source PDF document."""
        return self.data_dir / self.pdf_filename

    def validate(self) -> None:
        """Fail fast on configuration that would break the pipeline downstream."""
        if self.chunk_size <= 0:
            raise ValueError(f"CHUNK_SIZE must be positive, got {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP must be in [0, CHUNK_SIZE); got {self.chunk_overlap} with chunk_size={self.chunk_size}"
            )
        if self.retriever_k <= 0:
            raise ValueError(f"RETRIEVER_K must be positive, got {self.retriever_k}")
        if self.search_type not in ("similarity", "mmr"):
            raise ValueError(f"SEARCH_TYPE must be 'similarity' or 'mmr', got {self.search_type!r}")


settings: Final[Settings] = Settings()
settings.validate()


def describe() -> dict[str, object]:
    """Return a JSON-serialisable view of the active settings (for /health and the UI)."""
    return {
        "pdf_path": str(settings.pdf_path),
        "pdf_present": settings.pdf_path.is_file(),
        "chroma_dir": str(settings.chroma_dir),
        "collection_name": settings.collection_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model_name,
        "distance_metric": settings.distance_metric,
        "search_type": settings.search_type,
        "retriever_k": settings.retriever_k,
        "llm_backend": settings.llm_backend,
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import json

    print(json.dumps(describe(), indent=2))
