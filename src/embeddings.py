"""Local HuggingFace sentence-transformer embeddings.

Embeddings run entirely on-device: no document text leaves the machine and
no API key is required. ``all-MiniLM-L6-v2`` produces 384-dimensional
vectors and is normalised so cosine similarity reduces to a dot product.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from .config import Settings, settings

logger = logging.getLogger(__name__)


def _import_hf_embeddings():
    """Import HuggingFaceEmbeddings from whichever package provides it.

    ``langchain-huggingface`` is the maintained home; the deprecated
    ``langchain_community`` copy is the fallback for older installs.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

        return HuggingFaceEmbeddings
    except ImportError:  # pragma: no cover - depends on installed extras
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

        logger.warning(
            "langchain-huggingface not installed; falling back to the deprecated "
            "langchain_community implementation."
        )
        return HuggingFaceEmbeddings


def build_embeddings(config: Settings = settings) -> Embeddings:
    """Instantiate the embedding model described by ``config``.

    The first call downloads the model (~90 MB) into the HuggingFace cache;
    subsequent calls load from disk and work offline.
    """
    HuggingFaceEmbeddings = _import_hf_embeddings()
    logger.info(
        "Loading embedding model %s on %s", config.embedding_model_name, config.embedding_device
    )
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model_name,
        model_kwargs={"device": config.embedding_device},
        encode_kwargs={
            "normalize_embeddings": config.normalize_embeddings,
            "batch_size": config.embedding_batch_size,
        },
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Process-wide singleton.

    Streamlit reruns the whole script on every interaction and FastAPI serves
    concurrent requests; without caching, each would reload the transformer
    weights and dominate latency.
    """
    return build_embeddings()


def embedding_dimension(embeddings: Embeddings | None = None) -> int:
    """Return the vector width by embedding a probe string."""
    model = embeddings if embeddings is not None else get_embeddings()
    return len(model.embed_query("dimension probe"))


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    model = get_embeddings()
    print(f"model={settings.embedding_model_name} dim={embedding_dimension(model)}")
