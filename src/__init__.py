"""Tesla Q4-2025 Enterprise RAG pipeline.

Modular components:
    config      -- centralised, environment-overridable settings
    loader      -- PDF ingestion and financial-aware text splitting
    embeddings  -- local HuggingFace sentence-transformer embeddings
    vectorstore -- persistent ChromaDB index and retriever factory
    rag_engine  -- retrieval QA chain with citations and confidence metadata
"""

__version__ = "1.0.0"

__all__ = ["config", "loader", "embeddings", "vectorstore", "rag_engine"]
