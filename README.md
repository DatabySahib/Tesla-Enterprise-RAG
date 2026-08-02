# ⚡ Tesla Q4-2025 — Enterprise RAG Pipeline

Production-grade Retrieval-Augmented Generation over Tesla's **Q4 & FY 2025 Shareholder Update**.
Ask financial, operational and technical questions in natural language and get answers grounded
in the source PDF, cited to the exact page, with a retrieval-confidence score attached.

Runs **fully offline** — embeddings are computed locally with `all-MiniLM-L6-v2` and stored in a
local ChromaDB. No API key is required; no document text leaves the machine.

---

## Architecture

```
                       ┌──────────────────────────────┐
                       │  data/TSLA-Q4-2025-Update.pdf │
                       └───────────────┬──────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
   INGEST           │  src/loader.py                       │
   (one-time)       │  · PyPDFLoader → 1 Document / page   │
                    │  · header/hyphenation normalisation  │
                    │  · RecursiveCharacterTextSplitter    │
                    │    800 chars / 150 overlap           │
                    │  · table-aware separators            │
                    │  · metadata: page, section, tabular  │
                    └──────────────────┬──────────────────┘
                                       │  chunks
                    ┌──────────────────▼──────────────────┐
                    │  src/embeddings.py                   │
                    │  all-MiniLM-L6-v2 · 384-dim · local  │
                    └──────────────────┬──────────────────┘
                                       │  vectors
                    ┌──────────────────▼──────────────────┐
                    │  src/vectorstore.py                  │
                    │  ChromaDB · cosine · ./chroma_db     │
                    │  deterministic IDs → idempotent      │
                    └──────────────────┬──────────────────┘
                                       │
   QUERY            ┌──────────────────▼──────────────────┐
   (per request)    │  src/rag_engine.py                   │
                    │  · MMR retrieval (k, fetch_k, λ)     │
                    │  · analyst system prompt             │
                    │  · LLM: openai → ollama → hf →       │
                    │         extractive fallback          │
                    │  · citations + confidence scoring    │
                    └────────┬────────────────────┬────────┘
                             │                    │
                  ┌──────────▼─────────┐  ┌───────▼──────────┐
                  │  app.py            │  │  api.py          │
                  │  Streamlit chat    │  │  FastAPI REST    │
                  │  source previews   │  │  /query /health  │
                  └────────────────────┘  └──────────────────┘
```

### Repository layout

```
tesla-rag-enterprise/
├── data/
│   └── TSLA-Q4-2025-Update.pdf   # Source document
├── chroma_db/                    # Vector store (generated, git-ignored)
├── src/
│   ├── __init__.py
│   ├── config.py                 # Env-overridable settings + validation
│   ├── loader.py                 # PDF load & financial-aware splitting
│   ├── embeddings.py             # Local HuggingFace embeddings (cached)
│   ├── vectorstore.py            # ChromaDB index & retriever factory
│   └── rag_engine.py             # QA chain, citations, confidence
├── app.py                        # Streamlit chat dashboard
├── api.py                        # FastAPI REST service
├── requirements.txt              # Pinned dependencies
├── .env.example                  # Configuration reference
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+ and ~1.5 GB of disk for dependencies and the embedding model.

```bash
python -m venv .venv
```

Activate it — Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies (the extra index pulls CPU-only PyTorch, ~200 MB instead of ~2.5 GB):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Optionally copy the configuration reference and edit it:

```bash
cp .env.example .env
```

### Build the index

```bash
python -m src.vectorstore --build
```

First run downloads the MiniLM model (~90 MB) and embeds the document. Subsequent runs are
offline. Use `--rebuild` to discard the existing index and start clean, `--stats` to inspect it.

### Run the UI

```bash
streamlit run app.py
```

### Run the API

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Interactive OpenAPI docs at <http://127.0.0.1:8000/docs>.

---

## Usage

### Command line

Run the blueprint's three verification queries:

```bash
python -m src.rag_engine
```

Ask a single question, as JSON:

```bash
python -m src.rag_engine "What was Tesla's free cash flow in Q4 2025?" --json
```

Retrieval-only smoke test (no LLM, shows exactly which chunks match):

```bash
python -m src.vectorstore --query "Optimus Gen 3 production capacity" -k 3
```

### REST API

```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"question\": \"What was Tesla's total gross profit in Q4 2025?\", \"k\": 5}"
```

Response shape (shown with a generative backend configured; in the default extractive mode
`answer` contains the ranked passages instead of synthesised prose):

```json
{
  "question": "What was Tesla's total gross profit in Q4 2025?",
  "answer": "Total gross profit was $5,009 million in Q4 2025, up 20% year-over-year [p. 4].",
  "sources": [
    {
      "page_number": 4,
      "section": "Financial Summary",
      "chunk_id": "p004-c0012",
      "snippet": "Total gross profit 4,179 3,153 3,878 5,054 5,009 20% …",
      "relevance": 0.71,
      "is_tabular": true
    }
  ],
  "confidence": 0.66,
  "confidence_label": "high",
  "pages_cited": [4, 5],
  "backend": "openai",
  "model": "gpt-4o-mini",
  "chunks_retrieved": 5,
  "latency_seconds": 0.42
}
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness, index statistics, active backend |
| `/query` | POST | Answer a question with citations and confidence |
| `/config` | GET | Effective pipeline configuration |
| `/examples` | GET | The blueprint's verification queries |
| `/reindex` | POST | Re-ingest the PDF (`?rebuild=true` to wipe first) |

---

## Verification queries

| Category | Query | Target section |
|---|---|---|
| Financial Metrics | What was Tesla's total gross profit and operating margin in Q4 2025? | Financial Summary (p. 4) |
| AI & Silicon | What are the key specifications and compute gains of the new AI5 inference chip? | AI & Software (p. 10) |
| Robotics & Fleet | What is the planned timeline and capacity for the Optimus Gen 3 production line? | Manufacturing & Hardware (p. 8) |

Verified against the built index (89 chunks, 34 pages, `k=5`, MMR): each query returns its
target section as the **top-ranked** passage.

| Query | Top hit | Relevance | Confidence |
|---|---|---|---|
| Financial Metrics | p. 4 — Financial Summary | 0.62 | high (0.61) |
| AI & Silicon | p. 10 — AI & Software | 0.64 | medium (0.59) |
| Robotics & Fleet | p. 8 — Manufacturing & Hardware | 0.67 | high (0.62) |

Ground truth from the source document, for checking retrieval quality:

- **p. 4** — Q4-2025 total gross profit **$5,009M**, GAAP gross margin **20.1%**,
  operating margin **5.7%**, income from operations **$1,409M**.
- **p. 10** — **AI5** (the blueprint calls it "A15"; the document says AI5) targets a
  **50× performance improvement over AI4**, via **10× raw compute**, **9× memory capacity**
  and **5× hardened block quantization/softmax**. Production planned for **2027** (AI6 in 2028).
- **p. 8** — **Optimus Gen 3** unveil planned for **Q1 2026**, start of production
  **before the end of 2026**, eventual planned capacity **1 million robots per year**.

> **Note on the blueprint's wording:** the specification sheet refers to an "A15 inference chip".
> The source document consistently names it **AI5**. The code and examples use AI5.

Other questions the corpus answers well: Q4 energy storage deployments (14.2 GWh, p. 6),
FY2025 free cash flow ($6.2B, p. 5), Robotaxi metro rollout timeline (p. 11), Cortex 2 AI
training compute (p. 9), installed manufacturing capacity by region (p. 8).

---

## Design decisions

**Table-aware splitting.** The financial summaries encode a full year of data per line
(`Total revenues 25,707 19,335 22,496 28,095 24,901 -3%`). The splitter's separator list
prioritises `\n\n` and `\n` and reaches `" "` only as a last resort, so a table row is never
sheared mid-row. Chunks are flagged `is_tabular` by a numeric-density heuristic; the prompt
then tells the model the column order so it does not read a Q1 figure as a Q4 one.

**Letter-spaced header repair.** Section titles are typeset as `F I N A N C I A L   S U M M A R Y`.
Left alone they tokenise into meaningless single characters. `loader.py` collapses them, which
makes section names both searchable and useful as chunk metadata.

**MMR by default.** The quarterly (p. 4) and annual (p. 5) summary tables contain near-identical
rows. Plain top-k similarity returns five variants of the same row and starves the model of
context; maximal marginal relevance diversifies across pages and sections.

**Confidence from embeddings, not the retriever.** Chroma reports distances on different scales
across versions and MMR surfaces no score at all. `score_documents` instead re-embeds the query
and the retrieved chunks and computes cosine directly, so the number means the same thing under
any configuration. The reported score blends the best hit (70%) with the retrieval mean (30%) —
a set where everything is weakly related is a signal the question may be out of scope.

**Deterministic chunk IDs.** IDs are a SHA-256 of source + page + offset + content, so re-running
ingestion upserts instead of duplicating.

**Graceful backend degradation.** With no LLM configured the engine does not fail — it returns the
ranked source passages verbatim (`extractive` mode). The pipeline stays demonstrable on any
machine, and this mode cannot hallucinate.

---

## Configuration

All settings are environment variables with sensible defaults; see [`.env.example`](.env.example)
for the annotated list. Key values:

| Variable | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Blueprint specification |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, local |
| `RETRIEVER_K` | `5` | Passages fed to the model |
| `SEARCH_TYPE` | `mmr` | `mmr` or `similarity` |
| `LLM_BACKEND` | `auto` | `auto` tries openai → ollama → extractive |
| `CHROMA_DIR` | `chroma_db` | Relative paths resolve against the project root |

### Enabling generated answers

The default install runs in extractive mode. To get synthesised prose:

**OpenAI** — `pip install langchain-openai`, then set `OPENAI_API_KEY` in `.env`.

**Ollama (local, private)** — install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.1
```

and `pip install langchain-ollama`. The engine probes the daemon automatically.

**Local HuggingFace** — set `LLM_BACKEND=huggingface` in `.env`. Downloads `google/flan-t5-base`
(~1 GB) on first use. Fully offline, but noticeably weaker at multi-column table reasoning than
the other two options. This backend is deliberately **excluded from `auto`** so that a default
first run never triggers a large unattended download.

---

## Troubleshooting

**`No index found`** — run `python -m src.vectorstore --build`.

**`numpy.dtype size changed` / binary incompatibility** — a stale `scikit-learn` compiled against
NumPy 1.x. Use the project venv rather than a shared Anaconda environment; that is precisely what
the isolated install above avoids.

**First query is slow** — the embedding model loads on first use (a few seconds). It is cached
process-wide thereafter; Streamlit and FastAPI both load it once at startup.

**Answers cite the wrong quarter** — raise `RETRIEVER_K`, or switch `SEARCH_TYPE=similarity` if
the question targets one specific table row rather than a topic.

---

## Notes and limitations

- **Text-only ingestion.** Charts and photo pages (14–23) carry their data in images; PyPDF
  extracts only their axis labels. Figures that appear solely in a chart are not retrievable.
  Adding OCR or a multimodal extractor would close this gap.
- **No cross-page table reconstruction.** A table spanning a page break is indexed as two
  independent chunks.
- **Not investment advice.** This is a document-retrieval tool. Verify every figure against the
  cited page before relying on it.

---

*Source document: Tesla Q4 & FY 2025 Update, © Tesla, Inc. Used here as sample data for a
document-retrieval demonstration.*
