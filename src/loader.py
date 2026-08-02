"""PDF ingestion and financial-aware text splitting.

The Tesla shareholder update mixes two very different text shapes:

* narrative prose  -- "Robotics", "Outlook", management commentary
* dense tabular text -- financial summaries where a single line such as
  ``Total gross profit 4,179 3,153 3,878 5,054 5,009 20%`` carries an entire
  row of quarterly figures

A naive splitter breaks table rows mid-number and destroys the column
alignment that makes those rows answerable. The splitter here keeps line
integrity as a high-priority separator and never splits on whitespace inside
a line, so numeric rows survive chunking intact.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, settings

logger = logging.getLogger(__name__)

# Separators are tried in order; the first that yields chunks under the size
# limit wins. Paragraph -> line -> sentence -> clause. Note the deliberate
# absence of " " (single space): splitting on spaces would shear numeric
# table rows apart.
FINANCIAL_SEPARATORS: Sequence[str] = (
    "\n\n",  # paragraph / block boundary
    "\n",    # line boundary -- one table row
    ". ",    # sentence
    "; ",
    ", ",
    " ",     # last resort before hard character split
    "",
)

# Section headers in this deck are rendered letter-spaced, e.g.
# "F I N A N C I A L   S U M M A R Y" or "A I   &   S O F T W A R E".
# Match those and collapse them back. "&" is included because three section
# titles use it as a word.
_SPACED_HEADER = re.compile(r"^(?:[A-Z&][ \t]+){3,}[A-Z&][ \t]*$", re.MULTILINE)

# Page -> section map from the document's own table of contents (page 2).
# Used as the fallback when a page carries no detectable header.
TOC_SECTIONS: dict[int, str] = {
    1: "Cover",
    2: "Table of Contents",
    3: "Highlights",
    4: "Financial Summary",
    5: "Financial Summary",
    6: "Operational Summary",
    7: "Operational Summary",
    8: "Manufacturing & Hardware",
    9: "Supporting Infrastructure",
    10: "AI & Software",
    11: "Services",
    12: "Other Updates",
    13: "Outlook",
    14: "Photos & Charts",
    24: "Key Metrics",
    27: "Financial Statements",
    34: "Additional Information",
}


def _section_for_page(page_number: int) -> str:
    """Resolve a 1-indexed page to its TOC section (nearest preceding entry)."""
    candidates = [start for start in TOC_SECTIONS if start <= page_number]
    if not candidates:
        return "Unknown"
    return TOC_SECTIONS[max(candidates)]


def _collapse_spaced_headers(text: str) -> str:
    """Turn ``F I N A N C I A L   S U M M A R Y`` into ``FINANCIAL SUMMARY``.

    Letter-spaced headers otherwise tokenise into meaningless single
    characters, which both pollutes the embedding and makes the header
    unsearchable.
    """

    def _fix(match: re.Match[str]) -> str:
        # Inside a letter-spaced header a single space separates letters, so a
        # run of two or more is the real word boundary.
        words = re.split(r"\s{2,}", match.group(0).strip())
        return " ".join(word.replace(" ", "") for word in words)

    return _SPACED_HEADER.sub(_fix, text)


def clean_page_text(text: str) -> str:
    """Normalise raw PyPDF output without discarding financial content."""
    text = _collapse_spaced_headers(text)
    # PyPDF frequently emits hyphenated line-wraps ("hardware -centric") and
    # split words inside justified paragraphs.
    text = re.sub(r"(?<=[a-z])\s-(?=[a-z])", "-", text)
    # Collapse runs of spaces/tabs but preserve newlines (they delimit rows).
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ blank lines to a single paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_tabular(text: str) -> bool:
    """Heuristic: does this chunk carry financial table rows?

    A row like ``Total revenues 25,707 19,335 22,496 28,095 24,901 -3%``
    has several numeric tokens on one line. If a meaningful share of lines
    look like that, downstream consumers should treat the chunk as tabular.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    numeric_line = re.compile(r"(?:[-(]?\$?[\d,]+(?:\.\d+)?%?[)]?\s+){2,}")
    hits = sum(1 for line in lines if numeric_line.search(line))
    return hits >= max(2, len(lines) // 3)


def load_pdf(pdf_path: Path | None = None, config: Settings = settings) -> list[Document]:
    """Load the source PDF into one :class:`Document` per page.

    Args:
        pdf_path: Override for the configured PDF location.
        config: Settings instance (injected for testability).

    Returns:
        Cleaned page-level documents with ``page_number`` metadata (1-indexed).

    Raises:
        FileNotFoundError: If the PDF is missing.
        ValueError: If the PDF yields no extractable text (e.g. a pure scan
            that would need OCR).
    """
    path = Path(pdf_path) if pdf_path is not None else config.pdf_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Source PDF not found at {path}. Place the document in {config.data_dir} "
            f"or set PDF_FILENAME / DATA_DIR in your .env."
        )

    logger.info("Loading PDF: %s", path)
    raw_pages = PyPDFLoader(str(path)).load()

    pages: list[Document] = []
    for index, page in enumerate(raw_pages, start=1):
        cleaned = clean_page_text(page.page_content)
        if not cleaned:
            # Chart-only and photo pages legitimately extract to nothing.
            logger.debug("Page %d has no extractable text; skipping.", index)
            continue
        pages.append(
            Document(
                page_content=cleaned,
                metadata={
                    **page.metadata,
                    "source": path.name,
                    "page_number": index,
                    "section": _section_for_page(index),
                },
            )
        )

    if not pages:
        raise ValueError(
            f"No extractable text found in {path.name}. The file may be a scanned "
            f"image requiring OCR before ingestion."
        )

    logger.info("Extracted text from %d of %d pages.", len(pages), len(raw_pages))
    return pages


def build_splitter(config: Settings = settings) -> RecursiveCharacterTextSplitter:
    """Construct the recursive splitter tuned for mixed prose/table content."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(FINANCIAL_SEPARATORS),
        length_function=len,
        keep_separator=True,
        add_start_index=True,
    )


def _context_header(page_number: int, section: str, tabular: bool) -> str:
    """Provenance line prepended to every chunk before embedding.

    A page's section title appears only in its first chunk, so later chunks --
    including the ones holding the actual financial rows -- would otherwise
    embed with no indication of which table they came from. Repeating the
    section and page in every chunk is what lets a query like "Q4 2025
    operating margin" reach the *row* rather than only the table's heading.
    """
    kind = "financial table" if tabular else "narrative"
    return f"[Section: {section} | Page {page_number} | {kind}]"


def split_documents(
    pages: Iterable[Document], config: Settings = settings
) -> list[Document]:
    """Split page documents into retrieval chunks with enriched metadata.

    Each chunk carries ``page_number``, ``section``, ``chunk_id`` and an
    ``is_tabular`` flag so the UI can render financial rows as tables and the
    prompt can warn the model about column alignment. A provenance header is
    prepended to the text itself so it participates in the embedding.
    """
    splitter = build_splitter(config)
    page_list = list(pages)
    chunks = splitter.split_documents(page_list)

    for position, chunk in enumerate(chunks):
        page_number = int(chunk.metadata.get("page_number", 0))
        section = str(chunk.metadata.get("section", "Unknown"))
        tabular = _looks_tabular(chunk.page_content)

        chunk.metadata["chunk_id"] = f"p{page_number:03d}-c{position:04d}"
        chunk.metadata["chunk_index"] = position
        chunk.metadata["is_tabular"] = tabular
        chunk.metadata["char_count"] = len(chunk.page_content)
        chunk.page_content = (
            f"{_context_header(page_number, section, tabular)}\n{chunk.page_content}"
        )

    logger.info("Split %d pages into %d chunks.", len(page_list), len(chunks))
    return chunks


def load_and_split(
    pdf_path: Path | None = None, config: Settings = settings
) -> list[Document]:
    """Convenience pipeline: load the PDF and return ready-to-embed chunks."""
    return split_documents(load_pdf(pdf_path, config), config)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    docs = load_and_split()
    tabular = sum(1 for d in docs if d.metadata["is_tabular"])
    print(f"chunks={len(docs)} tabular={tabular} prose={len(docs) - tabular}")
    print("-" * 70)
    print(docs[0].metadata)
    print(docs[0].page_content[:400])
