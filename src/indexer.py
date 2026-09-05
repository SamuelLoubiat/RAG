"""Corpus indexing: chunk every file under a raw corpus directory and
persist a BM25 index over the resulting chunks.
"""

from pathlib import Path
from typing import Any, Dict, List

import bm25s
from tqdm import tqdm

from src.chunking import chunk_markdown_source, chunk_python_source
from src.models import MinimalSource

PYTHON_EXTENSIONS = {".py"}
TEXT_EXTENSIONS = {".md", ".rst", ".txt"}
INDEXED_EXTENSIONS = PYTHON_EXTENSIONS | TEXT_EXTENSIONS


def _iter_corpus_files(raw_dir: Path) -> List[Path]:
    """List every file under ``raw_dir`` whose extension we can chunk."""
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix in INDEXED_EXTENSIONS
    )


def _read_text(path: Path) -> str:
    """Read a file as UTF-8 text, replacing any undecodable bytes."""
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk_file(source: str, suffix: str, max_chunk_size: int) -> List[Any]:
    """Return the character spans for a file, picking the strategy by type."""
    if suffix in PYTHON_EXTENSIONS:
        return chunk_python_source(source, max_chunk_size=max_chunk_size)
    return chunk_markdown_source(source, max_chunk_size=max_chunk_size)


def build_index(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    max_chunk_size: int = 2000,
) -> int:
    """Chunk every indexable file under ``raw_dir`` and persist a BM25
    index over the resulting chunks under ``processed_dir``.

    Each indexed chunk is described by a :class:`MinimalSource`
    (``file_path``, ``first_character_index``, ``last_character_index``)
    saved alongside the index so retrieval can rebuild sources directly.

    Returns the number of chunks indexed. Raises ``ValueError`` if no
    indexable file is found under ``raw_dir``.
    """
    files = _iter_corpus_files(Path(raw_dir))

    corpus_texts: List[str] = []
    corpus_metadata: List[Dict[str, Any]] = []

    for path in tqdm(files, desc="Chunking", unit="file"):
        try:
            source = _read_text(path)
        except OSError:
            continue

        spans = _chunk_file(source, path.suffix, max_chunk_size)
        file_path = str(path)
        for start, end in spans:
            text = source[start:end]
            if not text.strip():
                continue
            source_model = MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end,
            )
            corpus_texts.append(text)
            corpus_metadata.append(source_model.model_dump())

    if not corpus_texts:
        raise ValueError(f"No indexable file found under '{raw_dir}'")

    corpus_tokens = bm25s.tokenize(
        corpus_texts, stopwords=None, show_progress=True, leave=False
    )
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True, leave_progress=False)

    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    retriever.save(
        str(processed_path), corpus=corpus_metadata, show_progress=True
    )

    return len(corpus_texts)
