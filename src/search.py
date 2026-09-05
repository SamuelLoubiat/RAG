"""Query the persisted BM25 index for the most relevant sources."""

from pathlib import Path
from typing import List, Optional

import bm25s

from src.models import MinimalSource


def load_retriever(index_dir: str = "data/processed") -> bm25s.BM25:
    """Load the BM25 index and its associated source metadata.

    Raises FileNotFoundError if no index was built yet under ``index_dir``.
    """
    if not Path(index_dir).exists():
        raise FileNotFoundError(
            f"No index found under '{index_dir}'. Run the `index` "
            "command first."
        )
    return bm25s.BM25.load(index_dir, load_corpus=True)


def search(
    query: str,
    k: int = 10,
    retriever: Optional[bm25s.BM25] = None,
    index_dir: str = "data/processed",
) -> List[MinimalSource]:
    """Return the top-k most relevant sources for ``query``.

    Returns an empty list for a blank query, a non-positive ``k``, or an
    empty index, instead of raising.
    """
    if not query or not query.strip() or k <= 0:
        return []

    if retriever is None:
        retriever = load_retriever(index_dir)

    corpus_size = len(retriever.corpus) if retriever.corpus else 0
    top_k = min(k, corpus_size)
    if top_k <= 0:
        return []

    query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
    results, _ = retriever.retrieve(query_tokens, k=top_k, show_progress=False)
    return [MinimalSource(**doc) for doc in results[0]]
