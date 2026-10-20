"""Identifier-aware text expansion for BM25 tokenization.

``bm25s`` tokenizes on word boundaries only, so a code identifier like
``get_kv_cache_size`` or ``kvCacheSize`` stays a single opaque token and
never matches a natural-language query such as "What is a KV cache?"
(tokens ``kv``, ``cache``). Expanding identifiers into their snake_case/
camelCase subwords — alongside the original text, not instead of it —
lets BM25 match both the verbatim identifier and its natural-language
subwords.
"""

import re

# bm25s' built-in English stopword list, shared by indexing and search so
# both sides tokenize identically.
STOPWORDS = "en"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)


def _split_identifier(identifier: str) -> str:
    """Split one snake_case/camelCase identifier into space-separated
    subwords.
    """
    subwords = [
        subword
        for part in identifier.split("_")
        if part
        for subword in _CAMEL_BOUNDARY.split(part)
    ]
    return " ".join(subwords)


def expand_identifiers(text: str) -> str:
    """Append split subwords for every multi-part identifier in ``text``.

    Identifiers with no ``_`` or case boundary are left out since
    splitting them would produce no new token.
    """
    extra_lines = []
    for match in _IDENTIFIER.finditer(text):
        identifier = match.group()
        split = _split_identifier(identifier)
        if split.lower() != identifier.lower():
            extra_lines.append(split)

    if not extra_lines:
        return text
    return text + "\n" + "\n".join(extra_lines)


_PATH_SEPARATORS = re.compile(r"[/\\_.\-]+")


def path_context(file_path: str, depth: int = 2) -> str:
    """Return the last ``depth`` components of ``file_path`` as words.

    Many files carry their topic in their path rather than their content
    (``docs/cli/bench/latency.md`` is little more than mkdocs includes),
    so indexing ``bench latency`` alongside each chunk lets BM25 match
    such files at all.
    """
    parts = file_path.replace("\\", "/").split("/")[-depth:]
    stem = "/".join(parts).rsplit(".", 1)[0]
    return _PATH_SEPARATORS.sub(" ", stem).strip()
