"""Chunking strategies that split source files into indexable spans.

Each strategy returns a list of ``(first_character_index,
last_character_index)`` spans over the original file content, ready to
be paired with a ``file_path`` to build a ``MinimalSource``.
"""

import ast
import re
from typing import List, Optional, Tuple

Span = Tuple[int, int]

_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s+\S")


def _line_offsets(source: str) -> List[int]:
    """Return the character offset at which each source line starts."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_span(
    node: ast.stmt, line_offsets: List[int], source_len: int
) -> Span:
    """Return the char span of a top-level statement, decorators included."""
    start_line = node.lineno
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        start_line = decorators[0].lineno
    end_line = getattr(node, "end_lineno", node.lineno)
    start = line_offsets[start_line - 1]
    if end_line < len(line_offsets):
        end = line_offsets[end_line]
    else:
        end = source_len
    return start, end


def _split_oversized(
    start: int, end: int, source: str, max_chunk_size: int
) -> List[Span]:
    """Hard-split a span wider than ``max_chunk_size`` on line boundaries."""
    spans: List[Span] = []
    cursor = start
    while cursor < end:
        limit = min(cursor + max_chunk_size, end)
        if limit < end:
            newline = source.rfind("\n", cursor, limit)
            if newline > cursor:
                limit = newline + 1
        spans.append((cursor, limit))
        cursor = limit
    return spans


def chunk_python_source(source: str, max_chunk_size: int = 2000) -> List[Span]:
    """Split Python source code into character spans.

    Each top-level function or class definition becomes its own chunk
    (decorators included). Other top-level statements (imports, module
    docstring, constants...) are grouped together up to ``max_chunk_size``.
    Any resulting chunk still wider than ``max_chunk_size`` is hard-split
    on line boundaries. Falls back to a plain hard split if the file does
    not parse as valid Python.
    """
    if not source:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _split_oversized(0, len(source), source, max_chunk_size)

    line_offsets = _line_offsets(source)
    spans: List[Span] = []
    pending: Optional[Span] = None

    def_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in tree.body:
        start, end = _node_span(node, line_offsets, len(source))
        if isinstance(node, def_types):
            if pending is not None:
                spans.append(pending)
                pending = None
            spans.append((start, end))
        elif pending is None:
            pending = (start, end)
        elif end - pending[0] <= max_chunk_size:
            pending = (pending[0], end)
        else:
            spans.append(pending)
            pending = (start, end)

    if pending is not None:
        spans.append(pending)

    chunks: List[Span] = []
    for start, end in spans:
        if end - start > max_chunk_size:
            chunks.extend(_split_oversized(start, end, source, max_chunk_size))
        else:
            chunks.append((start, end))
    return chunks


def _is_markdown_header(text: str) -> bool:
    """Return True if a paragraph starts with a Markdown ATX heading."""
    return bool(_MARKDOWN_HEADER_RE.match(text.lstrip()))


def _split_into_paragraphs(source: str) -> List[Span]:
    """Split text into paragraph spans, separated by blank lines."""
    spans: List[Span] = []
    offset = 0
    para_start: Optional[int] = None
    for line in source.splitlines(keepends=True):
        if line.strip():
            if para_start is None:
                para_start = offset
        elif para_start is not None:
            spans.append((para_start, offset))
            para_start = None
        offset += len(line)
    if para_start is not None:
        spans.append((para_start, offset))
    return spans


def chunk_markdown_source(
    source: str, max_chunk_size: int = 2000
) -> List[Span]:
    """Split Markdown or plain text into character spans.

    Consecutive paragraphs are grouped together up to ``max_chunk_size``.
    A Markdown heading (``#`` to ``######``) always starts a new chunk,
    so a section's heading stays attached to the content that follows
    it instead of trailing the previous section. Any resulting chunk
    still wider than ``max_chunk_size`` is hard-split on line
    boundaries.
    """
    if not source:
        return []

    paragraphs = _split_into_paragraphs(source)
    spans: List[Span] = []
    pending: Optional[Span] = None

    for start, end in paragraphs:
        is_header = _is_markdown_header(source[start:end])
        if pending is None:
            pending = (start, end)
        elif is_header or (end - pending[0] > max_chunk_size):
            spans.append(pending)
            pending = (start, end)
        else:
            pending = (pending[0], end)

    if pending is not None:
        spans.append(pending)

    chunks: List[Span] = []
    for start, end in spans:
        if end - start > max_chunk_size:
            chunks.extend(_split_oversized(start, end, source, max_chunk_size))
        else:
            chunks.append((start, end))
    return chunks
