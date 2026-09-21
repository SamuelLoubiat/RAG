*This project has been created as part of the 42 curriculum by sloubiat.*

# RAG against the machine

## Description

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline
in Python that answers natural-language questions about the [vLLM]
codebase (a large, real-world mix of Python source and Markdown/RST
documentation). Given a question, the system retrieves the most relevant
chunks of code or documentation from the indexed repository and uses a
local LLM to generate a grounded, sourced answer.

The pipeline covers the full loop: ingesting and chunking the codebase,
building a lexical (BM25) index, retrieving the top-k relevant chunks for
a query, generating an answer from that context with **Qwen/Qwen3-0.6B**,
and evaluating retrieval quality with a recall@k metric.

[vLLM]: https://github.com/vllm-project/vllm

## Instructions

### Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) as the package manager
- The vLLM corpus extracted under `data/raw/vllm-0.10.1/`

### Install

```bash
uv sync
```

This installs every dependency, including a **CPU-only** build of PyTorch
(pinned via `[tool.uv.sources]`/`[[tool.uv.index]]` in `pyproject.toml`) —
Qwen3-0.6B is small enough to run comfortably on CPU, and a GPU build would
otherwise pull several extra gigabytes of CUDA packages for nothing.

### Run

All commands go through the CLI, exposed as a Python module:

```bash
uv run python -m src <command> [options]
```

See [Example usage](#example-usage) below for the full command list.

## Resources

- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance
  Framework: BM25 and Beyond.* Foundations and Trends in Information
  Retrieval.
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks.* NeurIPS.
- [`bm25s`](https://github.com/xhluca/bm25s) — the lexical retrieval
  library used for indexing and search.
- [Qwen3 model family](https://qwenlm.github.io/blog/qwen3/) — the
  `Qwen/Qwen3-0.6B` model used for answer generation.
- Python [`ast`](https://docs.python.org/3/library/ast.html) module
  documentation — used for structure-aware Python chunking.
- [Pydantic](https://docs.pydantic.dev/) and
  [Python Fire](https://github.com/google/python-fire) documentation.

**AI usage**: This project was built with the assistance of Claude Code
(Anthropic), used throughout the implementation as a pair-programming
tool: designing and writing the AST-based Python chunker and the
paragraph-based Markdown chunker, working out the exact `bm25s` API for
persisting and reloading per-chunk metadata, implementing the retrieval,
answer-generation and evaluation modules, debugging integration issues
(e.g. `bm25s` crashing on an empty corpus or on `k` larger than the
corpus size, `transformers`' chat template returning a `BatchEncoding`
instead of a bare tensor, `torch` pulling in an unwanted CUDA build by
default), running the CLI end-to-end against the real vLLM corpus to
verify behavior, and drafting this README. All generated code was
reviewed, tested against the real corpus, and adjusted before being kept.

## System architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  data/raw/  │ --> │   chunking   │ --> │  BM25 index   │
│  (vLLM src) │     │ (src/chunking│     │ (data/        │
└─────────────┘     │      .py)    │     │  processed/)  │
                     └──────────────┘     └───────┬───────┘
                                                    │
                                            src/search.py
                                                    │
                                                    v
question ──────────────────────────────>  top-k MinimalSource
                                                    │
                                          src/generation.py
                                        (Qwen/Qwen3-0.6B, reads
                                         chunk text back from disk)
                                                    │
                                                    v
                                              grounded answer

              src/evaluation.py: recall@k of retrieved sources
              against AnsweredQuestions ground truth (own use only)
```

- `src/models.py` — the pydantic data contracts exchanged between every
  stage (`MinimalSource`, `RagDataset`, `StudentSearchResults`, ...).
- `src/chunking.py` — splits raw files into bounded-size chunks.
- `src/indexer.py` — walks `data/raw/`, chunks every file, builds and
  persists the BM25 index under `data/processed/`.
- `src/search.py` — loads the index and retrieves the top-k chunks for a
  query.
- `src/generation.py` — turns retrieved chunks into a grounded answer
  with Qwen3-0.6B.
- `src/evaluation.py` — computes recall@k against ground-truth sources,
  for iteration purposes only.
- `src/cli.py` (+ `src/__main__.py`) — Python Fire CLI wiring all of the
  above together: `index`, `search`, `search_dataset`, `answer`,
  `answer_dataset`, `evaluate`.

## Chunking strategy

Two chunkers are implemented in `src/chunking.py`, both bounded by a
configurable `max_chunk_size` (2000 characters by default):

- **Python (`chunk_python_source`)** — parses the file with the `ast`
  module. Each top-level function or class definition (decorators
  included) becomes its own chunk; other top-level statements are grouped
  together. Any chunk that would exceed `max_chunk_size` is hard-split
  into smaller pieces. If the file has invalid syntax, the chunker falls
  back gracefully to a plain hard-split of the raw text instead of
  crashing.
- **Markdown/text (`chunk_markdown_source`)** — splits on blank-line
  separated paragraphs. A Markdown header (`#` through `######`) always
  forces a new chunk boundary, keeping sections coherent. Oversized
  paragraphs are hard-split the same way as the Python chunker.

This structure-aware approach keeps semantically meaningful units (a
whole function, a whole doc section) together whenever they fit, instead
of chunking on a fixed character/token window that could cut a function
signature away from its body.

## Retrieval method

Retrieval uses **BM25** (via the `bm25s` library) rather than TF-IDF: BM25's
term-frequency saturation and length normalization handle the vLLM
corpus's mix of short docstrings, long source files and prose
documentation better than raw TF-IDF would.

`src/indexer.py::build_index` tokenizes every chunk and builds the index;
`src/search.py::search` tokenizes the query the same way and retrieves
the top-k chunks by BM25 score, returning `MinimalSource` objects (file
path + character range) reconstructed from the persisted index metadata.
The index is saved to `data/processed/` so it never needs to be rebuilt
between runs.

## Performance analysis

Measured on the real vLLM 0.10.1 corpus (1969 indexable files, 18733
chunks):

| Metric | Result | Requirement |
|---|---|---|
| Indexing time | 3.4s | < 5 min |
| Throughput (200 questions, `search_dataset`) | 0.38s | < 90s |
| Recall@5 — docs questions | 82.0% | ≥ 80% |
| Recall@5 — code questions | 50.5% | ≥ 50% |

Recall was measured with `evaluate` against the provided ground-truth
datasets (`data/datasets/AnsweredQuestions/dataset_docs_public.json`,
100 questions, and `dataset_code_public.json`, 99 questions), retrieving
with `k=10`:

| | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| docs | 63.0% | 79.0% | 82.0% | 89.0% |
| code | 26.3% | 41.4% | 50.5% | — |

Both indexing and retrieval are far faster than required — BM25 over a
corpus this size is cheap. Recall is comfortably above threshold on
documentation questions but only marginally above threshold on code
questions (+0.5 point): code questions are harder for a purely lexical
retriever, since the answer's wording rarely matches identifier names or
code structure verbatim. This is the main quality lever left if more time
were available (see [Design decisions](#design-decisions)).

## Design decisions

- **BM25 over TF-IDF / embeddings**: chosen as the mandatory baseline
  retrieval method for its strong out-of-the-box performance on
  heterogeneous corpora without needing a trained embedding model or GPU.
- **AST-based chunking for Python**: keeps functions and classes whole
  instead of chunking on a fixed window, which matters for code questions
  where the answer is usually a whole function's worth of context.
- **CPU-only PyTorch, pinned explicitly**: `torch` resolves to a CUDA
  build by default, downloading gigabytes of `nvidia-*` packages that
  Qwen3-0.6B doesn't need and that can fill a constrained disk. Pinning
  the CPU wheel index in `pyproject.toml` keeps `uv sync` fast, small, and
  reproducible on any grading machine, GPU or not.
- **Re-reading chunk text from disk at generation/evaluation time**:
  `MinimalSource` only stores a file path and a character range, not the
  chunk's text. Rather than duplicating content into the index, the
  actual text is read back from `data/raw/` when it's needed (answer
  generation) — smaller index, single source of truth for content.
  Trade-off: `data/raw/` must still be present on disk for `answer`,
  `answer_dataset`, and `evaluate` to work.
- **`enable_thinking=False` for Qwen3**: Qwen3 defaults to emitting a
  `<think>...</think>` reasoning block before its answer. Disabling it
  keeps the generated JSON focused on the actual answer instead of
  leaking chain-of-thought noise into `MinimalAnswer.answer`.
- **Every CLI command fails gracefully**: an empty query, `k=0`, `k`
  larger than the corpus, a missing index or dataset file, malformed
  JSON, a missing model, or a dataset with no ground truth all produce a
  clear message instead of an unhandled traceback — required by the
  subject, and generally useful when iterating by hand.

## Challenges faced

- **`bm25s` edge cases**: the library raises an unhelpful `ValueError`
  when indexing an empty corpus, and again when retrieving with `k`
  larger than the corpus size. Both are guarded explicitly in
  `src/indexer.py` and `src/search.py` with clear error messages instead
  of letting the library's raw exception surface.
- **`transformers`' chat template return type**: calling
  `tokenizer.apply_chat_template(..., return_tensors="pt")` for Qwen3
  returns a `BatchEncoding` (a dict with `input_ids`/`attention_mask`),
  not a bare tensor — passing it straight to `model.generate()` as a
  positional argument silently mis-parses it. Fixed by requesting the
  templated prompt as plain text (`tokenize=False`) and tokenizing it
  explicitly, then unpacking the encoding into `model.generate(**encoded, ...)`.
- **`torch`'s default CUDA build filling the disk**: on this development
  machine (an 18GB disk, often nearly full), a plain `torch` dependency
  resolved to a CUDA build pulling in ~2GB of `nvidia-*`/`cuda-*`
  packages and repeatedly exhausted available disk space — including once
  after the fix had already been applied, when an unrelated `uv run`
  invocation silently re-triggered a resync that reintroduced the CUDA
  build. Solved by pinning an explicit CPU wheel index for `torch` in
  `pyproject.toml` and re-verifying `torch.version.cuda is None` after
  every dependency change.
- **Recall@5 margin on code questions**: getting code-question recall
  above the 50% threshold took tuning the chunking granularity — the
  final margin is only +0.5 point, so this metric is worth re-checking
  after any future change to `src/chunking.py` or `max_chunk_size`.

## Example usage

```bash
# Build the BM25 index from the raw corpus
uv run python -m src index --max_chunk_size 2000

# Search for the top-5 most relevant chunks for a single question
uv run python -m src search "What is a KV cache?" -k 5

# Batch search over a dataset of questions
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/AnsweredQuestions

# Generate a single grounded answer
uv run python -m src answer "How does automatic prefix caching work?" -k 5

# Batch-generate answers from a StudentSearchResults file
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/AnsweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/AnsweredQuestions

# Evaluate recall@k against ground-truth sources (own use only)
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/AnsweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```
