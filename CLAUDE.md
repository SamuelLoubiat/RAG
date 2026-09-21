# RAG against the machine — project context for Claude

42 school project. Subject: `en.subject.pdf`, **version 2.0** (changed from an
earlier version — do not trust generic assumptions about "RAG project"
requirements, always defer to `en.subject.pdf` / `TODO.md`).

**`TODO.md` is the source of truth for what's done and what's left.** Always
read it first. It's kept up to date with checkboxes as work progresses —
update it whenever a task is completed.

## What this project does

A RAG pipeline in Python answering questions about the vLLM codebase
(`data/raw/vllm-0.10.1/`, 3226 files, 1969 indexable `.py`/`.md`/`.rst`/`.txt`
files, gitignored — never commit it, the grader/moulinette regenerates it).
Pipeline: ingestion → chunking → BM25 indexing → retrieval → answer
generation (Qwen3-0.6B) → evaluation (recall@k).

## Critical constraints

- **The grader/moulinette only runs `uv sync`.** Nothing else. Every
  dependency must be in `pyproject.toml`; `uv.lock` must be committed
  (it's easy to accidentally gitignore it — check `.gitignore`).
- **Never import or call the `moulinette` executable** from project code.
  The official recall@k score at soutenance is computed by the moulinette's
  own `evaluate_student_search_results`, not by our code. Our `evaluate`
  command (section 6, not yet implemented) is for our own iteration only.
- **`data/` is gitignored** — corpus, index, model weights, generated
  outputs never get committed.
- Repo layout expected by reference scripts (`index` → `search_dataset` →
  moulinette `evaluate_student_search_results` → `answer_dataset`):
  ```
  src/                                                  # uv run python -m src <command>
  pyproject.toml / uv.lock / README.md / Makefile / .gitignore
  data/raw/                                             # vLLM corpus
  data/processed/                                       # BM25 index
  data/datasets/UnansweredQuestions/ , AnsweredQuestions/
  data/output/search_results/<DatasetScope>/
  data/output/search_results_and_answer/<DatasetScope>/
  ```

## Known gotcha: torch pulls CUDA by default

Plain `torch` in `pyproject.toml` resolves to a CUDA build (~2GB of
`nvidia-*`/`cuda-*` packages) which has repeatedly filled the disk on this
machine (only ~18G total, often <1G free). Qwen3-0.6B doesn't need a GPU.
Fix that must stay in `pyproject.toml`:
```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```
This config **has spontaneously reverted at least once** (to a
`[tool.uv] torch-backend = "cpu"` variant that doesn't reliably stick — a
`uv run mypy` invocation once silently re-triggered a resync that pulled
the CUDA build back in). If `uv.lock` ever shows
`source = { registry = "https://pypi.org/simple" }` for `torch` instead of
the `pytorch-cpu` index, or `torch.version.cuda` is not `None`, redo:
`rm uv.lock && uv sync`, then verify with
`uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"`
→ expect `2.14.0+cpu None`. If disk is full, check
`du -sh ~/Downloads/* ~/Documents/* 2>/dev/null | sort -rh | head` before
deleting anything — ask the user before removing files you didn't create.

## Architecture / key files

- `src/models.py` — all pydantic models from the subject: `MinimalSource`,
  `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`,
  `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`,
  `StudentSearchResultsAndAnswer`. Don't break these; can extend.
- `src/chunking.py` — `chunk_python_source` (AST-based: function/class =
  one chunk, decorators included, hard-split on oversized/invalid syntax),
  `chunk_markdown_source` (paragraph-based, headers force new chunk).
  Both respect `max_chunk_size` (default 2000 chars) via shared
  `_split_oversized` helper.
- `src/indexer.py` — `build_index(raw_dir, processed_dir, max_chunk_size)`:
  walks the corpus, chunks each file, tokenizes with `bm25s.tokenize`,
  saves the BM25 index + `MinimalSource` metadata (`corpus.jsonl`) under
  `processed_dir`. Raises `ValueError` on an empty corpus (bm25s crashes
  cryptically otherwise).
- `src/search.py` — `load_retriever(index_dir)` (raises
  `FileNotFoundError` if no index), `search(query, k, retriever)` → list
  of `MinimalSource`. Guards: blank query/`k<=0` → `[]`; `k` clamped to
  corpus size (bm25s raises `ValueError` if `k` > corpus size otherwise).
- `src/generation.py` — Qwen3-0.6B via `transformers`. `load_generator`,
  `generate_answer(question, sources, generator, max_new_tokens)`. Reads
  actual chunk text from disk via `MinimalSource.file_path` +
  character indices (models only store indices, not content — always
  re-read from the corpus file). Prompt forces grounding in context only;
  `enable_thinking=False` on `apply_chat_template` to suppress Qwen3's
  `<think>` reasoning noise in the output. `MAX_CONTEXT_CHARS` budgets how
  much retrieved text fits in the prompt (truncates rather than exceeding
  token limits). **Gotcha**: `tokenizer.apply_chat_template(...,
  return_tensors="pt")` returns a `BatchEncoding` (dict-like), not a bare
  `Tensor` — pass `tokenize=False` to get a prompt string, then
  `tokenizer(prompt_text, return_tensors="pt")` and `model.generate(**encoded, ...)`.
- `src/evaluation.py` — own-use-only recall@k (k=1,3,5,10), never the
  moulinette's official score. `_overlap_iou`/`_is_match` (IoU >= 5% +
  same `file_path`), `recall_at_k`, `evaluate_recall` (matches
  `StudentSearchResults` entries to `AnsweredQuestion` ground truth by
  `question_id`, skips questions without ground truth), `format_report`.
  Warns in the report when a requested k exceeds the k actually used at
  retrieval time (can't measure recall@10 from a search done with k=5).
- `src/cli.py` (`RagCLI`, exposed via `src/__main__.py` + Python Fire) —
  commands: `index`, `search`, `search_dataset`, `answer`, `answer_dataset`,
  `evaluate`. Every command catches its relevant exceptions and prints a
  clean message — no unhandled tracebacks. Run as:
  `uv run python -m src <command> [options]`.

## Testing commands

```bash
uv run python -m src index                                   # build data/processed/ index
uv run python -m src search "some query" -k 5
uv run python -m src search_dataset --dataset_path <p> --save_directory <d>
uv run python -m src answer "some query" -k 5                # downloads Qwen3-0.6B first run (~1.5G)
uv run python -m src answer_dataset --student_search_results_path <search_dataset output> --save_directory <d>
make lint    # flake8 (repo root, excludes .venv,data) + mypy (src/ + main.py only — data/ has duplicate module names, would break mypy)
```

Perf measured on the real vLLM corpus (informal, not the moulinette's own
run): indexing 1969 files → 18733 chunks in 3.4s; `search_dataset` on 200
questions in 0.38s. Both comfortably beat the subject's thresholds
(<5 min indexing, <90s for 200 questions).

## Status (see TODO.md for authoritative detail)

Done: setup, data models, Python/Markdown chunking, BM25 indexer,
retrieval, answer generation (Qwen3-0.6B), `evaluate`/recall@k
(section 6), CLI (`index`/`search`/`search_dataset`/`answer`/
`answer_dataset`/`evaluate`), project-wide flake8/mypy clean.

Not done: `README.md` (section 8, entirely empty, must be in English);
optional `lint-strict` Makefile rule; optional pytest/unittest tests;
bonus features (section 10, only gradable once mandatory part is fully
done).

Recall@5 thresholds (≥80% docs, ≥50% code) are now measured for real using
the provided ground-truth datasets
(`data/datasets/AnsweredQuestions/dataset_{docs,code}_public.json`, 100/99
questions, via `search_dataset --k 10` then `evaluate`): **docs 82.0%**,
**code 50.5%** — both pass, but code is very close to the threshold
(+0.5 pt margin) — re-check if the Python chunking changes.

## Style/collaboration notes

- User communicates in French; code/comments/docstrings/README stay in
  English (subject requires English README).
- Keep code minimal — no speculative abstractions, no comments beyond
  non-obvious WHY (e.g. the BatchEncoding gotcha above is worth a code
  comment; restating what a function does is not).
- Every CLI command must degrade gracefully (empty query, k=0, k > corpus
  size, missing index/files, malformed JSON, missing model) — this is a
  graded requirement, not a nice-to-have.
