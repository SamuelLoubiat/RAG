"""Command-line interface for the RAG pipeline, built with Python Fire."""

import json
from pathlib import Path

from tqdm import tqdm

from src.generation import DEFAULT_MODEL, generate_answer, load_generator
from src.indexer import build_index
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.search import load_retriever
from src.search import search as search_sources


class RagCLI:
    """Entry point exposing the RAG pipeline commands."""

    def index(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        max_chunk_size: int = 2000,
    ) -> None:
        """Ingest ``raw_dir`` and build the BM25 index under
        ``processed_dir``.
        """
        try:
            n_chunks = build_index(raw_dir, processed_dir, max_chunk_size)
        except ValueError as exc:
            print(f"Indexing failed: {exc}")
            return
        print(
            f"Ingestion complete! Indexed {n_chunks} chunks "
            f"under {processed_dir}/"
        )

    def search(
        self,
        query: str,
        k: int = 10,
        index_dir: str = "data/processed",
    ) -> None:
        """Print the top-k sources for a single query."""
        try:
            retriever = load_retriever(index_dir)
        except FileNotFoundError as exc:
            print(exc)
            return

        sources = search_sources(query, k=k, retriever=retriever)
        if not sources:
            print("No results.")
            return
        for source in sources:
            start = source.first_character_index
            end = source.last_character_index
            print(f"{source.file_path} [{start}:{end}]")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_dir: str = "data/processed",
    ) -> None:
        """Run search over a whole dataset and write a
        StudentSearchResults JSON file under ``save_directory``.
        """
        try:
            retriever = load_retriever(index_dir)
        except FileNotFoundError as exc:
            print(exc)
            return

        try:
            with open(dataset_path, encoding="utf-8") as dataset_file:
                raw_dataset = json.load(dataset_file)
            dataset = RagDataset.model_validate(raw_dataset)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Could not load dataset '{dataset_path}': {exc}")
            return

        results = []
        questions = tqdm(
            dataset.rag_questions, desc="Searching", unit="question"
        )
        for question in questions:
            sources = search_sources(
                question.question, k=k, retriever=retriever
            )
            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(search_results=results, k=k)

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(dataset_path).name
        out_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved student_search_results to {out_path}")

    def answer(
        self,
        query: str,
        k: int = 10,
        index_dir: str = "data/processed",
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 256,
    ) -> None:
        """Search for ``query`` and print a generated, grounded answer."""
        try:
            retriever = load_retriever(index_dir)
        except FileNotFoundError as exc:
            print(exc)
            return

        sources = search_sources(query, k=k, retriever=retriever)

        try:
            generator = load_generator(model_name)
        except OSError as exc:
            print(f"Could not load model '{model_name}': {exc}")
            return

        answer_text = generate_answer(
            query, sources, generator, max_new_tokens=max_new_tokens
        )
        for source in sources:
            start = source.first_character_index
            end = source.last_character_index
            print(f"{source.file_path} [{start}:{end}]")
        print(f"\n{answer_text}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 256,
    ) -> None:
        """Generate answers for a StudentSearchResults JSON file and write a
        StudentSearchResultsAndAnswer JSON file under ``save_directory``.
        """
        try:
            with open(
                student_search_results_path, encoding="utf-8"
            ) as results_file:
                raw_results = json.load(results_file)
            student_results = StudentSearchResults.model_validate(
                raw_results
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(
                "Could not load student search results "
                f"'{student_search_results_path}': {exc}"
            )
            return

        try:
            generator = load_generator(model_name)
        except OSError as exc:
            print(f"Could not load model '{model_name}': {exc}")
            return

        answers = []
        questions = tqdm(
            student_results.search_results,
            desc="Answering",
            unit="question",
        )
        for result in questions:
            answer_text = generate_answer(
                result.question,
                result.retrieved_sources,
                generator,
                max_new_tokens=max_new_tokens,
            )
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=student_results.k
        )

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(student_search_results_path).name
        out_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved student_search_results_and_answer to {out_path}")
