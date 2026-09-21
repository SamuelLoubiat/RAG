"""Evaluate retrieval quality against ground-truth sources.

For our own iteration only: the official recall@k score at soutenance is
computed by the moulinette's own `evaluate_student_search_results`, never
by this module.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

from src.models import AnsweredQuestion, MinimalSearchResults, MinimalSource

MIN_IOU = 0.05
RECALL_KS: Sequence[int] = (1, 3, 5, 10)


@dataclass
class QuestionRecall:
    """Recall@k for a single question, keyed by k."""

    question_id: str
    recall: Dict[int, float]


@dataclass
class RecallReport:
    """Aggregated recall@k report over a dataset."""

    n_questions: int
    retrieval_k: int
    average_recall: Dict[int, float]
    per_question: List[QuestionRecall]


def _overlap_iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection-over-union of two character ranges in the same file."""
    if a.file_path != b.file_path:
        return 0.0
    start = max(a.first_character_index, b.first_character_index)
    end = min(a.last_character_index, b.last_character_index)
    intersection = max(0, end - start)
    if intersection == 0:
        return 0.0
    union = (
        (a.last_character_index - a.first_character_index)
        + (b.last_character_index - b.first_character_index)
        - intersection
    )
    if union <= 0:
        return 0.0
    return intersection / union


def _is_match(retrieved: MinimalSource, reference: MinimalSource) -> bool:
    """A reference source is matched by the same file and >=5% IoU."""
    return _overlap_iou(retrieved, reference) >= MIN_IOU


def recall_at_k(
    retrieved: Sequence[MinimalSource],
    references: Sequence[MinimalSource],
    k: int,
) -> float:
    """Fraction of ``references`` found among the top-k ``retrieved``."""
    if not references:
        return 1.0
    top_k = retrieved[:k]
    found = sum(
        1
        for reference in references
        if any(_is_match(candidate, reference) for candidate in top_k)
    )
    return found / len(references)


def evaluate_recall(
    search_results: List[MinimalSearchResults],
    ground_truth: Dict[str, AnsweredQuestion],
    retrieval_k: int,
    ks: Sequence[int] = RECALL_KS,
) -> RecallReport:
    """Compute recall@k per question, and averaged, against ground truth.

    Only questions present in ``ground_truth`` are scored; the rest are
    silently skipped (they have no reference sources to compare against).
    """
    per_question = []
    for result in search_results:
        reference = ground_truth.get(result.question_id)
        if reference is None:
            continue
        scores = {
            k: recall_at_k(result.retrieved_sources, reference.sources, k)
            for k in ks
        }
        per_question.append(
            QuestionRecall(question_id=result.question_id, recall=scores)
        )

    average_recall = {}
    for k in ks:
        values = [entry.recall[k] for entry in per_question]
        average_recall[k] = sum(values) / len(values) if values else 0.0

    return RecallReport(
        n_questions=len(per_question),
        retrieval_k=retrieval_k,
        average_recall=average_recall,
        per_question=per_question,
    )


def format_report(report: RecallReport) -> str:
    """Render an evaluation report as a human-readable string."""
    lines = [
        f"Evaluated {report.n_questions} question(s) with ground truth "
        f"(retrieved with k={report.retrieval_k})."
    ]
    for k in sorted(report.average_recall):
        marker = " (not fully measurable: k > retrieval k)" if (
            k > report.retrieval_k
        ) else ""
        lines.append(f"  Recall@{k}: {report.average_recall[k]:.1%}{marker}")
    return "\n".join(lines)
