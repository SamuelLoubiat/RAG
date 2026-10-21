"""Generate grounded answers from retrieved sources using a local LLM."""

from pathlib import Path
from typing import Any, List, Tuple, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers import PreTrainedTokenizerBase

from src.models import MinimalSource

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARS = 6000
DEFAULT_MAX_NEW_TOKENS = 256
NO_ANSWER = "No relevant context was found to answer this question."

Generator = Tuple[PreTrainedTokenizerBase, PreTrainedModel]


def load_generator(model_name: str = DEFAULT_MODEL) -> Generator:
    """Load the tokenizer and causal LM used to answer questions."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def _read_source_text(source: MinimalSource) -> str:
    """Read the chunk of text referenced by ``source``, or "" if unreadable."""
    try:
        text = Path(source.file_path).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return ""
    return text[source.first_character_index:source.last_character_index]


def _build_context(
    sources: List[MinimalSource],
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Concatenate the text of ``sources`` within a character budget."""
    context_blocks = []
    budget = max_context_chars
    for source in sources:
        text = _read_source_text(source)
        if not text or budget <= 0:
            continue
        block = f"# {source.file_path}\n{text[:budget]}"
        context_blocks.append(block)
        budget -= len(block)
    return "\n\n---\n\n".join(context_blocks)


# Qwen3-0.6B over-refuses when told to "say so if the context lacks the
# answer", so the refusal is reserved for clearly unrelated context.
SYSTEM_PROMPT = (
    "You are an expert assistant for the vLLM project. You are given "
    "excerpts from the vLLM repository that were retrieved for the user's "
    "question. Write a clear, direct answer of 2 to 5 sentences, based only "
    "on facts stated in these excerpts, and do not add anything they do not "
    "support. If the excerpts are about something unrelated to the question, "
    "answer only: I could not find the answer in the vLLM codebase."
)


def _build_messages(question: str, context: str) -> List[dict[str, str]]:
    """Assemble the chat messages for a grounded answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\n"
                                    f"Question: {question}"},
    ]


def generate_answer(
    question: str,
    sources: List[MinimalSource],
    generator: Generator,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Generate a grounded answer to ``question`` from ``sources``."""
    context = _build_context(sources)
    # Without a question or context, the model can only hallucinate.
    if not question.strip() or not context.strip():
        return NO_ANSWER
    tokenizer, model = generator
    messages = _build_messages(question, context)
    prompt_text = cast(
        str,
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        ),
    )
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = cast(torch.Tensor, encoded["input_ids"])
    output = cast(
        torch.Tensor,
        cast(Any, model).generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            # Greedy decoding keeps answers reproducible across runs; the
            # penalty avoids the loops greedy search is prone to.
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.1,
        ),
    )
    generated_tokens = output[0][input_ids.shape[-1]:]
    return cast(
        str,
        tokenizer.decode(generated_tokens, skip_special_tokens=True),
    ).strip()
