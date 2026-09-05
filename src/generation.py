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


def _build_prompt(
    question: str,
    sources: List[MinimalSource],
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Assemble a grounded prompt from the question and retrieved sources."""
    context_blocks = []
    budget = max_context_chars
    for source in sources:
        text = _read_source_text(source)
        if not text or budget <= 0:
            continue
        block = f"# {source.file_path}\n{text[:budget]}"
        context_blocks.append(block)
        budget -= len(block)

    context = "\n\n---\n\n".join(context_blocks)
    return (
        "You are a helpful assistant answering questions about the vLLM "
        "codebase. Answer the question using only the context below. "
        "If the context does not contain the answer, say so explicitly "
        "instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer(
    question: str,
    sources: List[MinimalSource],
    generator: Generator,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Generate a grounded answer to ``question`` from ``sources``."""
    tokenizer, model = generator
    prompt = _build_prompt(question, sources)
    messages = [{"role": "user", "content": prompt}]
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
        ),
    )
    generated_tokens = output[0][input_ids.shape[-1]:]
    return cast(
        str,
        tokenizer.decode(generated_tokens, skip_special_tokens=True),
    ).strip()
