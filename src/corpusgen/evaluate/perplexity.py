"""Corpus-level perplexity metrics via a pretrained causal language model.

Computes both **corpus perplexity** (the standard language-modelling
metric, token-weighted) and **mean per-sentence perplexity** (sentence-
weighted), along with per-sentence breakdowns for analysis.

This is a standalone post-hoc evaluation utility — it is *not*
auto-computed by ``evaluate()`` because it requires loading a GPU model.
Call it explicitly when you want perplexity analysis.

Requires ``torch`` and ``transformers`` (optional dependency group
``local``).

Usage::

    from corpusgen.evaluate.perplexity import compute_corpus_perplexity

    # Simple — loads gpt2 automatically
    metrics = compute_corpus_perplexity(sentences, model_name="gpt2")

    # Shared model — avoids loading the model twice when you are also
    # using PerplexityFluencyScorer or LocalBackend:
    from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

    scorer = PerplexityFluencyScorer(model_name="gpt2", device="cuda")
    scorer("warm-up call to trigger lazy load")

    metrics = compute_corpus_perplexity(
        sentences,
        model=scorer._model,
        tokenizer=scorer._tokenizer,
    )

Mathematical definitions:

    sentence_perplexity_i = exp(NLL_i / T_i)

    corpus_perplexity     = exp(sum(NLL_i) / sum(T_i))   # token-weighted
    mean_perplexity       = mean(sentence_perplexity_i)   # sentence-weighted

where NLL_i is the total negative log-likelihood for sentence *i* and
T_i is the number of next-token predictions (= tokens − 1) in that
sentence.  Corpus perplexity weights longer sentences more heavily;
mean perplexity treats each sentence equally.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusPerplexityMetrics:
    """Immutable container for corpus-level perplexity results.

    Attributes:
        per_sentence: Raw perplexity for each scored sentence (order
            matches the input order, excluding skipped sentences).
        corpus_perplexity: ``exp(total_nll / num_tokens)`` — the standard
            language-modelling perplexity, weighted by token count.
        mean_perplexity: Arithmetic mean of per-sentence perplexities.
        median_perplexity: Median of per-sentence perplexities.
        std_perplexity: Population standard deviation of per-sentence
            perplexities (0.0 when only one sentence).
        min_perplexity: Lowest per-sentence perplexity.
        max_perplexity: Highest per-sentence perplexity.
        num_sentences: Number of sentences actually scored (excludes
            empty/whitespace and single-token sentences).
        num_tokens: Total next-token predictions across all sentences.
        total_nll: Sum of negative log-likelihood across all tokens,
            stored for reproducibility (``corpus_perplexity ==
            exp(total_nll / num_tokens)``).
    """

    per_sentence: list[float]
    corpus_perplexity: float
    mean_perplexity: float
    median_perplexity: float
    std_perplexity: float
    min_perplexity: float
    max_perplexity: float
    num_sentences: int
    num_tokens: int
    total_nll: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_corpus_perplexity(
    sentences: list[str],
    model_name: str = "gpt2",
    device: str | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    model: Any = None,
    tokenizer: Any = None,
) -> CorpusPerplexityMetrics:
    """Compute corpus-level perplexity metrics.

    Args:
        sentences: List of text sentences to evaluate.
        model_name: HuggingFace model ID, used only when *model* and
            *tokenizer* are not supplied.  Defaults to ``"gpt2"``.
        device: Device string (``"cuda"``, ``"cpu"``, ``"auto"``).
            Auto-detected when *None*.  Ignored when *model* is provided.
        batch_size: Number of sentences per forward pass.
        max_length: Maximum token length per sentence (truncated).
        model: An already-loaded HuggingFace causal LM.  Pass together
            with *tokenizer* to avoid redundant model loading (e.g.,
            share with :class:`PerplexityFluencyScorer`).
        tokenizer: The corresponding HuggingFace tokenizer.  Must be
            provided together with *model*, or both omitted.

    Returns:
        :class:`CorpusPerplexityMetrics` with per-sentence and
        corpus-level statistics.

    Raises:
        ValueError: If both *model* and *tokenizer* are not provided
            together, or if no scoreable sentences remain.
        ImportError: If ``torch`` or ``transformers`` is not installed
            and no model is injected.
    """
    # ------------------------------------------------------------------
    # Validate model/tokenizer pairing
    # ------------------------------------------------------------------
    if (model is None) != (tokenizer is None):
        raise ValueError(
            "Both model and tokenizer must be provided together, or neither."
        )

    # ------------------------------------------------------------------
    # Filter valid sentences
    # ------------------------------------------------------------------
    valid_sentences = [s for s in sentences if s and s.strip()]
    if not valid_sentences:
        raise ValueError("No valid sentences to score (all empty or whitespace).")

    # ------------------------------------------------------------------
    # Load model/tokenizer if not injected
    # ------------------------------------------------------------------
    if model is None:
        from corpusgen.generate.scorers.fluency import (
            _detect_device,
            _load_model,
            _load_tokenizer,
        )

        if device is None:
            device = _detect_device()

        logger.info("Loading perplexity model %s on %s", model_name, device)
        tokenizer = _load_tokenizer(model_name)
        model = _load_model(model_name, device)
        logger.info("Perplexity model loaded successfully.")

    # ------------------------------------------------------------------
    # Prepare tokenizer for left-padded batching (required for causal LMs)
    # ------------------------------------------------------------------
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "pad_token_id") and hasattr(tokenizer, "eos_token_id"):
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------
    import torch
    import torch.nn.functional as F

    all_sentence_ppls: list[float] = []
    total_nll = 0.0
    total_tokens = 0

    try:
        with torch.no_grad():
            for i in range(0, len(valid_sentences), batch_size):
                batch = valid_sentences[i : i + batch_size]

                encodings = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )

                input_ids = encodings["input_ids"].to(model.device)
                attention_mask = encodings["attention_mask"].to(model.device)

                # Compute position_ids so that each sentence's first
                # real token gets position 0 regardless of left-padding.
                # Without this, models with absolute positional embeddings
                # (e.g. GPT-2) assign wrong positions to padded sequences.
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                logits = outputs.logits

                # Shift for causal LM next-token prediction:
                #   logits[:, t, :] predicts input_ids[:, t+1]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                # Valid prediction mask: both the context position (source)
                # and the target position must be non-padding.
                prediction_mask = (
                    attention_mask[:, :-1] * attention_mask[:, 1:]
                ).float()

                # Per-token cross-entropy (no reduction)
                B, T, V = shift_logits.shape
                per_token_ce = F.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(B, T)

                # Mask out padding positions and aggregate per sentence
                masked_ce = per_token_ce * prediction_mask
                sentence_nll = masked_ce.sum(dim=1)  # (B,)
                sentence_token_count = prediction_mask.sum(dim=1)  # (B,)

                for j in range(len(batch)):
                    count = int(sentence_token_count[j].item())
                    if count == 0:
                        # Single-token sentence → 0 predictions → skip
                        continue
                    nll_j = sentence_nll[j].item()
                    ppl_j = math.exp(nll_j / count)
                    all_sentence_ppls.append(ppl_j)
                    total_nll += nll_j
                    total_tokens += count
    finally:
        tokenizer.padding_side = original_padding_side

    # ------------------------------------------------------------------
    # Validate we have at least one scored sentence
    # ------------------------------------------------------------------
    if not all_sentence_ppls:
        raise ValueError(
            "No valid sentences to score (all had fewer than 2 tokens)."
        )

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    n = len(all_sentence_ppls)
    corpus_ppl = math.exp(total_nll / total_tokens)
    mean_ppl = sum(all_sentence_ppls) / n

    sorted_ppls = sorted(all_sentence_ppls)
    if n % 2 == 1:
        median_ppl = sorted_ppls[n // 2]
    else:
        median_ppl = (sorted_ppls[n // 2 - 1] + sorted_ppls[n // 2]) / 2.0

    if n > 1:
        variance = sum((p - mean_ppl) ** 2 for p in all_sentence_ppls) / n
        std_ppl = math.sqrt(variance)
    else:
        std_ppl = 0.0

    return CorpusPerplexityMetrics(
        per_sentence=all_sentence_ppls,
        corpus_perplexity=corpus_ppl,
        mean_perplexity=mean_ppl,
        median_perplexity=median_ppl,
        std_perplexity=std_ppl,
        min_perplexity=min(all_sentence_ppls),
        max_perplexity=max(all_sentence_ppls),
        num_sentences=n,
        num_tokens=total_tokens,
        total_nll=total_nll,
    )
