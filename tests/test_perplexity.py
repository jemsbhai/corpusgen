"""Tests for corpusgen.evaluate.perplexity — corpus-level perplexity metrics.

Test tiers:
    - **Fast tests** (default): Mock torch/transformers at the boundary.
      Tests dataclass construction, validation, scoring math, edge cases.
    - **Slow tests** (@pytest.mark.slow): Real GPT-2 model, real perplexity.

Scientific contract:
    - corpus_perplexity = exp(total_NLL / total_tokens)  (standard LM metric)
    - mean_perplexity = mean of per-sentence perplexities
    - These two are NOT equal in general (corpus weights by token count)
    - Grammatical sentences have lower perplexity than gibberish
    - Empty/whitespace sentences are excluded from computation
    - Per-sentence perplexity = exp(sentence_NLL / sentence_prediction_count)
    - For uniform logits (all zeros), PPL = vocab_size regardless of sentence length

Mathematical reference:
    With vocab_size V and uniform logits (all zeros):
        softmax -> uniform distribution -> P(any token) = 1/V
        CE per prediction = -log(1/V) = log(V)
        For sentence of N tokens -> N-1 predictions -> NLL = (N-1)*log(V)
        PPL = exp(NLL / (N-1)) = exp(log(V)) = V

Note on mock token IDs:
    All mock token IDs must be in [0, vocab_size-1].  With vocab_size=4,
    valid IDs are 0, 1, 2, 3.  ID 0 is reserved for padding.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from corpusgen.evaluate.perplexity import (
    CorpusPerplexityMetrics,
    compute_corpus_perplexity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx(value: float, abs_tol: float = 1e-4) -> pytest.approx:
    """Shorthand for pytest.approx with tolerance suitable for perplexity."""
    return pytest.approx(value, abs=abs_tol)


def _make_uniform_mocks(vocab_size: int = 4):
    """Create mock model and tokenizer that produce uniform logits.

    The model returns all-zero logits for every position, giving a
    uniform distribution over the vocabulary.  This means:
        CE per prediction = log(vocab_size)
        PPL per sentence  = vocab_size

    The tokenizer is configured per-test via its return_value to
    supply specific left-padded input_ids and attention_mask tensors.

    Returns:
        (model, tokenizer, vocab_size) tuple.
    """
    import torch

    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = MagicMock()
    model.device = torch.device("cpu")
    model.eval.return_value = model

    V = vocab_size

    def model_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        batch, seq_len = input_ids.shape
        output = MagicMock()
        output.logits = torch.zeros(batch, seq_len, V)
        return output

    model.side_effect = model_forward

    return model, tokenizer, V


def _make_tokenizer_output(token_sequences: list[list[int]], pad_id: int = 0):
    """Build left-padded input_ids and attention_mask tensors.

    Args:
        token_sequences: List of token ID lists per sentence, unpadded.
            All token IDs must be in [0, vocab_size-1].
        pad_id: Padding token ID.

    Returns:
        Dict with ``input_ids`` and ``attention_mask`` tensors.
    """
    import torch

    max_len = max(len(seq) for seq in token_sequences)
    padded_ids = []
    masks = []
    for seq in token_sequences:
        pad_len = max_len - len(seq)
        padded_ids.append([pad_id] * pad_len + seq)
        masks.append([0] * pad_len + [1] * len(seq))

    result = MagicMock()
    result.__getitem__ = lambda self, key: {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }[key]
    result.to = MagicMock(return_value=result)
    return result


# ---------------------------------------------------------------------------
# 1. Dataclass construction and immutability
# ---------------------------------------------------------------------------


class TestCorpusPerplexityMetrics:
    """Test the CorpusPerplexityMetrics frozen dataclass."""

    def test_construction(self):
        """All fields should be accessible after construction."""
        m = CorpusPerplexityMetrics(
            per_sentence=[4.0, 4.0],
            corpus_perplexity=4.0,
            mean_perplexity=4.0,
            median_perplexity=4.0,
            std_perplexity=0.0,
            min_perplexity=4.0,
            max_perplexity=4.0,
            num_sentences=2,
            num_tokens=4,
            total_nll=4 * math.log(4),
        )
        assert m.num_sentences == 2
        assert m.num_tokens == 4
        assert m.corpus_perplexity == 4.0
        assert m.mean_perplexity == 4.0
        assert len(m.per_sentence) == 2

    def test_frozen(self):
        """Dataclass should be immutable."""
        m = CorpusPerplexityMetrics(
            per_sentence=[4.0],
            corpus_perplexity=4.0,
            mean_perplexity=4.0,
            median_perplexity=4.0,
            std_perplexity=0.0,
            min_perplexity=4.0,
            max_perplexity=4.0,
            num_sentences=1,
            num_tokens=2,
            total_nll=math.log(4) * 2,
        )
        with pytest.raises(FrozenInstanceError):
            m.corpus_perplexity = 999.0  # type: ignore[misc]

    def test_all_fields_present(self):
        """Verify the complete set of expected fields."""
        m = CorpusPerplexityMetrics(
            per_sentence=[1.0],
            corpus_perplexity=1.0,
            mean_perplexity=1.0,
            median_perplexity=1.0,
            std_perplexity=0.0,
            min_perplexity=1.0,
            max_perplexity=1.0,
            num_sentences=1,
            num_tokens=1,
            total_nll=0.0,
        )
        expected_fields = {
            "per_sentence",
            "corpus_perplexity",
            "mean_perplexity",
            "median_perplexity",
            "std_perplexity",
            "min_perplexity",
            "max_perplexity",
            "num_sentences",
            "num_tokens",
            "total_nll",
        }
        assert set(m.__dataclass_fields__.keys()) == expected_fields


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------


class TestComputeCorpusPerplexityValidation:
    """Test input validation and error handling."""

    def test_empty_list_raises(self):
        """Empty sentence list should raise ValueError."""
        with pytest.raises(ValueError, match="[Nn]o valid"):
            compute_corpus_perplexity([], model_name="gpt2")

    def test_all_whitespace_raises(self):
        """All-whitespace sentences should raise ValueError."""
        model, tokenizer, _ = _make_uniform_mocks()
        with pytest.raises(ValueError, match="[Nn]o valid"):
            compute_corpus_perplexity(
                ["", "   ", "\n\t"],
                model=model,
                tokenizer=tokenizer,
            )

    def test_model_without_tokenizer_raises(self):
        """Providing model without tokenizer should raise ValueError."""
        model = MagicMock()
        with pytest.raises(ValueError, match="[Bb]oth.*model.*tokenizer"):
            compute_corpus_perplexity(["hello"], model=model, tokenizer=None)

    def test_tokenizer_without_model_raises(self):
        """Providing tokenizer without model should raise ValueError."""
        tokenizer = MagicMock()
        with pytest.raises(ValueError, match="[Bb]oth.*model.*tokenizer"):
            compute_corpus_perplexity(["hello"], model=None, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# 3. Core math -- uniform logits
#    With uniform logits and vocab_size V, every prediction has
#    CE = log(V), so PPL = V regardless of sentence content or length.
# ---------------------------------------------------------------------------


class TestComputeCorpusPerplexityUniformLogits:
    """Test perplexity computation with uniform (all-zero) logits."""

    def test_single_sentence(self):
        """Single sentence, uniform logits -> PPL = vocab_size.

        Sentence: 3 tokens [1, 2, 3] -> 2 predictions.
        CE per prediction = log(4) ~ 1.3863.
        NLL = 2 * log(4), num_tokens = 2.
        PPL = exp(2*log(4)/2) = 4.0.
        """
        model, tokenizer, V = _make_uniform_mocks(vocab_size=4)
        tok_out = _make_tokenizer_output([[1, 2, 3]])
        tokenizer.return_value = tok_out

        result = compute_corpus_perplexity(
            ["some text"],
            model=model,
            tokenizer=tokenizer,
            batch_size=1,
        )

        assert result.num_sentences == 1
        assert result.num_tokens == 2  # 3 tokens -> 2 predictions
        assert result.per_sentence[0] == _approx(float(V))
        assert result.corpus_perplexity == _approx(float(V))
        assert result.mean_perplexity == _approx(float(V))
        assert result.total_nll == _approx(2 * math.log(V))

    def test_two_equal_length_sentences(self):
        """Two sentences of equal length, uniform logits.

        Both 3 tokens -> PPL = 4.0 each.
        corpus_ppl = mean_ppl = 4.0.
        """
        model, tokenizer, V = _make_uniform_mocks(vocab_size=4)
        tok_out = _make_tokenizer_output([[1, 2, 3], [3, 2, 1]])
        tokenizer.return_value = tok_out

        result = compute_corpus_perplexity(
            ["sentence one", "sentence two"],
            model=model,
            tokenizer=tokenizer,
            batch_size=8,
        )

        assert result.num_sentences == 2
        assert result.num_tokens == 4  # 2 predictions per sentence
        assert result.corpus_perplexity == _approx(float(V))
        assert result.mean_perplexity == _approx(float(V))
        assert result.std_perplexity == _approx(0.0)
        assert result.min_perplexity == _approx(float(V))
        assert result.max_perplexity == _approx(float(V))

    def test_different_length_sentences(self):
        """Different-length sentences, uniform logits -> PPL still = V.

        Sentence A: 4 tokens -> 3 predictions.
        Sentence B: 2 tokens -> 1 prediction.
        Both have PPL = 4.0 (uniform logits -> PPL independent of length).
        corpus_ppl = exp(4*log(4)/4) = 4.0.
        """
        model, tokenizer, V = _make_uniform_mocks(vocab_size=4)
        tok_out = _make_tokenizer_output([[1, 2, 3, 1], [2, 3]])
        tokenizer.return_value = tok_out

        result = compute_corpus_perplexity(
            ["longer sentence", "short"],
            model=model,
            tokenizer=tokenizer,
            batch_size=8,
        )

        assert result.num_sentences == 2
        assert result.num_tokens == 4  # 3 + 1 predictions
        assert result.corpus_perplexity == _approx(float(V))
        assert result.mean_perplexity == _approx(float(V))

    def test_whitespace_sentences_filtered(self):
        """Whitespace sentences should be excluded, not cause errors.

        Three inputs but only one valid sentence.
        """
        model, tokenizer, V = _make_uniform_mocks(vocab_size=4)
        # Only one valid sentence tokenizes; empty/whitespace filtered before tokenization
        tok_out = _make_tokenizer_output([[1, 2, 3]])
        tokenizer.return_value = tok_out

        result = compute_corpus_perplexity(
            ["", "valid text", "   "],
            model=model,
            tokenizer=tokenizer,
            batch_size=8,
        )

        assert result.num_sentences == 1
        assert result.per_sentence[0] == _approx(float(V))


# ---------------------------------------------------------------------------
# 4. Core math -- corpus_perplexity vs mean_perplexity
#    When sentences have different per-token losses, the two metrics
#    diverge.  corpus_perplexity weights by token count.
# ---------------------------------------------------------------------------


class TestCorpusVsMeanPerplexity:
    """Verify that corpus_perplexity and mean_perplexity differ correctly.

    Setup:
        Sentence A: 4 tokens [1,2,3,1] -> 3 predictions, UNIFORM logits -> PPL = 4.0
        Sentence B: 2 tokens [1,2]     -> 1 prediction,  CONFIDENT logits -> PPL ~ 1.0

    The model returns different logits per batch element:
        - Sentence A positions: all-zero logits (uniform, CE = log(4))
        - Sentence B positions: logit 50.0 at target token (CE ~ 0)

    Expected:
        mean_ppl = (4.0 + ~1.0) / 2 ~ 2.5
        corpus_ppl = exp((3*log(4) + ~0) / (3+1)) = exp(3*log(4)/4) = 4^(3/4) ~ 2.828

    The corpus_perplexity is pulled toward the longer sentence's PPL
    because it weights by token count.
    """

    def _make_differential_mocks(self):
        """Model where sentence A gets uniform logits, sentence B gets confident logits."""
        import torch

        vocab_size = 4

        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "<eos>"
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        # Left-padded: A=[1,2,3,1], B=[1,2] -> padded to length 4
        # input_ids  = [[1,2,3,1], [0,0,1,2]]
        # attn_mask  = [[1,1,1,1], [0,0,1,1]]
        tok_out = _make_tokenizer_output([[1, 2, 3, 1], [1, 2]])
        tokenizer.return_value = tok_out

        model = MagicMock()
        model.device = torch.device("cpu")
        model.eval.return_value = model

        def model_forward(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            batch, seq_len = input_ids.shape
            logits = torch.zeros(batch, seq_len, vocab_size)

            # Sentence B (row 1): make logits "confident" at positions
            # where predictions happen.
            # After left-padding, B = [0, 0, 1, 2].
            # Shifted: position 2 predicts token at position 3 (= 2).
            # Set logits[1, 2, 2] = 50.0  (high logit at target index 2)
            logits[1, 2, 2] = 50.0

            output = MagicMock()
            output.logits = logits
            return output

        model.side_effect = model_forward

        return model, tokenizer

    def test_corpus_ppl_differs_from_mean_ppl(self):
        """corpus_perplexity and mean_perplexity should differ here."""
        model, tokenizer = self._make_differential_mocks()

        result = compute_corpus_perplexity(
            ["sentence a long", "short"],
            model=model,
            tokenizer=tokenizer,
            batch_size=8,
        )

        # Sentence A: 3 predictions, PPL = 4.0
        # Sentence B: 1 prediction, PPL ~ 1.0 (confident)
        assert result.per_sentence[0] == _approx(4.0)
        assert result.per_sentence[1] == _approx(1.0, abs_tol=0.01)

        # mean_ppl ~ (4.0 + 1.0) / 2 = 2.5
        assert result.mean_perplexity == _approx(2.5, abs_tol=0.01)

        # corpus_ppl = exp((3*log(4) + ~0) / 4) = 4^(3/4) ~ 2.828
        expected_corpus = 4.0 ** (3.0 / 4.0)
        assert result.corpus_perplexity == _approx(expected_corpus, abs_tol=0.05)

        # Key property: corpus_ppl > mean_ppl here because the longer
        # sentence (higher PPL) gets more weight.
        assert result.corpus_perplexity > result.mean_perplexity

    def test_total_nll_and_num_tokens(self):
        """Verify total_nll and num_tokens for reproducibility."""
        model, tokenizer = self._make_differential_mocks()

        result = compute_corpus_perplexity(
            ["sentence a long", "short"],
            model=model,
            tokenizer=tokenizer,
            batch_size=8,
        )

        assert result.num_sentences == 2
        assert result.num_tokens == 4  # 3 + 1 predictions
        # total_nll ~ 3*log(4) + ~0
        assert result.total_nll == _approx(3 * math.log(4), abs_tol=0.01)


# ---------------------------------------------------------------------------
# 5. Model/tokenizer injection
# ---------------------------------------------------------------------------


class TestModelInjection:
    """Test model and tokenizer injection (shared model pattern)."""

    def test_injected_model_not_reloaded(self):
        """When model+tokenizer are provided, no loading should occur."""
        model, tokenizer, V = _make_uniform_mocks(vocab_size=4)
        tok_out = _make_tokenizer_output([[1, 2, 3]])
        tokenizer.return_value = tok_out

        # If it tried to load, it would fail (no real model name)
        result = compute_corpus_perplexity(
            ["hello world"],
            model=model,
            tokenizer=tokenizer,
        )

        assert result.num_sentences == 1
        assert result.corpus_perplexity == _approx(float(V))

    def test_both_model_and_tokenizer_required(self):
        """Must provide both or neither."""
        model, _, _ = _make_uniform_mocks()
        with pytest.raises(ValueError):
            compute_corpus_perplexity(["hello"], model=model, tokenizer=None)
        tokenizer = MagicMock()
        with pytest.raises(ValueError):
            compute_corpus_perplexity(["hello"], model=None, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestPerplexityEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_token_sentence_skipped(self):
        """A sentence with only 1 token has 0 predictions -> skip it.

        input: ["single_tok", "two tokens"]
        Only the second sentence contributes.
        """
        model, tokenizer, V = _make_uniform_mocks(vocab_size=4)
        # First sentence: 1 token [3] -> 0 predictions -> skipped
        # Second sentence: 3 tokens [1,2,3] -> 2 predictions
        tok_out = _make_tokenizer_output([[3], [1, 2, 3]])
        tokenizer.return_value = tok_out

        result = compute_corpus_perplexity(
            ["x", "valid sentence"],
            model=model,
            tokenizer=tokenizer,
            batch_size=8,
        )

        assert result.num_sentences == 1
        assert result.per_sentence[0] == _approx(float(V))

    def test_batch_size_smaller_than_corpus(self):
        """Multiple batches should produce same result as single batch.

        4 sentences, batch_size=2 -> 2 batches.
        All uniform logits -> PPL = V for each.
        """
        model, tokenizer, V = _make_uniform_mocks(vocab_size=4)

        # With batch_size=2 and 4 sentences, there will be 2 tokenizer calls.
        tok_batch1 = _make_tokenizer_output([[1, 2, 3], [3, 2, 1]])
        tok_batch2 = _make_tokenizer_output([[2, 1, 3], [1, 3, 2]])
        tokenizer.side_effect = [tok_batch1, tok_batch2]

        result = compute_corpus_perplexity(
            ["s1", "s2", "s3", "s4"],
            model=model,
            tokenizer=tokenizer,
            batch_size=2,
        )

        assert result.num_sentences == 4
        assert result.corpus_perplexity == _approx(float(V))
        assert result.mean_perplexity == _approx(float(V))
        assert len(result.per_sentence) == 4

    def test_median_and_std(self):
        """Verify median and std with known per-sentence values.

        Two sentences with PPL = 4.0 each -> median = 4.0, std = 0.0.
        """
        model, tokenizer, V = _make_uniform_mocks(vocab_size=4)
        tok_out = _make_tokenizer_output([[1, 2, 3], [3, 2, 1]])
        tokenizer.return_value = tok_out

        result = compute_corpus_perplexity(
            ["sentence one", "sentence two"],
            model=model,
            tokenizer=tokenizer,
        )

        assert result.median_perplexity == _approx(float(V))
        assert result.std_perplexity == _approx(0.0)


# ---------------------------------------------------------------------------
# 7. Slow tests -- real model (gpt2)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCorpusPerplexityRealModel:
    """Slow tests with real GPT-2 model. Verify scientific properties."""

    def test_grammatical_lower_ppl_than_gibberish(self):
        """Grammatical text should have lower perplexity than random characters.

        Core scientific property: a language model trained on real text
        assigns higher probability (lower perplexity) to well-formed language.
        """
        result = compute_corpus_perplexity(
            [
                "The cat sat on the mat and looked out the window.",
                "xkq zmw bvt plr nfg ydc hjk wsx qqq zzz",
            ],
            model_name="gpt2",
            device="cpu",
        )

        assert result.per_sentence[0] < result.per_sentence[1]

    def test_corpus_ppl_differs_from_mean_ppl(self):
        """With heterogeneous text, corpus_ppl != mean_ppl.

        A short gibberish sentence and a long grammatical sentence
        should produce different corpus and mean perplexities because
        corpus_perplexity weights by token count.
        """
        result = compute_corpus_perplexity(
            [
                "xkq zmw",
                "The quick brown fox jumps over the lazy dog and then runs away into the forest.",
            ],
            model_name="gpt2",
            device="cpu",
        )

        # They shouldn't be exactly equal
        assert result.corpus_perplexity != result.mean_perplexity

    def test_all_scores_positive_and_finite(self):
        """Real perplexity values should be positive and finite."""
        result = compute_corpus_perplexity(
            ["Hello world.", "This is a test sentence."],
            model_name="gpt2",
            device="cpu",
        )

        assert result.corpus_perplexity > 0
        assert math.isfinite(result.corpus_perplexity)
        assert all(p > 0 and math.isfinite(p) for p in result.per_sentence)
        assert result.total_nll > 0
        assert result.num_tokens > 0
