"""Tests for the perplexity fluency scorer.

Test tiers:
    - **Fast tests** (default): Mock torch/transformers at the boundary.
      Tests initialization, scoring logic, normalization, edge cases.
    - **Slow tests** (@pytest.mark.slow): Real model (gpt2), real perplexity.

Scientific contract:
    - Grammatical sentences score higher than random character strings
    - Scores are in [0, 1] range (normalized)
    - Higher scores = more fluent / lower perplexity
    - Returns 0.0 for empty/None input
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Fast tests: mocked model
# ===========================================================================


def _make_mock_model_and_tokenizer():
    """Create mock model and tokenizer for fast tests.

    The mock model returns a fixed loss value when called, simulating
    the output of a causal LM forward pass.
    """
    import torch

    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"

    model = MagicMock()
    model.device = torch.device("cpu")
    model.eval = MagicMock(return_value=model)

    # Simulate forward pass returning a loss
    output = MagicMock()
    output.loss = torch.tensor(2.5)  # log-perplexity
    model.return_value = output

    return model, tokenizer


class TestPerplexityFluencyScorerConstruction:
    """Test PerplexityFluencyScorer initialization."""

    @patch("corpusgen.generate.scorers.fluency._load_tokenizer")
    @patch("corpusgen.generate.scorers.fluency._load_model")
    def test_lazy_loading(self, mock_load_model, mock_load_tok):
        """Model should not load until first call."""
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        scorer = PerplexityFluencyScorer(model_name="gpt2")
        assert not scorer.is_loaded
        mock_load_model.assert_not_called()
        mock_load_tok.assert_not_called()

    def test_from_model(self):
        """Construct from existing model/tokenizer objects."""
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = PerplexityFluencyScorer.from_model(model, tokenizer)
        assert scorer.is_loaded

    def test_model_name_stored(self):
        """model_name property should reflect construction arg."""
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        scorer = PerplexityFluencyScorer(model_name="gpt2")
        assert scorer.model_name == "gpt2"


class TestPerplexityFluencyScorerScoring:
    """Test scoring behavior with mocked models."""

    def test_returns_float(self):
        """Score should be a float."""
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = PerplexityFluencyScorer.from_model(model, tokenizer)
        score = scorer("The cat sat on the mat.")
        assert isinstance(score, float)

    def test_score_in_zero_one_range(self):
        """Score must be in [0, 1]."""
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = PerplexityFluencyScorer.from_model(model, tokenizer)
        score = scorer("The cat sat on the mat.")
        assert 0.0 <= score <= 1.0

    def test_none_returns_zero(self):
        """None input should return 0.0."""
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = PerplexityFluencyScorer.from_model(model, tokenizer)
        assert scorer(None) == 0.0

    def test_empty_string_returns_zero(self):
        """Empty string should return 0.0."""
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = PerplexityFluencyScorer.from_model(model, tokenizer)
        assert scorer("") == 0.0

    def test_whitespace_only_returns_zero(self):
        """Whitespace-only string should return 0.0."""
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = PerplexityFluencyScorer.from_model(model, tokenizer)
        assert scorer("   \n\t  ") == 0.0

    def test_callable_interface(self):
        """Scorer should work as callable for PhoneticScorer hook."""
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        model, tokenizer = _make_mock_model_and_tokenizer()
        scorer = PerplexityFluencyScorer.from_model(model, tokenizer)
        result = scorer("Hello world.")
        assert isinstance(result, float)

    def test_lower_loss_gives_higher_score(self):
        """Lower model loss (lower perplexity) should yield higher score."""
        import torch
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        model, tokenizer = _make_mock_model_and_tokenizer()

        # Score with loss = 1.0 (low perplexity)
        output_low = MagicMock()
        output_low.loss = torch.tensor(1.0)
        model.return_value = output_low
        scorer = PerplexityFluencyScorer.from_model(model, tokenizer)
        score_low_loss = scorer("Fluent sentence.")

        # Score with loss = 5.0 (high perplexity)
        output_high = MagicMock()
        output_high.loss = torch.tensor(5.0)
        model.return_value = output_high
        score_high_loss = scorer("Garbled nonsense xyz.")

        assert score_low_loss > score_high_loss

    @patch("corpusgen.generate.scorers.fluency._load_tokenizer")
    @patch("corpusgen.generate.scorers.fluency._load_model")
    def test_lazy_loads_on_first_call(self, mock_load_model, mock_load_tok):
        """Model should load on first scoring call."""
        import torch
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load_model.return_value = model
        mock_load_tok.return_value = tokenizer

        scorer = PerplexityFluencyScorer(model_name="gpt2")
        assert not scorer.is_loaded

        scorer("Hello world.")
        assert scorer.is_loaded
        mock_load_model.assert_called_once()
        mock_load_tok.assert_called_once()

    def test_high_loss_still_positive_score(self):
        """Even very high loss should produce score > 0."""
        import torch
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        model, tokenizer = _make_mock_model_and_tokenizer()
        output = MagicMock()
        output.loss = torch.tensor(100.0)
        model.return_value = output

        scorer = PerplexityFluencyScorer.from_model(model, tokenizer)
        score = scorer("Random text.")
        assert score > 0.0
        assert score <= 1.0


# ===========================================================================
# Slow tests: real model
# ===========================================================================


@pytest.mark.slow
class TestPerplexityFluencyScorerRealModel:
    """Slow tests with real gpt2 model. Verifies scientific properties."""

    def test_grammatical_higher_than_random(self):
        """Grammatical English should score higher than random characters.

        Core scientific property: perplexity-based fluency should
        prefer well-formed language over gibberish.
        """
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        scorer = PerplexityFluencyScorer(model_name="gpt2", device="cpu")

        grammatical = scorer("The cat sat on the mat and looked out the window.")
        gibberish = scorer("xkq zmw bvt plr nfg ydc hjk wsx")

        assert grammatical > gibberish

    def test_real_scores_in_range(self):
        """Real model scores should be in [0, 1]."""
        from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

        scorer = PerplexityFluencyScorer(model_name="gpt2", device="cpu")
        score = scorer("The quick brown fox jumps over the lazy dog.")
        assert 0.0 <= score <= 1.0
