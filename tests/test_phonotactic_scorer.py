"""Tests for the N-gram phonotactic scorer.

Test tiers:
    - **Fast tests** (default): Pure logic, no external dependencies.
      Tests n-gram construction, scoring, smoothing, edge cases.
    - **Slow tests** (@pytest.mark.slow): Real G2P + PHOIBLE data.

Scientific contract:
    - In-inventory transitions score higher than out-of-inventory transitions
    - Corpus-trained model reflects observed frequencies
    - Scores are in [0, 1] range
    - Higher scores = more phonotactically natural
"""

from __future__ import annotations

import pytest


# ===========================================================================
# Fast tests: N-gram construction and scoring
# ===========================================================================


class TestNgramPhonotacticScorerConstruction:
    """Test NgramPhonotacticScorer initialization and n-gram building."""

    def test_bigram_default(self):
        """Default n=2 builds bigram model from phoneme inventory."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t"])
        assert scorer.n == 2
        assert scorer.phonemes == ["p", "a", "t"]

    def test_trigram(self):
        """n=3 builds trigram model."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t"], n=3)
        assert scorer.n == 3

    def test_invalid_n_raises(self):
        """n < 2 should raise ValueError."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        with pytest.raises(ValueError, match="n must be >= 2"):
            NgramPhonotacticScorer(phonemes=["p", "a"], n=1)

    def test_empty_phonemes_raises(self):
        """Empty phoneme list should raise ValueError."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        with pytest.raises(ValueError, match="phonemes"):
            NgramPhonotacticScorer(phonemes=[])

    def test_single_phoneme_raises(self):
        """Single-phoneme inventory can't form bigrams — should raise."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        with pytest.raises(ValueError, match="at least 2"):
            NgramPhonotacticScorer(phonemes=["p"])


class TestNgramPhonotacticScorerScoring:
    """Test scoring behavior and phonotactic properties."""

    def test_returns_float(self):
        """Score should be a float."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t", "k"])
        score = scorer(["p", "a", "t"])
        assert isinstance(score, float)

    def test_score_in_zero_one_range(self):
        """Score must be in [0, 1]."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t", "k"])
        # In-inventory sequence
        score = scorer(["p", "a", "t", "k", "p"])
        assert 0.0 <= score <= 1.0

    def test_score_in_zero_one_for_out_of_inventory(self):
        """Out-of-inventory phonemes should still produce score in [0, 1]."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t"])
        # 'z' is not in inventory
        score = scorer(["z", "z", "z"])
        assert 0.0 <= score <= 1.0

    def test_in_inventory_scores_higher_than_out(self):
        """Sequences using only inventory phonemes should score higher
        than sequences with unknown phonemes.

        This is the core scientific property: the Laplace-smoothed model
        should prefer in-inventory transitions.
        """
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        phonemes = ["p", "a", "t", "k", "i", "s"]
        scorer = NgramPhonotacticScorer(phonemes=phonemes)

        in_score = scorer(["p", "a", "t", "a", "k", "i"])
        out_score = scorer(["z", "ʒ", "ð", "ɣ", "ʁ", "ɮ"])
        assert in_score > out_score

    def test_empty_phoneme_list_returns_zero(self):
        """Scoring an empty phoneme list should return 0.0."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t"])
        assert scorer([]) == 0.0

    def test_single_phoneme_returns_zero(self):
        """Single phoneme can't form a bigram — score should be 0.0."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t"])
        assert scorer(["p"]) == 0.0

    def test_callable_interface(self):
        """Scorer should be usable as a callable for PhoneticScorer hook."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t", "k"])
        # Should work when called like a function
        result = scorer(["p", "a", "t"])
        assert isinstance(result, float)

    def test_trigram_scoring(self):
        """Trigram model should produce valid scores."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t", "k"], n=3)
        score = scorer(["p", "a", "t", "k", "a"])
        assert 0.0 <= score <= 1.0

    def test_trigram_needs_at_least_three_phonemes(self):
        """Trigram scoring with < 3 phonemes should return 0.0."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t", "k"], n=3)
        assert scorer(["p", "a"]) == 0.0


class TestNgramPhonotacticScorerFromCorpus:
    """Test corpus-trained mode."""

    def test_from_corpus_basic(self):
        """Build a scorer from observed phoneme sequences."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        sequences = [
            ["p", "a", "t"],
            ["t", "a", "p"],
            ["p", "a", "k"],
            ["k", "a", "t"],
        ]
        scorer = NgramPhonotacticScorer.from_corpus(sequences, n=2)
        score = scorer(["p", "a", "t"])
        assert 0.0 <= score <= 1.0

    def test_from_corpus_reflects_frequencies(self):
        """Frequent transitions should score higher than rare ones.

        Scientific property: corpus-trained model should reflect
        observed bigram frequencies, not just inventory membership.
        """
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        # "p-a" appears many times, "k-i" appears once
        sequences = [
            ["p", "a", "t"],
            ["p", "a", "k"],
            ["p", "a", "s"],
            ["p", "a", "t"],
            ["p", "a", "k"],
            ["k", "i", "t"],  # k-i only once
        ]
        scorer = NgramPhonotacticScorer.from_corpus(sequences, n=2)

        # Sequence dominated by frequent "p-a" transition
        freq_score = scorer(["p", "a", "p", "a", "p", "a"])
        # Sequence dominated by rare "k-i" transition
        rare_score = scorer(["k", "i", "k", "i", "k", "i"])
        assert freq_score > rare_score

    def test_from_corpus_empty_raises(self):
        """Empty corpus should raise ValueError."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        with pytest.raises(ValueError, match="corpus"):
            NgramPhonotacticScorer.from_corpus([], n=2)

    def test_from_corpus_short_sequences_skipped(self):
        """Sequences shorter than n should be skipped, not crash."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        sequences = [
            ["p"],          # too short for bigram
            ["a"],          # too short
            ["p", "a", "t"],  # valid
        ]
        scorer = NgramPhonotacticScorer.from_corpus(sequences, n=2)
        score = scorer(["p", "a", "t"])
        assert 0.0 <= score <= 1.0

    def test_from_corpus_all_short_raises(self):
        """If ALL sequences are too short, should raise."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        with pytest.raises(ValueError, match="n-gram"):
            NgramPhonotacticScorer.from_corpus([["p"], ["a"]], n=2)


class TestNgramPhonotacticScorerPersistence:
    """Test save/load for reproducibility."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved model should produce identical scores after loading."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t", "k", "i"])
        path = tmp_path / "model.json"
        scorer.save(path)

        loaded = NgramPhonotacticScorer.load(path)
        seq = ["p", "a", "t", "k", "i", "p", "a"]
        assert scorer(seq) == loaded(seq)

    def test_save_and_load_preserves_n(self, tmp_path):
        """N-gram order should survive save/load."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t", "k"], n=3)
        path = tmp_path / "model.json"
        scorer.save(path)

        loaded = NgramPhonotacticScorer.load(path)
        assert loaded.n == 3

    def test_save_and_load_corpus_trained(self, tmp_path):
        """Corpus-trained model should roundtrip correctly."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        sequences = [
            ["p", "a", "t"],
            ["t", "a", "p"],
            ["p", "a", "k"],
        ]
        scorer = NgramPhonotacticScorer.from_corpus(sequences, n=2)
        path = tmp_path / "corpus_model.json"
        scorer.save(path)

        loaded = NgramPhonotacticScorer.load(path)
        seq = ["p", "a", "t"]
        assert scorer(seq) == loaded(seq)

    def test_load_nonexistent_raises(self):
        """Loading from a nonexistent path should raise FileNotFoundError."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        with pytest.raises(FileNotFoundError):
            NgramPhonotacticScorer.load("/nonexistent/path/model.json")

    def test_save_creates_file(self, tmp_path):
        """Save should create a JSON file."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t"])
        path = tmp_path / "model.json"
        scorer.save(path)
        assert path.exists()
        assert path.stat().st_size > 0


class TestNgramPhonotacticScorerSmoothing:
    """Test Laplace smoothing properties."""

    def test_unseen_bigram_gets_nonzero_score(self):
        """Laplace smoothing: unseen transitions get small but nonzero probability."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        # Small inventory — not all bigrams observed in uniform model
        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t"])
        # Score any sequence — should never be exactly 0
        # (unless too short, which is handled separately)
        score = scorer(["p", "a", "t", "p"])
        assert score > 0.0

    def test_smoothing_prevents_zero_probability(self):
        """Even completely out-of-inventory sequences should score > 0."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        scorer = NgramPhonotacticScorer(phonemes=["p", "a", "t"])
        score = scorer(["z", "ʒ", "ð"])
        assert score > 0.0

    def test_corpus_smoothing(self):
        """Corpus-trained model should also smooth unseen transitions."""
        from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

        sequences = [["p", "a", "t"], ["t", "a", "p"]]
        scorer = NgramPhonotacticScorer.from_corpus(sequences, n=2)

        # "p-t" never observed directly in corpus
        score = scorer(["p", "t", "a"])
        assert score > 0.0
