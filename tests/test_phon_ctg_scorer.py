"""Tests for PhoneticScorer — evaluates text against phonetic targets."""

import pytest

from corpusgen.coverage.tracker import CoverageTracker
from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_ctg.scorer import PhoneticScorer, ScoreResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_inventory():
    """A small phoneme inventory for predictable testing."""
    return PhoneticTargetInventory(
        target_phonemes=["p", "b", "t", "d", "k"],
        unit="phoneme",
    )


@pytest.fixture
def weighted_inventory():
    """Inventory with explicit weights."""
    return PhoneticTargetInventory(
        target_phonemes=["p", "b", "t", "d", "k"],
        unit="phoneme",
        weights={"p": 10.0, "b": 5.0, "t": 3.0, "d": 2.0, "k": 1.0},
    )


@pytest.fixture
def simple_scorer(simple_inventory):
    """Scorer with no optional hooks."""
    return PhoneticScorer(targets=simple_inventory)


@pytest.fixture
def weighted_scorer(weighted_inventory):
    """Scorer with weighted inventory."""
    return PhoneticScorer(targets=weighted_inventory)


# ---------------------------------------------------------------------------
# ScoreResult structure
# ---------------------------------------------------------------------------


class TestScoreResult:
    """ScoreResult is a structured container for all scoring signals."""

    def test_has_required_fields(self):
        result = ScoreResult(
            text="hello",
            phonemes=["h", "ɛ", "l", "oʊ"],
            coverage_gain=2,
            weighted_coverage_gain=3.5,
            phonotactic_score=1.0,
            fluency_score=1.0,
            composite_score=5.5,
            new_units={"h", "ɛ"},
        )
        assert result.text == "hello"
        assert result.phonemes == ["h", "ɛ", "l", "oʊ"]
        assert result.coverage_gain == 2
        assert result.weighted_coverage_gain == 3.5
        assert result.phonotactic_score == 1.0
        assert result.fluency_score == 1.0
        assert result.composite_score == 5.5
        assert result.new_units == {"h", "ɛ"}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """PhoneticScorer construction and configuration."""

    def test_basic_creation(self, simple_inventory):
        scorer = PhoneticScorer(targets=simple_inventory)
        assert scorer.targets is simple_inventory

    def test_custom_composite_weights(self, simple_inventory):
        scorer = PhoneticScorer(
            targets=simple_inventory,
            coverage_weight=0.7,
            phonotactic_weight=0.2,
            fluency_weight=0.1,
        )
        assert scorer.coverage_weight == 0.7
        assert scorer.phonotactic_weight == 0.2
        assert scorer.fluency_weight == 0.1

    def test_default_composite_weights(self, simple_inventory):
        scorer = PhoneticScorer(targets=simple_inventory)
        assert scorer.coverage_weight == 1.0
        assert scorer.phonotactic_weight == 0.0
        assert scorer.fluency_weight == 0.0

    def test_with_phonotactic_hook(self, simple_inventory):
        def phono_fn(phonemes):
            return 0.9

        scorer = PhoneticScorer(
            targets=simple_inventory,
            phonotactic_scorer=phono_fn,
            phonotactic_weight=0.3,
        )
        assert scorer.phonotactic_weight == 0.3

    def test_with_fluency_hook(self, simple_inventory):
        def fluency_fn(text):
            return 0.8

        scorer = PhoneticScorer(
            targets=simple_inventory,
            fluency_scorer=fluency_fn,
            fluency_weight=0.2,
        )
        assert scorer.fluency_weight == 0.2


# ---------------------------------------------------------------------------
# Score (peek mode — non-destructive)
# ---------------------------------------------------------------------------


class TestScorePeek:
    """score() evaluates without modifying inventory state."""

    def test_score_with_phonemes(self, simple_scorer):
        result = simple_scorer.score(phonemes=["p", "b", "t"])
        assert isinstance(result, ScoreResult)
        assert result.coverage_gain == 3
        assert result.new_units == {"p", "b", "t"}

    def test_score_does_not_modify_inventory(self, simple_scorer):
        assert simple_scorer.targets.covered_count == 0
        simple_scorer.score(phonemes=["p", "b"])
        assert simple_scorer.targets.covered_count == 0  # unchanged

    def test_score_only_counts_new_units(self, simple_scorer):
        # Pre-cover some units
        simple_scorer.targets.update(["p", "b"], sentence_index=0)
        result = simple_scorer.score(phonemes=["p", "b", "t"])
        assert result.coverage_gain == 1  # only "t" is new
        assert result.new_units == {"t"}

    def test_score_with_no_new_units(self, simple_scorer):
        simple_scorer.targets.update(["p", "b", "t"], sentence_index=0)
        result = simple_scorer.score(phonemes=["p", "b", "t"])
        assert result.coverage_gain == 0
        assert result.new_units == set()

    def test_score_with_out_of_inventory_phonemes(self, simple_scorer):
        result = simple_scorer.score(phonemes=["x", "y", "z"])
        assert result.coverage_gain == 0
        assert result.new_units == set()

    def test_score_with_empty_phonemes(self, simple_scorer):
        result = simple_scorer.score(phonemes=[])
        assert result.coverage_gain == 0
        assert result.new_units == set()

    def test_score_preserves_text(self, simple_scorer):
        result = simple_scorer.score(text="test sentence", phonemes=["p", "b"])
        assert result.text == "test sentence"

    def test_score_preserves_phonemes(self, simple_scorer):
        result = simple_scorer.score(phonemes=["p", "b", "p"])
        assert result.phonemes == ["p", "b", "p"]


# ---------------------------------------------------------------------------
# Weighted coverage gain
# ---------------------------------------------------------------------------


class TestWeightedCoverageGain:
    """Weighted coverage gain uses target inventory weights."""

    def test_weighted_gain(self, weighted_scorer):
        # p=10.0, b=5.0 -> weighted gain = 15.0
        result = weighted_scorer.score(phonemes=["p", "b"])
        assert result.weighted_coverage_gain == pytest.approx(15.0)

    def test_unweighted_defaults_to_count(self, simple_scorer):
        # No weights -> each unit has weight 1.0 -> weighted gain == raw gain
        result = simple_scorer.score(phonemes=["p", "b", "t"])
        assert result.weighted_coverage_gain == pytest.approx(3.0)

    def test_weighted_gain_only_new_units(self, weighted_scorer):
        weighted_scorer.targets.update(["p"], sentence_index=0)
        # p already covered, b=5.0, t=3.0
        result = weighted_scorer.score(phonemes=["p", "b", "t"])
        assert result.weighted_coverage_gain == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


class TestCompositeScore:
    """Composite score combines coverage, phonotactic, and fluency."""

    def test_default_composite_is_weighted_coverage(self, simple_scorer):
        """With default weights (cov=1.0, phono=0.0, fluency=0.0)."""
        result = simple_scorer.score(phonemes=["p", "b"])
        assert result.composite_score == pytest.approx(result.weighted_coverage_gain)

    def test_composite_with_all_hooks(self, simple_inventory):
        def phono_fn(phonemes):
            return 0.8

        def fluency_fn(text):
            return 0.9

        scorer = PhoneticScorer(
            targets=simple_inventory,
            phonotactic_scorer=phono_fn,
            fluency_scorer=fluency_fn,
            coverage_weight=0.5,
            phonotactic_weight=0.3,
            fluency_weight=0.2,
        )
        result = scorer.score(text="some text", phonemes=["p", "b"])
        expected = (0.5 * result.weighted_coverage_gain) + (0.3 * 0.8) + (0.2 * 0.9)
        assert result.composite_score == pytest.approx(expected)

    def test_composite_without_hooks_ignores_weights(self, simple_inventory):
        """If no hook is provided, that component contributes 0 regardless of weight."""
        scorer = PhoneticScorer(
            targets=simple_inventory,
            phonotactic_weight=0.5,  # no hook -> 0 contribution
            fluency_weight=0.3,     # no hook -> 0 contribution
            coverage_weight=0.2,
        )
        result = scorer.score(phonemes=["p", "b"])
        expected = 0.2 * result.weighted_coverage_gain
        assert result.composite_score == pytest.approx(expected)

    def test_phonotactic_hook_receives_phonemes(self, simple_inventory):
        received = {}

        def phono_fn(phonemes):
            received["phonemes"] = phonemes
            return 1.0

        scorer = PhoneticScorer(
            targets=simple_inventory,
            phonotactic_scorer=phono_fn,
            phonotactic_weight=1.0,
        )
        scorer.score(phonemes=["p", "b", "t"])
        assert received["phonemes"] == ["p", "b", "t"]

    def test_fluency_hook_receives_text(self, simple_inventory):
        received = {}

        def fluency_fn(text):
            received["text"] = text
            return 1.0

        scorer = PhoneticScorer(
            targets=simple_inventory,
            fluency_scorer=fluency_fn,
            fluency_weight=1.0,
        )
        scorer.score(text="hello world", phonemes=["p"])
        assert received["text"] == "hello world"

    def test_fluency_hook_gets_none_when_no_text(self, simple_inventory):
        received = {}

        def fluency_fn(text):
            received["text"] = text
            return 0.5

        scorer = PhoneticScorer(
            targets=simple_inventory,
            fluency_scorer=fluency_fn,
            fluency_weight=1.0,
        )
        result = scorer.score(phonemes=["p"])
        # No text provided -> fluency_fn receives None -> still called
        assert received["text"] is None


# ---------------------------------------------------------------------------
# Score and commit
# ---------------------------------------------------------------------------


class TestScoreAndCommit:
    """score_and_commit() scores then updates the inventory."""

    def test_commit_updates_inventory(self, simple_scorer):
        assert simple_scorer.targets.covered_count == 0
        result = simple_scorer.score_and_commit(
            phonemes=["p", "b"], sentence_index=0
        )
        assert simple_scorer.targets.covered_count == 2
        assert result.coverage_gain == 2

    def test_commit_returns_same_score_as_peek(self, simple_scorer):
        peek = simple_scorer.score(phonemes=["p", "b"])
        # Reset to get same starting state (peek didn't modify)
        commit = simple_scorer.score_and_commit(
            phonemes=["p", "b"], sentence_index=0
        )
        assert commit.coverage_gain == peek.coverage_gain
        assert commit.weighted_coverage_gain == peek.weighted_coverage_gain

    def test_successive_commits(self, simple_scorer):
        simple_scorer.score_and_commit(phonemes=["p", "b"], sentence_index=0)
        result = simple_scorer.score_and_commit(
            phonemes=["b", "t", "d"], sentence_index=1
        )
        # b already covered, only t and d are new
        assert result.coverage_gain == 2
        assert result.new_units == {"t", "d"}
        assert simple_scorer.targets.covered_count == 4


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


class TestScoreBatch:
    """score_batch() scores multiple candidates without modifying state."""

    def test_batch_returns_list(self, simple_scorer):
        candidates = [
            {"phonemes": ["p", "b"]},
            {"phonemes": ["t", "d", "k"]},
        ]
        results = simple_scorer.score_batch(candidates)
        assert len(results) == 2
        assert all(isinstance(r, ScoreResult) for r in results)

    def test_batch_does_not_modify_inventory(self, simple_scorer):
        candidates = [
            {"phonemes": ["p", "b"]},
            {"phonemes": ["t", "d"]},
        ]
        simple_scorer.score_batch(candidates)
        assert simple_scorer.targets.covered_count == 0

    def test_batch_scores_independently(self, simple_scorer):
        """Each candidate is scored against current state, not cumulatively."""
        candidates = [
            {"phonemes": ["p", "b"]},
            {"phonemes": ["p", "t"]},  # p overlaps with first, but independent
        ]
        results = simple_scorer.score_batch(candidates)
        assert results[0].coverage_gain == 2
        assert results[1].coverage_gain == 2  # not 1, because non-destructive

    def test_batch_with_text(self, simple_scorer):
        candidates = [
            {"text": "first", "phonemes": ["p"]},
            {"text": "second", "phonemes": ["b"]},
        ]
        results = simple_scorer.score_batch(candidates)
        assert results[0].text == "first"
        assert results[1].text == "second"

    def test_batch_empty_list(self, simple_scorer):
        results = simple_scorer.score_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


class TestRank:
    """rank() returns candidates sorted by composite score descending."""

    def test_rank_ordering(self, simple_scorer):
        candidates = [
            {"phonemes": ["p"]},              # gain 1
            {"phonemes": ["p", "b", "t"]},    # gain 3
            {"phonemes": ["p", "b"]},          # gain 2
        ]
        ranked = simple_scorer.rank(candidates)
        assert len(ranked) == 3
        assert ranked[0].coverage_gain == 3
        assert ranked[1].coverage_gain == 2
        assert ranked[2].coverage_gain == 1

    def test_rank_does_not_modify_inventory(self, simple_scorer):
        candidates = [
            {"phonemes": ["p", "b"]},
            {"phonemes": ["t", "d"]},
        ]
        simple_scorer.rank(candidates)
        assert simple_scorer.targets.covered_count == 0

    def test_rank_with_weights(self, weighted_scorer):
        candidates = [
            {"phonemes": ["k"]},       # k=1.0 -> weighted gain 1.0
            {"phonemes": ["p"]},       # p=10.0 -> weighted gain 10.0
            {"phonemes": ["b", "t"]},  # b=5.0, t=3.0 -> weighted gain 8.0
        ]
        ranked = weighted_scorer.rank(candidates)
        assert ranked[0].weighted_coverage_gain == pytest.approx(10.0)
        assert ranked[1].weighted_coverage_gain == pytest.approx(8.0)
        assert ranked[2].weighted_coverage_gain == pytest.approx(1.0)

    def test_rank_empty_list(self, simple_scorer):
        ranked = simple_scorer.rank([])
        assert ranked == []

    def test_rank_with_top_k(self, simple_scorer):
        candidates = [
            {"phonemes": ["p"]},
            {"phonemes": ["p", "b"]},
            {"phonemes": ["p", "b", "t"]},
        ]
        ranked = simple_scorer.rank(candidates, top_k=2)
        assert len(ranked) == 2
        assert ranked[0].coverage_gain == 3
        assert ranked[1].coverage_gain == 2


# ---------------------------------------------------------------------------
# Diphone/triphone scoring
# ---------------------------------------------------------------------------


class TestNgramScoring:
    """Scoring works correctly with diphone and triphone units."""

    def test_diphone_scoring(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="diphone",
        )
        scorer = PhoneticScorer(targets=inv)
        # Phonemes [p, b, p] -> diphones: p-b, b-p
        result = scorer.score(phonemes=["p", "b", "p"])
        assert result.coverage_gain == 2
        assert "p-b" in result.new_units
        assert "b-p" in result.new_units

    def test_triphone_scoring(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="triphone",
        )
        scorer = PhoneticScorer(targets=inv)
        # Phonemes [p, b, p, b] -> triphones: p-b-p, b-p-b
        result = scorer.score(phonemes=["p", "b", "p", "b"])
        assert result.coverage_gain == 2
        assert "p-b-p" in result.new_units
        assert "b-p-b" in result.new_units


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions."""

    def test_score_requires_phonemes(self, simple_scorer):
        """Must provide phonemes (text alone is not enough yet — G2P integration later)."""
        with pytest.raises(TypeError):
            simple_scorer.score()

    def test_score_and_commit_requires_sentence_index(self, simple_scorer):
        with pytest.raises(TypeError):
            simple_scorer.score_and_commit(phonemes=["p"])

    def test_all_covered_gives_zero_gain(self, simple_scorer):
        simple_scorer.targets.update(["p", "b", "t", "d", "k"], sentence_index=0)
        result = simple_scorer.score(phonemes=["p", "b", "t"])
        assert result.coverage_gain == 0
        assert result.composite_score == pytest.approx(0.0)
