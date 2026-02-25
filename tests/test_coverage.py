"""Tests for the coverage tracking module."""

import pytest

from corpusgen.coverage.tracker import CoverageTracker


class TestCoverageTrackerInit:
    """Tests for CoverageTracker initialization."""

    def test_create_with_phoneme_list(self):
        """Create tracker with an explicit phoneme inventory."""
        phonemes = ["p", "b", "t", "d", "k"]
        tracker = CoverageTracker(target_phonemes=phonemes, unit="phoneme")
        assert tracker.unit == "phoneme"
        assert tracker.target_size == 5
        assert tracker.coverage == 0.0

    def test_create_diphone_tracker(self):
        """Create tracker targeting diphones."""
        phonemes = ["p", "b", "t"]
        tracker = CoverageTracker(target_phonemes=phonemes, unit="diphone")
        assert tracker.unit == "diphone"
        # 3 phonemes → up to 9 diphones (3x3 including self-pairs)
        assert tracker.target_size > 0

    def test_create_triphone_tracker(self):
        """Create tracker targeting triphones."""
        phonemes = ["p", "b", "t"]
        tracker = CoverageTracker(target_phonemes=phonemes, unit="triphone")
        assert tracker.unit == "triphone"

    def test_invalid_unit_raises(self):
        """Unknown unit type should raise ValueError."""
        with pytest.raises(ValueError, match="unit"):
            CoverageTracker(target_phonemes=["p"], unit="quadphone")


class TestCoverageTrackerUpdate:
    """Tests for updating coverage state."""

    @pytest.fixture
    def tracker(self):
        """A phoneme tracker with a small inventory."""
        return CoverageTracker(
            target_phonemes=["p", "b", "t", "d", "k", "ɡ", "s", "z"],
            unit="phoneme",
        )

    def test_update_with_phonemes(self, tracker):
        """Adding phonemes increases coverage."""
        tracker.update(phonemes=["p", "b", "t"], sentence_index=0)
        assert tracker.covered_count == 3
        assert tracker.coverage == 3 / 8

    def test_update_accumulates(self, tracker):
        """Multiple updates accumulate coverage."""
        tracker.update(phonemes=["p", "b"], sentence_index=0)
        tracker.update(phonemes=["t", "d", "k"], sentence_index=1)
        assert tracker.covered_count == 5
        assert tracker.coverage == 5 / 8

    def test_duplicate_phonemes_dont_inflate_coverage(self, tracker):
        """Seeing the same phoneme again doesn't increase covered count."""
        tracker.update(phonemes=["p", "p", "p"], sentence_index=0)
        assert tracker.covered_count == 1

    def test_phoneme_frequency_tracked(self, tracker):
        """Each occurrence is counted even if already covered."""
        tracker.update(phonemes=["p", "p", "b"], sentence_index=0)
        tracker.update(phonemes=["p"], sentence_index=1)
        counts = tracker.phoneme_counts
        assert counts["p"] == 3
        assert counts["b"] == 1

    def test_out_of_inventory_phonemes_ignored(self, tracker):
        """Phonemes not in the target inventory are not counted toward coverage."""
        tracker.update(phonemes=["x", "y", "ɻ"], sentence_index=0)
        assert tracker.covered_count == 0

    def test_missing_phonemes(self, tracker):
        """Returns the set of phonemes not yet covered."""
        tracker.update(phonemes=["p", "b", "t"], sentence_index=0)
        missing = tracker.missing
        assert missing == {"d", "k", "ɡ", "s", "z"}

    def test_full_coverage(self, tracker):
        """Coverage reaches 1.0 when all target phonemes are covered."""
        tracker.update(
            phonemes=["p", "b", "t", "d", "k", "ɡ", "s", "z"],
            sentence_index=0,
        )
        assert tracker.coverage == 1.0
        assert tracker.missing == set()

    def test_sentence_mapping(self, tracker):
        """Tracks which sentence introduced each phoneme."""
        tracker.update(phonemes=["p", "b"], sentence_index=0)
        tracker.update(phonemes=["t", "p"], sentence_index=1)
        mapping = tracker.phoneme_sources
        assert 0 in mapping["p"]
        assert 1 in mapping["p"]
        assert 0 in mapping["b"]
        assert 1 in mapping["t"]


class TestCoverageTrackerReset:
    """Tests for resetting tracker state."""

    def test_reset_clears_state(self):
        """Reset returns tracker to initial state."""
        tracker = CoverageTracker(target_phonemes=["p", "b", "t"], unit="phoneme")
        tracker.update(phonemes=["p", "b"], sentence_index=0)
        assert tracker.covered_count == 2
        tracker.reset()
        assert tracker.covered_count == 0
        assert tracker.coverage == 0.0
        assert tracker.missing == {"p", "b", "t"}
