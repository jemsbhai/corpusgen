"""Tests for PhoneticTargetInventory — dynamic target tracking for Phon-CTG."""

import pytest

from corpusgen.coverage.tracker import CoverageTracker
from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory


# ---------------------------------------------------------------------------
# Construction: standalone mode
# ---------------------------------------------------------------------------


class TestStandaloneConstruction:
    """PhoneticTargetInventory created with target_phonemes (no existing tracker)."""

    def test_basic_creation(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d"],
            unit="phoneme",
        )
        assert inv.unit == "phoneme"
        assert inv.target_size == 4
        assert inv.coverage == 0.0
        assert len(inv.missing) == 4

    def test_creates_internal_tracker(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        assert isinstance(inv.tracker, CoverageTracker)

    def test_diphone_unit(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="diphone",
        )
        # 2 phonemes -> 4 diphones: p-p, p-b, b-p, b-b
        assert inv.target_size == 4
        assert inv.unit == "diphone"

    def test_triphone_unit(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="triphone",
        )
        # 2 phonemes -> 8 triphones
        assert inv.target_size == 8

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Invalid unit"):
            PhoneticTargetInventory(target_phonemes=["p"], unit="quadphone")


# ---------------------------------------------------------------------------
# Construction: wrap mode (existing tracker)
# ---------------------------------------------------------------------------


class TestWrapConstruction:
    """PhoneticTargetInventory wrapping an existing CoverageTracker."""

    def test_wrap_existing_tracker(self):
        tracker = CoverageTracker(target_phonemes=["p", "b", "t"], unit="phoneme")
        tracker.update(["p", "b"], sentence_index=0)

        inv = PhoneticTargetInventory(tracker=tracker)
        assert inv.unit == "phoneme"
        assert inv.target_size == 3
        assert inv.covered_count == 2
        assert inv.coverage == pytest.approx(2 / 3)

    def test_wrap_shares_tracker_identity(self):
        tracker = CoverageTracker(target_phonemes=["p", "b"], unit="phoneme")
        inv = PhoneticTargetInventory(tracker=tracker)
        assert inv.tracker is tracker

    def test_wrap_inherits_unit(self):
        tracker = CoverageTracker(target_phonemes=["p", "b"], unit="diphone")
        inv = PhoneticTargetInventory(tracker=tracker)
        assert inv.unit == "diphone"


# ---------------------------------------------------------------------------
# Construction: validation
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    """Mutual exclusivity and required argument checks."""

    def test_both_tracker_and_phonemes_raises(self):
        tracker = CoverageTracker(target_phonemes=["p"], unit="phoneme")
        with pytest.raises(ValueError, match="[Cc]annot.*both"):
            PhoneticTargetInventory(
                tracker=tracker,
                target_phonemes=["p", "b"],
            )

    def test_neither_tracker_nor_phonemes_raises(self):
        with pytest.raises(ValueError, match="[Mm]ust provide"):
            PhoneticTargetInventory()

    def test_unit_ignored_when_wrapping(self):
        """When wrapping a tracker, unit comes from the tracker, not the arg."""
        tracker = CoverageTracker(target_phonemes=["p", "b"], unit="diphone")
        inv = PhoneticTargetInventory(tracker=tracker, unit="phoneme")
        # Should use tracker's unit, not the passed-in unit
        assert inv.unit == "diphone"


# ---------------------------------------------------------------------------
# Coverage delegation
# ---------------------------------------------------------------------------


class TestCoverageDelegation:
    """Properties that delegate to the underlying CoverageTracker."""

    def test_target_units(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        assert inv.target_units == {"p", "b", "t"}

    def test_covered_units_initially_empty(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        assert inv.covered_units == set()

    def test_missing_initially_full(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        assert inv.missing == {"p", "b"}

    def test_coverage_updates_after_update(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d"],
            unit="phoneme",
        )
        inv.update(["p", "b"], sentence_index=0)
        assert inv.covered_count == 2
        assert inv.coverage == pytest.approx(0.5)
        assert inv.missing == {"t", "d"}
        assert inv.covered_units == {"p", "b"}


# ---------------------------------------------------------------------------
# Weights and priority
# ---------------------------------------------------------------------------


class TestWeightsAndPriority:
    """Weighted prioritization of uncovered targets."""

    def test_default_uniform_weights(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        # Without weights, all targets should be present in next_targets
        targets = inv.next_targets(k=3)
        assert len(targets) == 3
        assert set(targets) == {"p", "b", "t"}

    def test_weighted_priority_ordering(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
            weights={"p": 10.0, "b": 1.0, "t": 5.0},
        )
        targets = inv.next_targets(k=3)
        assert targets[0] == "p"   # highest weight
        assert targets[1] == "t"   # second highest
        assert targets[2] == "b"   # lowest

    def test_next_targets_excludes_covered(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
            weights={"p": 10.0, "b": 5.0, "t": 1.0},
        )
        inv.update(["p"], sentence_index=0)
        targets = inv.next_targets(k=3)
        assert "p" not in targets
        assert targets[0] == "b"

    def test_next_targets_k_less_than_remaining(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d"],
            unit="phoneme",
            weights={"p": 4.0, "b": 3.0, "t": 2.0, "d": 1.0},
        )
        targets = inv.next_targets(k=2)
        assert len(targets) == 2
        assert targets == ["p", "b"]

    def test_next_targets_k_exceeds_remaining(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        inv.update(["p"], sentence_index=0)
        targets = inv.next_targets(k=5)
        assert targets == ["b"]

    def test_next_targets_all_covered_returns_empty(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        inv.update(["p", "b"], sentence_index=0)
        targets = inv.next_targets(k=5)
        assert targets == []

    def test_unweighted_targets_get_default_weight(self):
        """Weights dict doesn't need to cover all targets — missing ones get 1.0."""
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
            weights={"p": 10.0},  # b, t not specified
        )
        targets = inv.next_targets(k=3)
        assert targets[0] == "p"  # weight 10.0
        # b and t both have default weight 1.0, order among them is stable but unspecified


# ---------------------------------------------------------------------------
# Update and dynamic behavior
# ---------------------------------------------------------------------------


class TestUpdateDynamic:
    """Update method and dynamic re-prioritization."""

    def test_update_with_phoneme_list(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d"],
            unit="phoneme",
        )
        inv.update(["p", "t"], sentence_index=0)
        assert inv.covered_count == 2
        assert inv.missing == {"b", "d"}

    def test_successive_updates(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d"],
            unit="phoneme",
        )
        inv.update(["p"], sentence_index=0)
        inv.update(["b", "t"], sentence_index=1)
        assert inv.covered_count == 3
        assert inv.missing == {"d"}

    def test_update_with_diphones(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="diphone",
        )
        # Phoneme sequence [p, b, p] yields diphones: p-b, b-p
        inv.update(["p", "b", "p"], sentence_index=0)
        assert "p-b" in inv.covered_units
        assert "b-p" in inv.covered_units

    def test_priorities_shift_after_update(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
            weights={"p": 10.0, "b": 5.0, "t": 1.0},
        )
        assert inv.next_targets(k=1) == ["p"]
        inv.update(["p"], sentence_index=0)
        assert inv.next_targets(k=1) == ["b"]


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Reset clears coverage but preserves targets and weights."""

    def test_reset_clears_coverage(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        inv.update(["p", "b"], sentence_index=0)
        assert inv.covered_count == 2

        inv.reset()
        assert inv.covered_count == 0
        assert inv.coverage == 0.0
        assert inv.missing == {"p", "b", "t"}

    def test_reset_preserves_weights(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
            weights={"p": 10.0, "b": 5.0, "t": 1.0},
        )
        inv.update(["p", "b"], sentence_index=0)
        inv.reset()
        # After reset, priority should reflect original weights again
        assert inv.next_targets(k=1) == ["p"]

    def test_reset_preserves_target_size(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        inv.update(["p"], sentence_index=0)
        inv.reset()
        assert inv.target_size == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_empty_phoneme_list(self):
        inv = PhoneticTargetInventory(
            target_phonemes=[],
            unit="phoneme",
        )
        assert inv.target_size == 0
        assert inv.coverage == 1.0  # matches CoverageTracker convention
        assert inv.next_targets(k=5) == []

    def test_single_phoneme(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p"],
            unit="phoneme",
        )
        assert inv.target_size == 1
        inv.update(["p"], sentence_index=0)
        assert inv.coverage == 1.0

    def test_next_targets_k_zero(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        assert inv.next_targets(k=0) == []

    def test_update_with_empty_phoneme_list(self):
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        inv.update([], sentence_index=0)
        assert inv.covered_count == 0

    def test_update_with_out_of_inventory_phonemes(self):
        """Phonemes not in target inventory should be ignored gracefully."""
        inv = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        inv.update(["x", "y", "z"], sentence_index=0)
        assert inv.covered_count == 0
