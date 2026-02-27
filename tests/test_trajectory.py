"""Tests for corpusgen.evaluate.trajectory — coverage curve tracking.

Tests verify that CoverageTrajectory correctly computes step-by-step
coverage snapshots from an ordered sequence of phoneme lists against
a target inventory.
"""

from __future__ import annotations

import pytest

from corpusgen.evaluate.trajectory import (
    CoverageSnapshot,
    CoverageTrajectory,
    compute_coverage_trajectory,
)


def _approx(value: float, abs_tol: float = 1e-9):
    return pytest.approx(value, abs=abs_tol)


# ---------------------------------------------------------------------------
# 1. Basic trajectory computation — phoneme unit
# ---------------------------------------------------------------------------


class TestBasicPhonemeTrajectory:
    """Step-by-step coverage for simple phoneme targets."""

    @pytest.fixture()
    def trajectory(self) -> CoverageTrajectory:
        # Target: {a, b, c, d}
        # Sentence 0 covers a, b  → 2/4 = 0.5
        # Sentence 1 covers b, c  → 3/4 = 0.75  (c is new)
        # Sentence 2 covers d     → 4/4 = 1.0
        phoneme_sequences = [
            ["a", "b"],
            ["b", "c"],
            ["d"],
        ]
        target_units = {"a", "b", "c", "d"}
        return compute_coverage_trajectory(phoneme_sequences, target_units)

    def test_snapshot_count(self, trajectory):
        assert len(trajectory.snapshots) == 3

    def test_cumulative_coverage(self, trajectory):
        assert trajectory.snapshots[0].coverage == _approx(0.5)
        assert trajectory.snapshots[1].coverage == _approx(0.75)
        assert trajectory.snapshots[2].coverage == _approx(1.0)

    def test_covered_count(self, trajectory):
        assert trajectory.snapshots[0].covered_count == 2
        assert trajectory.snapshots[1].covered_count == 3
        assert trajectory.snapshots[2].covered_count == 4

    def test_new_units_count(self, trajectory):
        assert trajectory.snapshots[0].new_units_count == 2
        assert trajectory.snapshots[1].new_units_count == 1
        assert trajectory.snapshots[2].new_units_count == 1

    def test_new_units_content(self, trajectory):
        assert set(trajectory.snapshots[0].new_units) == {"a", "b"}
        assert set(trajectory.snapshots[1].new_units) == {"c"}
        assert set(trajectory.snapshots[2].new_units) == {"d"}

    def test_sentence_indices(self, trajectory):
        assert trajectory.snapshots[0].sentence_index == 0
        assert trajectory.snapshots[1].sentence_index == 1
        assert trajectory.snapshots[2].sentence_index == 2

    def test_target_size(self, trajectory):
        assert trajectory.target_size == 4

    def test_unit_default(self, trajectory):
        assert trajectory.unit == "phoneme"


# ---------------------------------------------------------------------------
# 2. Convenience properties for plotting
# ---------------------------------------------------------------------------


class TestTrajectoryProperties:
    """coverages and gains properties for easy plotting."""

    @pytest.fixture()
    def trajectory(self) -> CoverageTrajectory:
        phoneme_sequences = [["a", "b"], ["c"], ["c", "d"]]
        target_units = {"a", "b", "c", "d"}
        return compute_coverage_trajectory(phoneme_sequences, target_units)

    def test_coverages_list(self, trajectory):
        assert len(trajectory.coverages) == 3
        assert trajectory.coverages[0] == _approx(0.5)
        assert trajectory.coverages[1] == _approx(0.75)
        assert trajectory.coverages[2] == _approx(1.0)

    def test_gains_list(self, trajectory):
        assert trajectory.gains == [2, 1, 1]


# ---------------------------------------------------------------------------
# 3. Diphone unit
# ---------------------------------------------------------------------------


class TestDiphoneTrajectory:
    """Coverage trajectory for diphone units."""

    def test_diphone_extraction(self):
        # Phonemes: [a, b, c] → diphones: a-b, b-c
        # Phonemes: [b, c, d] → diphones: b-c, c-d
        phoneme_sequences = [
            ["a", "b", "c"],
            ["b", "c", "d"],
        ]
        target_units = {"a-b", "b-c", "c-d"}
        traj = compute_coverage_trajectory(
            phoneme_sequences, target_units, unit="diphone"
        )
        assert traj.snapshots[0].covered_count == 2  # a-b, b-c
        assert traj.snapshots[0].new_units_count == 2
        assert traj.snapshots[1].covered_count == 3  # +c-d
        assert traj.snapshots[1].new_units_count == 1
        assert set(traj.snapshots[1].new_units) == {"c-d"}

    def test_diphone_unit_field(self):
        traj = compute_coverage_trajectory(
            [["a", "b"]], {"a-b"}, unit="diphone"
        )
        assert traj.unit == "diphone"


# ---------------------------------------------------------------------------
# 4. Triphone unit
# ---------------------------------------------------------------------------


class TestTriphoneTrajectory:
    """Coverage trajectory for triphone units."""

    def test_triphone_extraction(self):
        # [a, b, c, d] → triphones: a-b-c, b-c-d
        phoneme_sequences = [["a", "b", "c", "d"]]
        target_units = {"a-b-c", "b-c-d", "x-y-z"}
        traj = compute_coverage_trajectory(
            phoneme_sequences, target_units, unit="triphone"
        )
        assert traj.snapshots[0].covered_count == 2
        assert traj.snapshots[0].coverage == _approx(2 / 3)
        assert set(traj.snapshots[0].new_units) == {"a-b-c", "b-c-d"}


# ---------------------------------------------------------------------------
# 5. Zero-gain sentences (redundant)
# ---------------------------------------------------------------------------


class TestRedundantSentences:
    """Sentences that add no new coverage."""

    def test_zero_gain_still_recorded(self):
        """Every sentence gets a snapshot, even if gain is 0."""
        phoneme_sequences = [
            ["a", "b"],
            ["a", "b"],  # redundant
            ["c"],
        ]
        target_units = {"a", "b", "c"}
        traj = compute_coverage_trajectory(phoneme_sequences, target_units)
        assert len(traj.snapshots) == 3
        assert traj.snapshots[1].new_units_count == 0
        assert traj.snapshots[1].new_units == []
        # Coverage stays flat
        assert traj.snapshots[1].coverage == traj.snapshots[0].coverage

    def test_gains_reflect_zero(self):
        phoneme_sequences = [["a"], ["a"], ["b"]]
        target_units = {"a", "b"}
        traj = compute_coverage_trajectory(phoneme_sequences, target_units)
        assert traj.gains == [1, 0, 1]


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestTrajectoryEdgeCases:
    """Degenerate inputs."""

    def test_empty_sequences(self):
        """No sentences → empty trajectory."""
        traj = compute_coverage_trajectory([], {"a", "b"})
        assert len(traj.snapshots) == 0
        assert traj.coverages == []
        assert traj.gains == []
        assert traj.target_size == 2

    def test_empty_target(self):
        """Empty target → all snapshots have coverage 1.0."""
        traj = compute_coverage_trajectory([["a", "b"]], set())
        assert len(traj.snapshots) == 1
        assert traj.snapshots[0].coverage == _approx(1.0)
        assert traj.snapshots[0].new_units_count == 0
        assert traj.target_size == 0

    def test_empty_phoneme_list(self):
        """A sentence with no phonemes contributes nothing."""
        traj = compute_coverage_trajectory([[], ["a"]], {"a", "b"})
        assert traj.snapshots[0].new_units_count == 0
        assert traj.snapshots[0].coverage == _approx(0.0)
        assert traj.snapshots[1].new_units_count == 1

    def test_non_target_phonemes_ignored(self):
        """Phonemes not in target set don't count."""
        traj = compute_coverage_trajectory([["x", "y", "z"]], {"a"})
        assert traj.snapshots[0].new_units_count == 0
        assert traj.snapshots[0].coverage == _approx(0.0)

    def test_single_sentence_full_coverage(self):
        traj = compute_coverage_trajectory([["a", "b"]], {"a", "b"})
        assert len(traj.snapshots) == 1
        assert traj.snapshots[0].coverage == _approx(1.0)
        assert traj.snapshots[0].new_units_count == 2


# ---------------------------------------------------------------------------
# 7. Snapshot dataclass basics
# ---------------------------------------------------------------------------


class TestSnapshotDataclass:
    """CoverageSnapshot is well-behaved."""

    def test_fields_accessible(self):
        s = CoverageSnapshot(
            sentence_index=0,
            coverage=0.5,
            covered_count=2,
            new_units_count=2,
            new_units=["a", "b"],
        )
        assert s.sentence_index == 0
        assert s.coverage == 0.5
        assert s.covered_count == 2
        assert s.new_units_count == 2
        assert s.new_units == ["a", "b"]


# ---------------------------------------------------------------------------
# 8. to_dict export
# ---------------------------------------------------------------------------


class TestTrajectoryExport:
    """CoverageTrajectory should export to a dict."""

    def test_to_dict_structure(self):
        traj = compute_coverage_trajectory(
            [["a", "b"], ["c"]], {"a", "b", "c"}
        )
        d = traj.to_dict()
        assert isinstance(d, dict)
        assert "unit" in d
        assert "target_size" in d
        assert "snapshots" in d
        assert len(d["snapshots"]) == 2
        snap = d["snapshots"][0]
        assert "sentence_index" in snap
        assert "coverage" in snap
        assert "covered_count" in snap
        assert "new_units_count" in snap
        assert "new_units" in snap
