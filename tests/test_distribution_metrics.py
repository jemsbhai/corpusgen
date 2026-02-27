"""Tests for corpusgen.evaluate.distribution — distribution quality metrics.

Tests are organized by metric category, then edge cases.
All expected values are computed by hand or from known mathematical identities.
"""

from __future__ import annotations

import math
import pytest

from corpusgen.evaluate.distribution import (
    DistributionMetrics,
    compute_distribution_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approx(value: float, abs_tol: float = 1e-9) -> pytest.approx:
    """Shorthand for pytest.approx with a tight absolute tolerance."""
    return pytest.approx(value, abs=abs_tol)


# ---------------------------------------------------------------------------
# 1. Perfectly uniform counts
#    All target units have equal counts → maximum entropy, zero divergence.
# ---------------------------------------------------------------------------

class TestUniformDistribution:
    """When every target unit has the same count, distribution is ideal."""

    def test_entropy_equals_log2_n(self):
        counts = {"a": 10, "b": 10, "c": 10, "d": 10}
        targets = ["a", "b", "c", "d"]
        m = compute_distribution_metrics(counts, targets)
        # H = log2(4) = 2.0
        assert m.entropy == _approx(2.0)

    def test_normalized_entropy_is_one(self):
        counts = {"a": 5, "b": 5, "c": 5}
        targets = ["a", "b", "c"]
        m = compute_distribution_metrics(counts, targets)
        assert m.normalized_entropy == _approx(1.0)

    def test_jsd_uniform_is_zero(self):
        counts = {"a": 7, "b": 7}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        assert m.jsd_uniform == _approx(0.0)

    def test_cv_is_zero(self):
        counts = {"a": 3, "b": 3, "c": 3, "d": 3}
        targets = ["a", "b", "c", "d"]
        m = compute_distribution_metrics(counts, targets)
        assert m.coefficient_of_variation == _approx(0.0)

    def test_count_ratio_is_one(self):
        counts = {"a": 4, "b": 4}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        assert m.count_ratio == _approx(1.0)

    def test_pcd_uniform_equals_coverage(self):
        """PCD = coverage * (1 - JSD_uniform).  JSD=0 → PCD = coverage."""
        counts = {"a": 2, "b": 2}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        assert m.pcd_uniform == _approx(1.0)


# ---------------------------------------------------------------------------
# 2. Maximally skewed distribution
#    All occurrences on a single unit → low entropy, high JSD.
# ---------------------------------------------------------------------------

class TestSkewedDistribution:
    """All counts concentrated on one unit out of many."""

    def test_entropy_is_zero_single_nonzero(self):
        counts = {"a": 100, "b": 0, "c": 0, "d": 0}
        targets = ["a", "b", "c", "d"]
        m = compute_distribution_metrics(counts, targets)
        # Only one unit has mass → H = 0
        assert m.entropy == _approx(0.0)

    def test_normalized_entropy_is_zero(self):
        counts = {"a": 100, "b": 0, "c": 0, "d": 0}
        targets = ["a", "b", "c", "d"]
        m = compute_distribution_metrics(counts, targets)
        assert m.normalized_entropy == _approx(0.0)

    def test_jsd_uniform_is_positive(self):
        counts = {"a": 100, "b": 0, "c": 0, "d": 0}
        targets = ["a", "b", "c", "d"]
        m = compute_distribution_metrics(counts, targets)
        # JSD must be > 0 and ≤ 1
        assert m.jsd_uniform > 0.0
        assert m.jsd_uniform <= 1.0

    def test_jsd_uniform_known_value(self):
        """Hand-computed JSD for P=[1,0] vs Q=[0.5,0.5] (base 2).

        M = [0.75, 0.25]
        KL(P||M) = 1*log2(1/0.75) + 0*log2(0/0.25) = log2(4/3)
        KL(Q||M) = 0.5*log2(0.5/0.75) + 0.5*log2(0.5/0.25)
                 = 0.5*log2(2/3) + 0.5*log2(2)
                 = 0.5*log2(2/3) + 0.5
        JSD = 0.5*log2(4/3) + 0.5*(0.5*log2(2/3) + 0.5)
        """
        counts = {"a": 100, "b": 0}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)

        kl_p_m = math.log2(4 / 3)
        kl_q_m = 0.5 * math.log2(2 / 3) + 0.5
        expected_jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
        assert m.jsd_uniform == _approx(expected_jsd)

    def test_zero_count_is_correct(self):
        counts = {"a": 100, "b": 0, "c": 0, "d": 0}
        targets = ["a", "b", "c", "d"]
        m = compute_distribution_metrics(counts, targets)
        assert m.zero_count == 3

    def test_min_count_is_zero(self):
        counts = {"a": 50, "b": 0}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        assert m.min_count == 0

    def test_count_ratio_is_zero_when_min_is_zero(self):
        counts = {"a": 50, "b": 0}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        assert m.count_ratio == _approx(0.0)


# ---------------------------------------------------------------------------
# 3. Intermediate / realistic distribution
#    Verify metrics for a non-trivial case with hand-computed values.
# ---------------------------------------------------------------------------

class TestIntermediateDistribution:
    """A realistic but simple distribution for exact verification."""

    @pytest.fixture()
    def metrics(self):
        # Counts: a=6, b=3, c=1.  Total=10.  P = [0.6, 0.3, 0.1]
        counts = {"a": 6, "b": 3, "c": 1}
        targets = ["a", "b", "c"]
        return compute_distribution_metrics(counts, targets)

    def test_entropy(self, metrics):
        # H = -(0.6*log2(0.6) + 0.3*log2(0.3) + 0.1*log2(0.1))
        expected = -(
            0.6 * math.log2(0.6)
            + 0.3 * math.log2(0.3)
            + 0.1 * math.log2(0.1)
        )
        assert metrics.entropy == _approx(expected)

    def test_normalized_entropy(self, metrics):
        h = -(
            0.6 * math.log2(0.6)
            + 0.3 * math.log2(0.3)
            + 0.1 * math.log2(0.1)
        )
        expected = h / math.log2(3)
        assert metrics.normalized_entropy == _approx(expected)

    def test_jsd_uniform(self, metrics):
        # P = [0.6, 0.3, 0.1], Q = [1/3, 1/3, 1/3]
        p = [0.6, 0.3, 0.1]
        q = [1 / 3, 1 / 3, 1 / 3]
        m_dist = [(pi + qi) / 2 for pi, qi in zip(p, q)]
        kl_p_m = sum(
            pi * math.log2(pi / mi) for pi, mi in zip(p, m_dist) if pi > 0
        )
        kl_q_m = sum(
            qi * math.log2(qi / mi) for qi, mi in zip(q, m_dist) if qi > 0
        )
        expected = 0.5 * kl_p_m + 0.5 * kl_q_m
        assert metrics.jsd_uniform == _approx(expected)

    def test_cv(self, metrics):
        import statistics
        counts_list = [6, 3, 1]
        mean = statistics.mean(counts_list)
        stdev = statistics.pstdev(counts_list)  # population stdev
        expected = stdev / mean
        assert metrics.coefficient_of_variation == _approx(expected)

    def test_min_max(self, metrics):
        assert metrics.min_count == 1
        assert metrics.max_count == 6

    def test_count_ratio(self, metrics):
        assert metrics.count_ratio == _approx(1 / 6)

    def test_zero_count(self, metrics):
        assert metrics.zero_count == 0

    def test_pcd_uniform(self, metrics):
        # coverage = 3/3 = 1.0, PCD = 1.0 * (1 - jsd)
        expected = 1.0 * (1.0 - metrics.jsd_uniform)
        assert metrics.pcd_uniform == _approx(expected)


# ---------------------------------------------------------------------------
# 4. Reference distribution
#    JSD and Pearson against a user-supplied reference.
# ---------------------------------------------------------------------------

class TestReferenceDistribution:
    """Metrics computed against a user-supplied reference distribution."""

    def test_jsd_reference_identical_is_zero(self):
        """Corpus distribution matches reference exactly → JSD = 0."""
        counts = {"a": 60, "b": 30, "c": 10}
        targets = ["a", "b", "c"]
        # Reference proportional to corpus: [0.6, 0.3, 0.1]
        reference = {"a": 0.6, "b": 0.3, "c": 0.1}
        m = compute_distribution_metrics(counts, targets, reference)
        assert m.jsd_reference == _approx(0.0)

    def test_jsd_reference_different_is_positive(self):
        counts = {"a": 60, "b": 30, "c": 10}
        targets = ["a", "b", "c"]
        # Reference is uniform — different from corpus
        reference = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        m = compute_distribution_metrics(counts, targets, reference)
        assert m.jsd_reference is not None
        assert m.jsd_reference > 0.0

    def test_jsd_reference_bounded(self):
        """JSD is always in [0, 1] with base-2 log."""
        counts = {"a": 100, "b": 0}
        targets = ["a", "b"]
        reference = {"a": 0.0, "b": 1.0}
        m = compute_distribution_metrics(counts, targets, reference)
        assert m.jsd_reference is not None
        assert 0.0 <= m.jsd_reference <= 1.0

    def test_pearson_perfect_positive(self):
        """When corpus distribution is proportional to reference → r = 1.0."""
        counts = {"a": 60, "b": 30, "c": 10}
        targets = ["a", "b", "c"]
        reference = {"a": 0.6, "b": 0.3, "c": 0.1}
        m = compute_distribution_metrics(counts, targets, reference)
        assert m.pearson_correlation == _approx(1.0)

    def test_pearson_perfect_negative(self):
        """Corpus counts are perfectly negatively linear with reference.

        Counts [1, 2, 3] vs reference [3, 2, 1] (normalizes to
        [0.5, 1/3, 1/6]).  Deviations: x=[-1,0,1], y=[1/6,0,-1/6]
        → y = -x/6, perfectly negative linear → r = -1.0.
        """
        counts = {"a": 1, "b": 2, "c": 3}
        targets = ["a", "b", "c"]
        reference = {"a": 3.0, "b": 2.0, "c": 1.0}
        m = compute_distribution_metrics(counts, targets, reference)
        assert m.pearson_correlation == _approx(-1.0)

    def test_pearson_is_none_without_reference(self):
        counts = {"a": 5, "b": 5}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        assert m.pearson_correlation is None

    def test_jsd_reference_is_none_without_reference(self):
        counts = {"a": 5, "b": 5}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        assert m.jsd_reference is None

    def test_reference_unnormalized_is_normalized(self):
        """Reference values that don't sum to 1.0 should be auto-normalized."""
        counts = {"a": 60, "b": 30, "c": 10}
        targets = ["a", "b", "c"]
        # Unnormalized reference (sums to 10, not 1.0)
        reference = {"a": 6.0, "b": 3.0, "c": 1.0}
        m = compute_distribution_metrics(counts, targets, reference)
        # After normalization, reference matches corpus → JSD ≈ 0
        assert m.jsd_reference == _approx(0.0)

    def test_reference_missing_units_treated_as_zero(self):
        """Reference that omits some target units → those get weight 0."""
        counts = {"a": 50, "b": 50}
        targets = ["a", "b"]
        # Reference only mentions "a"
        reference = {"a": 1.0}
        m = compute_distribution_metrics(counts, targets, reference)
        assert m.jsd_reference is not None
        assert m.jsd_reference > 0.0


# ---------------------------------------------------------------------------
# 5. Edge cases: empty and degenerate inputs
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Degenerate inputs must produce well-defined, meaningful results."""

    def test_empty_corpus_no_counts(self):
        """No phonemes observed at all."""
        counts: dict[str, int] = {}
        targets = ["a", "b", "c"]
        m = compute_distribution_metrics(counts, targets)
        assert m.entropy == _approx(0.0)
        assert m.normalized_entropy == _approx(0.0)
        assert m.zero_count == 3
        assert m.min_count == 0
        assert m.max_count == 0

    def test_empty_corpus_jsd_is_max(self):
        """Empty corpus vs uniform → maximum divergence."""
        counts: dict[str, int] = {}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        # JSD(zero_dist, uniform) should be well-defined
        # P is all-zero → but we can't form a valid probability distribution.
        # Convention: JSD = 1.0 (maximal divergence) when corpus is empty.
        assert m.jsd_uniform == _approx(1.0)

    def test_empty_target_list(self):
        """No target phonemes → trivially perfect."""
        counts = {"a": 5}
        targets: list[str] = []
        m = compute_distribution_metrics(counts, targets)
        assert m.entropy == _approx(0.0)
        assert m.normalized_entropy == _approx(1.0)
        assert m.jsd_uniform == _approx(0.0)
        assert m.zero_count == 0
        assert m.pcd_uniform == _approx(1.0)

    def test_single_target_unit(self):
        """Single target unit → trivially uniform."""
        counts = {"a": 42}
        targets = ["a"]
        m = compute_distribution_metrics(counts, targets)
        assert m.entropy == _approx(0.0)
        assert m.normalized_entropy == _approx(1.0)  # convention for N=1
        assert m.jsd_uniform == _approx(0.0)
        assert m.coefficient_of_variation == _approx(0.0)
        assert m.count_ratio == _approx(1.0)

    def test_all_counts_zero(self):
        """All target units exist but have zero occurrences."""
        counts = {"a": 0, "b": 0, "c": 0}
        targets = ["a", "b", "c"]
        m = compute_distribution_metrics(counts, targets)
        assert m.entropy == _approx(0.0)
        assert m.zero_count == 3
        assert m.jsd_uniform == _approx(1.0)

    def test_counts_for_non_target_units_ignored(self):
        """Counts for phonemes not in target list are ignored."""
        counts = {"a": 10, "b": 10, "z": 999}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        # Only a and b matter; z is ignored
        assert m.entropy == _approx(math.log2(2))
        assert m.max_count == 10
        assert m.zero_count == 0


# ---------------------------------------------------------------------------
# 6. Pearson edge cases
# ---------------------------------------------------------------------------

class TestPearsonEdgeCases:
    """Pearson correlation has its own degenerate cases."""

    def test_pearson_constant_corpus_is_none(self):
        """All corpus counts equal → zero variance → Pearson undefined.

        When variance is zero, Pearson is mathematically undefined.
        We return None rather than raising an error.
        """
        counts = {"a": 5, "b": 5, "c": 5}
        targets = ["a", "b", "c"]
        reference = {"a": 0.5, "b": 0.3, "c": 0.2}
        m = compute_distribution_metrics(counts, targets, reference)
        assert m.pearson_correlation is None

    def test_pearson_constant_reference_is_none(self):
        """All reference values equal → zero variance → Pearson undefined."""
        counts = {"a": 6, "b": 3, "c": 1}
        targets = ["a", "b", "c"]
        reference = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        m = compute_distribution_metrics(counts, targets, reference)
        assert m.pearson_correlation is None

    def test_pearson_two_units(self):
        """With exactly 2 units, Pearson is ±1 if not constant."""
        counts = {"a": 8, "b": 2}
        targets = ["a", "b"]
        reference = {"a": 0.7, "b": 0.3}
        m = compute_distribution_metrics(counts, targets, reference)
        # Both have a in first, b in second — same direction → r = 1.0
        assert m.pearson_correlation == _approx(1.0)

    def test_pearson_single_unit_is_none(self):
        """Single target unit → no variance → undefined."""
        counts = {"a": 10}
        targets = ["a"]
        reference = {"a": 1.0}
        m = compute_distribution_metrics(counts, targets, reference)
        assert m.pearson_correlation is None


# ---------------------------------------------------------------------------
# 7. PCD metric
# ---------------------------------------------------------------------------

class TestPCDUniform:
    """PCD_uniform = coverage * (1 - JSD_uniform)."""

    def test_full_coverage_uniform_distribution(self):
        """100% coverage, uniform → PCD = 1.0."""
        counts = {"a": 5, "b": 5}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        assert m.pcd_uniform == _approx(1.0)

    def test_partial_coverage_reduces_pcd(self):
        """Missing units lower coverage, which lowers PCD."""
        counts = {"a": 10, "b": 0}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        # coverage = 1/2 = 0.5, JSD > 0 → PCD < 0.5
        assert m.pcd_uniform < 0.5

    def test_pcd_is_zero_for_empty_corpus(self):
        counts: dict[str, int] = {}
        targets = ["a", "b", "c"]
        m = compute_distribution_metrics(counts, targets)
        # coverage = 0, JSD = 1.0 → PCD = 0 * (1-1) = 0
        assert m.pcd_uniform == _approx(0.0)


# ---------------------------------------------------------------------------
# 8. DistributionMetrics dataclass basics
# ---------------------------------------------------------------------------

class TestDistributionMetricsDataclass:
    """The dataclass should be well-behaved."""

    def test_fields_are_accessible(self):
        counts = {"a": 5, "b": 3}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        # All fields should exist and be numeric
        assert isinstance(m.entropy, float)
        assert isinstance(m.normalized_entropy, float)
        assert isinstance(m.jsd_uniform, float)
        assert isinstance(m.coefficient_of_variation, float)
        assert isinstance(m.min_count, int)
        assert isinstance(m.max_count, int)
        assert isinstance(m.count_ratio, float)
        assert isinstance(m.zero_count, int)
        assert isinstance(m.pcd_uniform, float)
        # Without reference, these are None
        assert m.jsd_reference is None
        assert m.pearson_correlation is None

    def test_to_dict(self):
        counts = {"a": 5, "b": 3}
        targets = ["a", "b"]
        m = compute_distribution_metrics(counts, targets)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "entropy" in d
        assert "jsd_uniform" in d
        assert "jsd_reference" in d
        assert "pearson_correlation" in d
        assert d["jsd_reference"] is None
        assert d["pearson_correlation"] is None

    def test_to_dict_with_reference(self):
        counts = {"a": 6, "b": 3, "c": 1}
        targets = ["a", "b", "c"]
        reference = {"a": 0.5, "b": 0.3, "c": 0.2}
        m = compute_distribution_metrics(counts, targets, reference)
        d = m.to_dict()
        assert isinstance(d["jsd_reference"], float)
        # Pearson may be float or None depending on variance
        assert d["pearson_correlation"] is None or isinstance(
            d["pearson_correlation"], float
        )
