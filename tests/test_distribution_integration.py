"""Tests for distribution metrics integration into EvaluationReport and evaluate().

Verifies that:
    1. EvaluationReport has an optional `distribution` field.
    2. evaluate() populates it automatically.
    3. render(), to_dict(), and to_jsonld_ex() include distribution data.
    4. Existing reports without distribution still work (backward compat).
"""

from __future__ import annotations

import json
import math

import pytest

from corpusgen.evaluate import evaluate
from corpusgen.evaluate.distribution import DistributionMetrics
from corpusgen.evaluate.report import EvaluationReport, SentenceDetail, Verbosity


# ---------------------------------------------------------------------------
# 1. EvaluationReport field existence and backward compatibility
# ---------------------------------------------------------------------------


class TestReportDistributionField:
    """The distribution field on EvaluationReport."""

    def test_distribution_defaults_to_none(self):
        """Existing code that constructs reports without distribution still works."""
        report = EvaluationReport(
            language="en-us",
            unit="phoneme",
            target_phonemes=["p", "b", "t"],
            covered_phonemes={"p", "b"},
            missing_phonemes={"t"},
            coverage=2 / 3,
            phoneme_counts={"p": 3, "b": 1},
            total_sentences=2,
        )
        assert report.distribution is None

    def test_distribution_accepts_metrics_object(self):
        """Can explicitly pass a DistributionMetrics instance."""
        from corpusgen.evaluate.distribution import compute_distribution_metrics

        counts = {"p": 3, "b": 1}
        targets = ["p", "b", "t"]
        dm = compute_distribution_metrics(counts, targets)

        report = EvaluationReport(
            language="en-us",
            unit="phoneme",
            target_phonemes=targets,
            covered_phonemes={"p", "b"},
            missing_phonemes={"t"},
            coverage=2 / 3,
            phoneme_counts=counts,
            total_sentences=2,
            distribution=dm,
        )
        assert report.distribution is dm
        assert isinstance(report.distribution, DistributionMetrics)


# ---------------------------------------------------------------------------
# 2. evaluate() auto-populates distribution
# ---------------------------------------------------------------------------


class TestEvaluatePopulatesDistribution:
    """evaluate() should compute distribution metrics automatically."""

    @pytest.fixture
    def report(self) -> EvaluationReport:
        return evaluate(
            sentences=["The cat sat on the mat.", "Big dogs dig deep holes."],
            language="en-us",
        )

    def test_distribution_is_not_none(self, report):
        assert report.distribution is not None

    def test_distribution_is_correct_type(self, report):
        assert isinstance(report.distribution, DistributionMetrics)

    def test_distribution_entropy_is_positive(self, report):
        """A real corpus with multiple phonemes should have H > 0."""
        assert report.distribution.entropy > 0.0

    def test_distribution_jsd_uniform_is_nonnegative(self, report):
        assert report.distribution.jsd_uniform >= 0.0

    def test_distribution_jsd_reference_is_none(self, report):
        """evaluate() does not supply a reference → jsd_reference is None."""
        assert report.distribution.jsd_reference is None

    def test_distribution_pearson_is_none(self, report):
        """evaluate() does not supply a reference → pearson is None."""
        assert report.distribution.pearson_correlation is None

    def test_distribution_zero_count_matches_missing(self, report):
        """zero_count should equal the number of missing target units."""
        assert report.distribution.zero_count == len(report.missing_phonemes)

    def test_distribution_pcd_uniform_bounded(self, report):
        assert 0.0 <= report.distribution.pcd_uniform <= 1.0


# ---------------------------------------------------------------------------
# 3. evaluate() with diphone and triphone units
# ---------------------------------------------------------------------------


class TestEvaluateDistributionUnits:
    """Distribution metrics work for diphone and triphone units too."""

    def test_diphone_has_distribution(self):
        report = evaluate(
            sentences=["The cat sat."],
            language="en-us",
            unit="diphone",
        )
        assert report.distribution is not None
        assert report.distribution.entropy >= 0.0

    def test_triphone_has_distribution(self):
        report = evaluate(
            sentences=["The cat sat."],
            language="en-us",
            unit="triphone",
        )
        assert report.distribution is not None
        assert report.distribution.entropy >= 0.0


# ---------------------------------------------------------------------------
# 4. Rendering includes distribution
# ---------------------------------------------------------------------------


class TestRenderIncludesDistribution:
    """render() should include distribution info at NORMAL and VERBOSE."""

    @pytest.fixture
    def report(self) -> EvaluationReport:
        return evaluate(
            sentences=["The cat sat on the mat."],
            language="en-us",
        )

    def test_minimal_render_excludes_distribution(self, report):
        """MINIMAL verbosity should not include distribution details."""
        text = report.render(Verbosity.MINIMAL)
        assert "entropy" not in text.lower()

    def test_normal_render_includes_distribution(self, report):
        """NORMAL verbosity should include key distribution metrics."""
        text = report.render(Verbosity.NORMAL)
        assert "entropy" in text.lower() or "distribution" in text.lower()

    def test_verbose_render_includes_distribution(self, report):
        """VERBOSE verbosity should include distribution metrics."""
        text = report.render(Verbosity.VERBOSE)
        assert "entropy" in text.lower() or "distribution" in text.lower()


# ---------------------------------------------------------------------------
# 5. Export includes distribution
# ---------------------------------------------------------------------------


class TestExportIncludesDistribution:
    """to_dict() and to_jsonld_ex() should include distribution data."""

    @pytest.fixture
    def report(self) -> EvaluationReport:
        return evaluate(
            sentences=["The cat sat on the mat."],
            language="en-us",
        )

    def test_to_dict_has_distribution(self, report):
        d = report.to_dict()
        assert "distribution" in d
        assert isinstance(d["distribution"], dict)
        assert "entropy" in d["distribution"]
        assert "jsd_uniform" in d["distribution"]

    def test_to_dict_distribution_none_when_absent(self):
        """Legacy reports without distribution export None."""
        report = EvaluationReport(
            language="en-us",
            unit="phoneme",
            target_phonemes=["p"],
            covered_phonemes={"p"},
            missing_phonemes=set(),
            coverage=1.0,
            phoneme_counts={"p": 5},
            total_sentences=1,
        )
        d = report.to_dict()
        assert d["distribution"] is None

    def test_to_json_roundtrips(self, report):
        """JSON export should be valid JSON containing distribution."""
        j = report.to_json(indent=2)
        parsed = json.loads(j)
        assert "distribution" in parsed
        assert isinstance(parsed["distribution"], dict)

    def test_to_jsonld_ex_has_distribution(self, report):
        doc = report.to_jsonld_ex()
        assert "distribution" in doc
        assert isinstance(doc["distribution"], dict)
