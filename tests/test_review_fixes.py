"""Regression tests for code review fixes A–G.

Each test targets a specific bug or design issue identified in review.
"""

from __future__ import annotations

import pytest


# ── Fix A: DistributionAwareSelector tie-break uses weighted gain ──


class TestFixA_DistributionTieBreak:
    """The tie-break comparison must use wgain (weighted), not gain (unweighted).

    We construct a scenario where two candidates have equal KL divergence
    but different weighted gains, so the tie-break must pick the one with
    higher *weighted* new coverage.
    """

    def test_tiebreak_prefers_higher_weighted_gain(self):
        from corpusgen.select.distribution import DistributionAwareSelector

        # Two candidates that each cover one new unit after the first round.
        # "rare" has weight 10.0, "common" has weight 1.0.
        # Both add exactly 1 new unit (unweighted gain = 1 for both),
        # but weighted gain differs: 10.0 vs 1.0.
        # With a flat target distribution they'll have identical KL,
        # so the tie-break on weighted gain must pick "rare".
        candidates = ["sent_common", "sent_rare"]
        candidate_phonemes = [["common"], ["rare"]]
        target_units = {"common", "rare"}
        weights = {"common": 1.0, "rare": 10.0}

        # Flat target distribution → identical KL for both candidates
        target_dist = {"common": 0.5, "rare": 0.5}
        selector = DistributionAwareSelector(
            unit="phoneme", target_distribution=target_dist
        )
        result = selector.select(
            candidates=candidates,
            candidate_phonemes=candidate_phonemes,
            target_units=target_units,
            max_sentences=1,
            weights=weights,
        )

        # The selector should prefer the sentence covering "rare"
        assert result.selected_sentences == ["sent_rare"]


# ── Fix B: ILP ceiling for target_coverage ──


class TestFixB_ILPCoverage:
    """int(0.95 * 10) = 9 (wrong), ceil(0.95 * 10) = 10 (correct)."""

    def test_ilp_requires_ceil_coverage(self):
        pytest.importorskip("pulp")
        from corpusgen.select.ilp import ILPSelector

        # 10 target units, each covered by exactly one dedicated sentence,
        # plus one extra unit "bonus" that nothing covers.
        phonemes = list("abcdefghij")  # 10 units
        target_units = set(phonemes)
        candidates = [f"sent_{p}" for p in phonemes]
        candidate_phonemes = [[p] for p in phonemes]

        selector = ILPSelector(unit="phoneme")
        result = selector.select(
            candidates=candidates,
            candidate_phonemes=candidate_phonemes,
            target_units=target_units,
            target_coverage=0.95,
        )

        # ceil(0.95 * 10) = 10, so ILP must select all 10 sentences
        assert len(result.selected_sentences) == 10
        assert result.coverage == pytest.approx(1.0)

    def test_ilp_coverage_boundary_value(self):
        """Verify ceiling at a boundary: 0.51 * 2 = 1.02 → ceil = 2."""
        pytest.importorskip("pulp")
        from corpusgen.select.ilp import ILPSelector

        target_units = {"a", "b"}
        candidates = ["s1", "s2"]
        candidate_phonemes = [["a"], ["b"]]

        selector = ILPSelector(unit="phoneme")
        result = selector.select(
            candidates=candidates,
            candidate_phonemes=candidate_phonemes,
            target_units=target_units,
            target_coverage=0.51,
        )

        # ceil(0.51 * 2) = ceil(1.02) = 2 → must cover both
        assert len(result.selected_sentences) == 2


# ── Fix C: dynamic __all__ with missing optional deps ──


class TestFixC_DynamicAll:
    """__all__ must not list ILPSelector/NSGA2Selector if they failed to import."""

    def test_all_contains_core_selectors(self):
        from corpusgen import select

        for name in [
            "SelectorBase",
            "SelectionResult",
            "GreedySelector",
            "CELFSelector",
            "StochasticGreedySelector",
            "DistributionAwareSelector",
            "select_sentences",
        ]:
            assert name in select.__all__

    def test_ilp_in_all_only_if_available(self):
        from corpusgen import select

        if select.ILPSelector is not None:
            assert "ILPSelector" in select.__all__
        else:
            assert "ILPSelector" not in select.__all__

    def test_nsga2_in_all_only_if_available(self):
        from corpusgen import select

        if select.NSGA2Selector is not None:
            assert "NSGA2Selector" in select.__all__
        else:
            assert "NSGA2Selector" not in select.__all__


# ── Fix D: public target_units and covered_units properties ──


class TestFixD_PublicProperties:
    """CoverageTracker must expose target_units and covered_units publicly."""

    def test_target_units_phoneme(self):
        from corpusgen.coverage.tracker import CoverageTracker

        tracker = CoverageTracker(target_phonemes=["a", "b", "c"], unit="phoneme")
        assert tracker.target_units == {"a", "b", "c"}

    def test_target_units_diphone(self):
        from corpusgen.coverage.tracker import CoverageTracker

        tracker = CoverageTracker(target_phonemes=["a", "b"], unit="diphone")
        assert tracker.target_units == {"a-a", "a-b", "b-a", "b-b"}

    def test_target_units_returns_copy(self):
        """Mutating the returned set must not affect internal state."""
        from corpusgen.coverage.tracker import CoverageTracker

        tracker = CoverageTracker(target_phonemes=["a", "b"], unit="phoneme")
        units = tracker.target_units
        units.add("z")
        assert "z" not in tracker.target_units

    def test_covered_units_empty_initially(self):
        from corpusgen.coverage.tracker import CoverageTracker

        tracker = CoverageTracker(target_phonemes=["a", "b"], unit="phoneme")
        assert tracker.covered_units == set()

    def test_covered_units_after_update(self):
        from corpusgen.coverage.tracker import CoverageTracker

        tracker = CoverageTracker(target_phonemes=["a", "b", "c"], unit="phoneme")
        tracker.update(["a", "b"], sentence_index=0)
        assert tracker.covered_units == {"a", "b"}

    def test_covered_units_returns_copy(self):
        from corpusgen.coverage.tracker import CoverageTracker

        tracker = CoverageTracker(target_phonemes=["a"], unit="phoneme")
        tracker.update(["a"], sentence_index=0)
        covered = tracker.covered_units
        covered.discard("a")
        assert "a" in tracker.covered_units

    def test_no_private_access_in_evaluate(self):
        """Verify evaluate.py doesn't use _target_set or _covered."""
        import inspect
        from corpusgen.evaluate import evaluate as eval_mod

        source = inspect.getsource(eval_mod)
        assert "_target_set" not in source, "evaluate.py still uses _target_set"
        assert "._covered" not in source, "evaluate.py still uses _covered"

    def test_no_private_access_in_select_init(self):
        """Verify select/__init__.py doesn't use _target_set."""
        import importlib
        import pathlib

        mod = importlib.import_module("corpusgen.select")
        source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        assert "tracker._target_set" not in source


# ── Fix E: safety guard for cartesian explosion ──


class TestFixE_TargetSizeGuard:
    """Large phoneme inventories must be rejected for diphone/triphone."""

    def test_diphone_within_limit(self):
        """10 phonemes → 100 diphones — should work."""
        from corpusgen.coverage.tracker import CoverageTracker

        phonemes = [f"p{i}" for i in range(10)]
        tracker = CoverageTracker(target_phonemes=phonemes, unit="diphone")
        assert tracker.target_size == 100

    def test_diphone_exceeds_default_limit(self):
        """800 phonemes → 640,000 diphones > 500,000 default limit."""
        from corpusgen.coverage.tracker import CoverageTracker

        phonemes = [f"p{i}" for i in range(800)]
        with pytest.raises(ValueError, match="exceeds max_target_size"):
            CoverageTracker(target_phonemes=phonemes, unit="diphone")

    def test_triphone_exceeds_default_limit(self):
        """80 phonemes → 512,000 triphones > 500,000."""
        from corpusgen.coverage.tracker import CoverageTracker

        phonemes = [f"p{i}" for i in range(80)]
        with pytest.raises(ValueError, match="exceeds max_target_size"):
            CoverageTracker(target_phonemes=phonemes, unit="triphone")

    def test_custom_limit_allows_larger(self):
        """Custom max_target_size can raise the ceiling."""
        from corpusgen.coverage.tracker import CoverageTracker

        phonemes = [f"p{i}" for i in range(800)]
        tracker = CoverageTracker(
            target_phonemes=phonemes,
            unit="diphone",
            max_target_size=1_000_000,
        )
        assert tracker.target_size == 640_000

    def test_custom_limit_can_be_stricter(self):
        """Custom max_target_size can also lower the ceiling."""
        from corpusgen.coverage.tracker import CoverageTracker

        phonemes = [f"p{i}" for i in range(20)]
        # 20^2 = 400 diphones, but limit set to 100
        with pytest.raises(ValueError, match="exceeds max_target_size"):
            CoverageTracker(
                target_phonemes=phonemes, unit="diphone", max_target_size=100
            )

    def test_phoneme_unit_ignores_limit(self):
        """Phoneme unit never triggers the guard (no cartesian product)."""
        from corpusgen.coverage.tracker import CoverageTracker

        phonemes = [f"p{i}" for i in range(1000)]
        tracker = CoverageTracker(target_phonemes=phonemes, unit="phoneme")
        assert tracker.target_size == 1000


# ── Fix F: render() truncation for large missing sets ──


class TestFixF_RenderTruncation:
    """Missing units display must be truncated when > MAX_MISSING_DISPLAY."""

    def test_small_missing_not_truncated(self):
        from corpusgen.evaluate.report import EvaluationReport, Verbosity

        report = EvaluationReport(
            language="en",
            unit="phoneme",
            target_phonemes=["a", "b", "c"],
            covered_phonemes={"a"},
            missing_phonemes={"b", "c"},
            coverage=1 / 3,
            phoneme_counts={"a": 1},
            total_sentences=1,
        )
        rendered = report.render(Verbosity.MINIMAL)
        assert "b, c" in rendered
        assert "more)" not in rendered

    def test_large_missing_is_truncated(self):
        from corpusgen.evaluate.report import EvaluationReport, Verbosity

        # 200 missing units → should show first 50 + "(+150 more)"
        all_units = [f"p{i:03d}" for i in range(200)]
        report = EvaluationReport(
            language="en",
            unit="diphone",
            target_phonemes=all_units,
            covered_phonemes=set(),
            missing_phonemes=set(all_units),
            coverage=0.0,
            phoneme_counts={},
            total_sentences=0,
        )
        rendered = report.render(Verbosity.MINIMAL)
        assert "(+150 more)" in rendered
        # Should contain exactly 50 comma-separated items before the truncation
        missing_line = [
            line for line in rendered.splitlines() if line.startswith("Missing:")
        ][0]
        # The truncated part contains 50 items + "(+150 more)"
        assert "p000" in missing_line  # first sorted item present
        assert "p199" not in missing_line  # last sorted item truncated

    def test_exactly_at_limit_not_truncated(self):
        from corpusgen.evaluate.report import EvaluationReport, Verbosity

        units = [f"p{i:02d}" for i in range(50)]
        report = EvaluationReport(
            language="en",
            unit="phoneme",
            target_phonemes=units,
            covered_phonemes=set(),
            missing_phonemes=set(units),
            coverage=0.0,
            phoneme_counts={},
            total_sentences=0,
        )
        rendered = report.render(Verbosity.MINIMAL)
        assert "more)" not in rendered


# ── Fix G: frequency_inverse_weights unit-aware counting ──


class TestFixG_UnitAwareWeights:
    """frequency_inverse_weights must count the correct unit type."""

    def test_phoneme_unit_backward_compatible(self):
        """Default unit='phoneme' counts raw phonemes as before."""
        from corpusgen.weights import frequency_inverse_weights

        target = {"a", "b"}
        corpus = [["a", "a", "b"], ["a", "b", "b"]]
        weights = frequency_inverse_weights(target, corpus, unit="phoneme")
        # "a" appears 3 times, "b" appears 3 times → equal weights
        assert weights["a"] == pytest.approx(weights["b"])

    def test_diphone_unit_counts_diphones(self):
        """With unit='diphone', must count diphone occurrences, not phonemes."""
        from corpusgen.weights import frequency_inverse_weights

        # Corpus: ["a", "b", "a"] → diphones "a-b", "b-a"
        # Corpus: ["a", "b", "a"] → diphones "a-b", "b-a" (same)
        # So "a-b" count=2, "b-a" count=2, "a-a" count=0
        target = {"a-b", "b-a", "a-a"}
        corpus = [["a", "b", "a"], ["a", "b", "a"]]

        weights = frequency_inverse_weights(target, corpus, unit="diphone")

        # "a-a" never appears → should have higher weight than "a-b" or "b-a"
        assert weights["a-a"] > weights["a-b"]
        assert weights["a-a"] > weights["b-a"]
        # "a-b" and "b-a" appear equally → same weight
        assert weights["a-b"] == pytest.approx(weights["b-a"])

    def test_triphone_unit_counts_triphones(self):
        """With unit='triphone', must count triphone occurrences."""
        from corpusgen.weights import frequency_inverse_weights

        # ["a","b","c","a"] → triphones "a-b-c", "b-c-a"
        target = {"a-b-c", "b-c-a", "c-a-b"}
        corpus = [["a", "b", "c", "a"]]

        weights = frequency_inverse_weights(target, corpus, unit="triphone")

        # "c-a-b" never appears → highest weight
        assert weights["c-a-b"] > weights["a-b-c"]
        assert weights["c-a-b"] > weights["b-c-a"]

    def test_diphone_weights_differ_from_phoneme_weights(self):
        """Confirm diphone counting gives different results than phoneme counting."""
        from corpusgen.weights import frequency_inverse_weights

        # Two diphone targets with different actual diphone frequencies.
        # ["a","b","a"] → diphones "a-b", "b-a"
        # So "a-b" appears once, "b-a" appears once, "a-a" appears zero times.
        target = {"a-b", "b-a", "a-a"}
        corpus = [["a", "b", "a"]]

        w_diphone = frequency_inverse_weights(target, corpus, unit="diphone")
        w_phoneme = frequency_inverse_weights(target, corpus, unit="phoneme")

        # With correct diphone counting: "a-a"=0, "a-b"=1, "b-a"=1
        # → "a-a" gets highest weight
        assert w_diphone["a-a"] > w_diphone["a-b"]

        # With wrong phoneme counting: none of "a-b","b-a","a-a" match
        # raw phonemes "a" or "b", so all get zero freq → equal weights
        assert w_phoneme["a-a"] == pytest.approx(w_phoneme["a-b"])
        assert w_phoneme["a-a"] == pytest.approx(w_phoneme["b-a"])
