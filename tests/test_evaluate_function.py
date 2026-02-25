"""Tests for the top-level evaluate() function.

TDD RED phase — these tests define the contract for corpusgen.evaluate.evaluate().
The function wires G2PManager + CoverageTracker + EvaluationReport into a single
user-facing API:

    evaluate(sentences, language, target_phonemes, unit) -> EvaluationReport
"""

import pytest

from corpusgen.evaluate import evaluate
from corpusgen.evaluate.report import EvaluationReport, SentenceDetail, Verbosity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def english_sentences() -> list[str]:
    """Small English corpus with known phonemic variety."""
    return [
        "The cat sat on the mat.",
        "She sells seashells by the seashore.",
        "Big dogs dig deep holes.",
    ]


@pytest.fixture
def narrow_target() -> list[str]:
    """A small target inventory that any English text will partially cover."""
    return ["p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ"]


@pytest.fixture
def tiny_corpus() -> list[str]:
    """Single short sentence for minimal tests."""
    return ["The big dog."]


# ---------------------------------------------------------------------------
# 1. Return type and basic field correctness
# ---------------------------------------------------------------------------


class TestEvaluateReturnType:
    """evaluate() must return a well-formed EvaluationReport."""

    def test_returns_evaluation_report(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert isinstance(report, EvaluationReport)

    def test_language_field_matches_input(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert report.language == "en-us"

    def test_unit_field_defaults_to_phoneme(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert report.unit == "phoneme"

    def test_unit_field_matches_input(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target, unit="diphone")
        assert report.unit == "diphone"

    def test_total_sentences_matches_input(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert report.total_sentences == len(english_sentences)

    def test_target_phonemes_matches_input(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert report.target_phonemes == narrow_target


# ---------------------------------------------------------------------------
# 2. Coverage invariants (must hold for any valid input)
# ---------------------------------------------------------------------------


class TestCoverageInvariants:
    """Structural invariants that must hold regardless of espeak output."""

    def test_coverage_between_0_and_1(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert 0.0 <= report.coverage <= 1.0

    def test_covered_plus_missing_equals_target(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert report.covered_phonemes | report.missing_phonemes == set(narrow_target)

    def test_covered_and_missing_are_disjoint(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert report.covered_phonemes & report.missing_phonemes == set()

    def test_coverage_ratio_is_correct(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        expected = len(report.covered_phonemes) / len(narrow_target)
        assert report.coverage == pytest.approx(expected)

    def test_covered_is_subset_of_target(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert report.covered_phonemes <= set(narrow_target)

    def test_missing_is_subset_of_target(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert report.missing_phonemes <= set(narrow_target)


# ---------------------------------------------------------------------------
# 3. target_phonemes=None (derive inventory from corpus)
# ---------------------------------------------------------------------------


class TestEvaluateDerivedInventory:
    """When target_phonemes is None, evaluate() should derive the inventory
    from the phonemes actually found in the corpus."""

    def test_none_target_gives_full_coverage(self, english_sentences):
        report = evaluate(english_sentences, language="en-us", target_phonemes=None)
        assert report.coverage == pytest.approx(1.0)

    def test_none_target_missing_is_empty(self, english_sentences):
        report = evaluate(english_sentences, language="en-us", target_phonemes=None)
        assert report.missing_phonemes == set()

    def test_none_target_has_nonempty_target_list(self, english_sentences):
        report = evaluate(english_sentences, language="en-us", target_phonemes=None)
        assert len(report.target_phonemes) > 0

    def test_none_target_covered_equals_target(self, english_sentences):
        report = evaluate(english_sentences, language="en-us", target_phonemes=None)
        assert report.covered_phonemes == set(report.target_phonemes)


# ---------------------------------------------------------------------------
# 4. Sentence details
# ---------------------------------------------------------------------------


class TestSentenceDetails:
    """Per-sentence breakdown must be accurate and complete."""

    def test_details_length_matches_sentences(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert len(report.sentence_details) == len(english_sentences)

    def test_details_indices_are_sequential(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        indices = [sd.index for sd in report.sentence_details]
        assert indices == list(range(len(english_sentences)))

    def test_details_texts_match_input(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        texts = [sd.text for sd in report.sentence_details]
        assert texts == english_sentences

    def test_details_are_sentence_detail_instances(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        for sd in report.sentence_details:
            assert isinstance(sd, SentenceDetail)

    def test_details_phoneme_count_positive_for_nonempty(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        for sd in report.sentence_details:
            assert sd.phoneme_count > 0

    def test_details_all_phonemes_nonempty(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        for sd in report.sentence_details:
            assert len(sd.all_phonemes) > 0

    def test_details_phoneme_count_equals_all_phonemes_length(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        for sd in report.sentence_details:
            assert sd.phoneme_count == len(sd.all_phonemes)

    def test_new_phonemes_are_subset_of_all_phonemes(self, english_sentences, narrow_target):
        """Each sentence's new_phonemes must be a subset of its all_phonemes."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        for sd in report.sentence_details:
            assert set(sd.new_phonemes) <= set(sd.all_phonemes)

    def test_new_phonemes_are_subset_of_target(self, english_sentences, narrow_target):
        """new_phonemes only includes phonemes from the target inventory."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        target_set = set(narrow_target)
        for sd in report.sentence_details:
            assert set(sd.new_phonemes) <= target_set

    def test_new_phonemes_no_duplicates_across_sentences(self, english_sentences, narrow_target):
        """A phoneme should appear as 'new' in at most one sentence."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        all_new = []
        for sd in report.sentence_details:
            all_new.extend(sd.new_phonemes)
        assert len(all_new) == len(set(all_new))

    def test_union_of_new_phonemes_equals_covered(self, english_sentences, narrow_target):
        """The union of all new_phonemes across sentences equals covered_phonemes."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        all_new = set()
        for sd in report.sentence_details:
            all_new.update(sd.new_phonemes)
        assert all_new == report.covered_phonemes


# ---------------------------------------------------------------------------
# 5. Phoneme counts and sources
# ---------------------------------------------------------------------------


class TestPhonemeCounts:
    """Phoneme frequency counts and source tracking."""

    def test_counts_keys_equal_covered(self, english_sentences, narrow_target):
        """Every covered phoneme has a count entry, and vice versa."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert set(report.phoneme_counts.keys()) == report.covered_phonemes

    def test_counts_all_positive(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        for phoneme, count in report.phoneme_counts.items():
            assert count > 0, f"phoneme {phoneme!r} has count {count}"

    def test_sources_keys_equal_covered(self, english_sentences, narrow_target):
        """Every covered phoneme has a source entry."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert set(report.phoneme_sources.keys()) == report.covered_phonemes

    def test_sources_indices_are_valid(self, english_sentences, narrow_target):
        """All source indices are within [0, total_sentences)."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        for phoneme, indices in report.phoneme_sources.items():
            for idx in indices:
                assert 0 <= idx < report.total_sentences, (
                    f"phoneme {phoneme!r} has invalid source index {idx}"
                )

    def test_sources_indices_are_sorted(self, english_sentences, narrow_target):
        """Source indices for each phoneme should be in ascending order."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        for phoneme, indices in report.phoneme_sources.items():
            assert indices == sorted(indices), (
                f"phoneme {phoneme!r} sources not sorted: {indices}"
            )

    def test_sources_no_duplicate_indices(self, english_sentences, narrow_target):
        """No duplicate sentence indices per phoneme."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        for phoneme, indices in report.phoneme_sources.items():
            assert len(indices) == len(set(indices)), (
                f"phoneme {phoneme!r} has duplicate source indices: {indices}"
            )


# ---------------------------------------------------------------------------
# 6. Different coverage units (diphone, triphone)
# ---------------------------------------------------------------------------


class TestEvaluateUnits:
    """evaluate() with diphone and triphone units."""

    def test_diphone_unit_accepted(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target, unit="diphone")
        assert report.unit == "diphone"
        assert isinstance(report.coverage, float)

    def test_triphone_unit_accepted(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target, unit="triphone")
        assert report.unit == "triphone"
        assert isinstance(report.coverage, float)

    def test_invalid_unit_raises(self, english_sentences, narrow_target):
        with pytest.raises(ValueError, match="unit"):
            evaluate(english_sentences, language="en-us", target_phonemes=narrow_target, unit="quadphone")

    def test_diphone_covered_missing_invariant(self, english_sentences, narrow_target):
        """covered + missing = target set, even for diphone unit."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target, unit="diphone")
        assert report.covered_phonemes | report.missing_phonemes == set(report.target_phonemes)
        assert report.covered_phonemes & report.missing_phonemes == set()

    def test_diphone_coverage_ratio(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target, unit="diphone")
        if len(report.target_phonemes) > 0:
            expected = len(report.covered_phonemes) / len(report.target_phonemes)
            assert report.coverage == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEvaluateEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_empty_sentence_list(self):
        """Empty corpus should produce a valid report with zero coverage."""
        report = evaluate([], language="en-us", target_phonemes=["p", "b", "t"])
        assert report.total_sentences == 0
        assert report.coverage == pytest.approx(0.0)
        assert report.covered_phonemes == set()
        assert report.missing_phonemes == {"p", "b", "t"}
        assert report.sentence_details == []
        assert report.phoneme_counts == {}
        assert report.phoneme_sources == {}

    def test_empty_sentence_list_no_target(self):
        """Empty corpus with no target should still return valid report."""
        report = evaluate([], language="en-us", target_phonemes=None)
        assert report.total_sentences == 0
        assert report.sentence_details == []
        assert report.coverage == pytest.approx(1.0)  # 0/0 → 1.0 by convention

    def test_single_sentence(self, narrow_target):
        report = evaluate(["The cat sat."], language="en-us", target_phonemes=narrow_target)
        assert report.total_sentences == 1
        assert len(report.sentence_details) == 1
        assert report.sentence_details[0].index == 0
        assert report.sentence_details[0].text == "The cat sat."

    def test_whitespace_only_sentence(self, narrow_target):
        """A whitespace-only sentence should not crash; it contributes no phonemes."""
        report = evaluate(["   ", "The cat."], language="en-us", target_phonemes=narrow_target)
        assert report.total_sentences == 2
        assert len(report.sentence_details) == 2
        # Whitespace sentence has 0 phonemes
        assert report.sentence_details[0].phoneme_count == 0
        assert report.sentence_details[0].all_phonemes == []
        assert report.sentence_details[0].new_phonemes == []

    def test_empty_string_sentence(self, narrow_target):
        """An empty string sentence should not crash."""
        report = evaluate(["", "Hello."], language="en-us", target_phonemes=narrow_target)
        assert report.total_sentences == 2
        assert report.sentence_details[0].phoneme_count == 0

    def test_target_with_phonemes_not_in_corpus(self):
        """Target phonemes that don't appear in the corpus stay in missing."""
        # ʒ (voiced postalveolar fricative) is rare in English
        report = evaluate(
            ["The cat sat."],
            language="en-us",
            target_phonemes=["ʒ", "ʁ", "ɣ"],
        )
        assert report.coverage == pytest.approx(0.0)
        assert report.missing_phonemes == {"ʒ", "ʁ", "ɣ"}

    def test_empty_target_phonemes_list(self):
        """An empty target list: coverage should be 1.0 (vacuous truth)."""
        report = evaluate(["The cat."], language="en-us", target_phonemes=[])
        assert report.coverage == pytest.approx(1.0)
        assert report.covered_phonemes == set()
        assert report.missing_phonemes == set()


# ---------------------------------------------------------------------------
# 8. Non-English language
# ---------------------------------------------------------------------------


class TestEvaluateMultilingual:
    """evaluate() works with non-English languages via espeak-ng."""

    def test_french(self):
        report = evaluate(
            ["Bonjour le monde."],
            language="fr-fr",
            target_phonemes=None,
        )
        assert report.language == "fr-fr"
        assert report.total_sentences == 1
        assert report.coverage == pytest.approx(1.0)
        assert len(report.target_phonemes) > 0

    def test_german(self):
        report = evaluate(
            ["Guten Morgen."],
            language="de",
            target_phonemes=None,
        )
        assert report.language == "de"
        assert report.coverage == pytest.approx(1.0)

    def test_arabic(self):
        report = evaluate(
            ["مرحبا بالعالم"],
            language="ar",
            target_phonemes=None,
        )
        assert report.language == "ar"
        assert report.coverage == pytest.approx(1.0)
        assert len(report.target_phonemes) > 0


# ---------------------------------------------------------------------------
# 9. Report rendering and export integration
# ---------------------------------------------------------------------------


class TestEvaluateReportUsability:
    """The returned report should be immediately usable for rendering/export."""

    def test_render_minimal(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        text = report.render(verbosity=Verbosity.MINIMAL)
        assert "%" in text
        assert "Missing" in text or "missing" in text

    def test_render_verbose(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        text = report.render(verbosity=Verbosity.VERBOSE)
        # Should contain all input sentences
        for sentence in english_sentences:
            assert sentence in text

    def test_to_dict_roundtrip(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        d = report.to_dict()
        assert d["language"] == "en-us"
        assert d["total_sentences"] == len(english_sentences)
        assert len(d["sentence_details"]) == len(english_sentences)

    def test_to_json_is_valid(self, english_sentences, narrow_target):
        import json
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["language"] == "en-us"

    def test_to_jsonld_ex_has_context(self, english_sentences, narrow_target):
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        doc = report.to_jsonld_ex()
        assert "@context" in doc
        assert "@type" in doc


# ---------------------------------------------------------------------------
# 10. Consistency between components
# ---------------------------------------------------------------------------


class TestCrossComponentConsistency:
    """Verify that the wiring between G2P, CoverageTracker, and Report is correct."""

    def test_phoneme_counts_sum_matches_total_occurrences(self, english_sentences, narrow_target):
        """Sum of phoneme_counts should equal the total number of target-phoneme
        occurrences across all sentences."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        target_set = set(narrow_target)

        total_from_counts = sum(report.phoneme_counts.values())

        # Count from sentence details
        total_from_details = 0
        for sd in report.sentence_details:
            for ph in sd.all_phonemes:
                if ph in target_set:
                    total_from_details += 1

        assert total_from_counts == total_from_details

    def test_sources_consistent_with_sentence_details(self, english_sentences, narrow_target):
        """If phoneme P has source sentence S, then P must appear in S's all_phonemes."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        for phoneme, indices in report.phoneme_sources.items():
            for idx in indices:
                sd = report.sentence_details[idx]
                assert phoneme in sd.all_phonemes, (
                    f"phoneme {phoneme!r} claims source sentence {idx}, "
                    f"but not in that sentence's all_phonemes: {sd.all_phonemes}"
                )

    def test_covered_count_matches_len_covered_phonemes(self, english_sentences, narrow_target):
        """Report coverage ratio should be derivable from covered/target counts."""
        report = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert len(report.covered_phonemes) + len(report.missing_phonemes) == len(narrow_target)

    def test_deterministic_across_calls(self, english_sentences, narrow_target):
        """Same input should produce identical output."""
        r1 = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        r2 = evaluate(english_sentences, language="en-us", target_phonemes=narrow_target)
        assert r1.coverage == r2.coverage
        assert r1.covered_phonemes == r2.covered_phonemes
        assert r1.missing_phonemes == r2.missing_phonemes
        assert r1.phoneme_counts == r2.phoneme_counts
        for sd1, sd2 in zip(r1.sentence_details, r2.sentence_details):
            assert sd1.all_phonemes == sd2.all_phonemes
            assert sd1.new_phonemes == sd2.new_phonemes
