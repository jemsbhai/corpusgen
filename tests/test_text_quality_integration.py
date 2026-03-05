"""Tests for text_quality integration into EvaluationReport and evaluate().

Verifies that:
    1. EvaluationReport has an optional `text_quality` field.
    2. evaluate() populates it automatically.
    3. render(), to_dict(), and to_jsonld_ex() include text_quality data.
    4. Existing reports without text_quality still work (backward compat).
"""

from __future__ import annotations

import json

import pytest

from corpusgen.evaluate import evaluate
from corpusgen.evaluate.text_quality import TextQualityMetrics
from corpusgen.evaluate.report import EvaluationReport, SentenceDetail, Verbosity


# ---------------------------------------------------------------------------
# 1. Field existence and backward compatibility
# ---------------------------------------------------------------------------


class TestReportTextQualityField:

    def test_defaults_to_none(self):
        report = EvaluationReport(
            language="en-us",
            unit="phoneme",
            target_phonemes=["p", "b"],
            covered_phonemes={"p"},
            missing_phonemes={"b"},
            coverage=0.5,
            phoneme_counts={"p": 3},
            total_sentences=1,
        )
        assert report.text_quality is None

    def test_accepts_metrics_object(self):
        from corpusgen.evaluate.text_quality import compute_text_quality_metrics

        tq = compute_text_quality_metrics(["The cat."], [["ð", "ə", "k", "æ", "t"]])
        report = EvaluationReport(
            language="en-us",
            unit="phoneme",
            target_phonemes=["p"],
            covered_phonemes={"p"},
            missing_phonemes=set(),
            coverage=1.0,
            phoneme_counts={"p": 1},
            total_sentences=1,
            text_quality=tq,
        )
        assert report.text_quality is tq


# ---------------------------------------------------------------------------
# 2. evaluate() auto-populates
# ---------------------------------------------------------------------------


class TestEvaluatePopulatesTextQuality:

    @pytest.fixture
    def report(self) -> EvaluationReport:
        return evaluate(
            sentences=["The cat sat on the mat.", "Big dogs dig deep holes."],
            language="en-us",
        )

    def test_not_none(self, report):
        assert report.text_quality is not None

    def test_correct_type(self, report):
        assert isinstance(report.text_quality, TextQualityMetrics)

    def test_total_words_positive(self, report):
        assert report.text_quality.total_words > 0

    def test_sentence_count_matches(self, report):
        # 2 sentences → word length stats should be based on 2 data points
        assert report.text_quality.sentence_length_words_min > 0

    def test_ttr_bounded(self, report):
        assert 0.0 < report.text_quality.type_token_ratio <= 1.0

    def test_readability_present_for_english(self, report):
        assert report.text_quality.flesch_reading_ease is not None
        assert report.text_quality.flesch_kincaid_grade is not None


# ---------------------------------------------------------------------------
# 3. Rendering
# ---------------------------------------------------------------------------


class TestRenderIncludesTextQuality:

    @pytest.fixture
    def report(self) -> EvaluationReport:
        return evaluate(
            sentences=["The cat sat on the mat."],
            language="en-us",
        )

    def test_minimal_excludes(self, report):
        text = report.render(Verbosity.MINIMAL)
        assert "text quality" not in text.lower()
        assert "readability" not in text.lower()

    def test_normal_includes(self, report):
        text = report.render(Verbosity.NORMAL)
        assert "text quality" in text.lower() or "ttr" in text.lower() or "type-token" in text.lower()

    def test_verbose_includes(self, report):
        text = report.render(Verbosity.VERBOSE)
        assert "text quality" in text.lower() or "ttr" in text.lower() or "type-token" in text.lower()


# ---------------------------------------------------------------------------
# 4. Export
# ---------------------------------------------------------------------------


class TestExportIncludesTextQuality:

    @pytest.fixture
    def report(self) -> EvaluationReport:
        return evaluate(
            sentences=["The cat sat on the mat."],
            language="en-us",
        )

    def test_to_dict(self, report):
        d = report.to_dict()
        assert "text_quality" in d
        assert isinstance(d["text_quality"], dict)
        assert "type_token_ratio" in d["text_quality"]

    def test_to_dict_none_when_absent(self):
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
        assert d["text_quality"] is None

    def test_to_json_roundtrips(self, report):
        j = report.to_json(indent=2)
        parsed = json.loads(j)
        assert "text_quality" in parsed

    def test_to_jsonld_ex(self, report):
        doc = report.to_jsonld_ex()
        assert "text_quality" in doc
