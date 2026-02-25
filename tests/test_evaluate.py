"""Tests for the evaluation module: reports, verbosity levels, and export."""

import json

import pytest

from corpusgen.evaluate.report import EvaluationReport, Verbosity, SentenceDetail


# --- EvaluationReport construction ---


class TestEvaluationReport:
    """Tests for the EvaluationReport data container."""

    @pytest.fixture
    def minimal_report(self) -> EvaluationReport:
        """A report with basic coverage data."""
        return EvaluationReport(
            language="en-us",
            unit="phoneme",
            target_phonemes=["p", "b", "t", "d", "k", "s", "z", "ʃ"],
            covered_phonemes={"p", "b", "t", "d", "k"},
            missing_phonemes={"s", "z", "ʃ"},
            coverage=5 / 8,
            phoneme_counts={"p": 4, "b": 2, "t": 7, "d": 1, "k": 3},
            total_sentences=3,
            sentence_details=[
                SentenceDetail(
                    index=0,
                    text="The pink bat danced.",
                    phoneme_count=12,
                    new_phonemes=["p", "b", "t", "d"],
                    all_phonemes=["ð", "ə", "p", "ɪ", "ŋ", "k", "b", "æ", "t", "d", "æ", "n", "s", "t"],
                ),
                SentenceDetail(
                    index=1,
                    text="The dark pit cracked.",
                    phoneme_count=10,
                    new_phonemes=["k"],
                    all_phonemes=["ð", "ə", "d", "ɑː", "k", "p", "ɪ", "t", "k", "ɹ"],
                ),
                SentenceDetail(
                    index=2,
                    text="A pet bat.",
                    phoneme_count=6,
                    new_phonemes=[],
                    all_phonemes=["ə", "p", "ɛ", "t", "b", "æ", "t"],
                ),
            ],
            phoneme_sources={
                "p": [0, 1, 2],
                "b": [0, 2],
                "t": [0, 1, 2],
                "d": [0, 1],
                "k": [1],
            },
        )

    # --- Basic properties ---

    def test_coverage_value(self, minimal_report):
        assert minimal_report.coverage == pytest.approx(0.625)

    def test_missing_phonemes(self, minimal_report):
        assert minimal_report.missing_phonemes == {"s", "z", "ʃ"}

    def test_covered_phonemes(self, minimal_report):
        assert minimal_report.covered_phonemes == {"p", "b", "t", "d", "k"}

    def test_total_sentences(self, minimal_report):
        assert minimal_report.total_sentences == 3


# --- Verbosity levels ---


class TestReportVerbosity:
    """Tests for the .report() method at different verbosity levels."""

    @pytest.fixture
    def report(self) -> EvaluationReport:
        return EvaluationReport(
            language="en-us",
            unit="phoneme",
            target_phonemes=["p", "b", "t", "d", "k", "s"],
            covered_phonemes={"p", "b", "t"},
            missing_phonemes={"d", "k", "s"},
            coverage=0.5,
            phoneme_counts={"p": 3, "b": 1, "t": 5},
            total_sentences=2,
            sentence_details=[
                SentenceDetail(
                    index=0,
                    text="A big tap.",
                    phoneme_count=6,
                    new_phonemes=["p", "b", "t"],
                    all_phonemes=["ə", "b", "ɪ", "ɡ", "t", "æ", "p"],
                ),
                SentenceDetail(
                    index=1,
                    text="The top bit.",
                    phoneme_count=7,
                    new_phonemes=[],
                    all_phonemes=["ð", "ə", "t", "ɒ", "p", "b", "ɪ", "t"],
                ),
            ],
            phoneme_sources={"p": [0, 1], "b": [0, 1], "t": [0, 1]},
        )

    def test_minimal_report_contains_coverage(self, report):
        """Minimal output includes coverage percentage and missing list."""
        text = report.render(verbosity=Verbosity.MINIMAL)
        assert "50.0%" in text
        assert "3/6" in text

    def test_minimal_report_shows_missing(self, report):
        """Minimal output lists missing phonemes."""
        text = report.render(verbosity=Verbosity.MINIMAL)
        # All three missing phonemes should appear
        for phoneme in ["d", "k", "s"]:
            assert phoneme in text

    def test_normal_report_includes_counts(self, report):
        """Normal output includes per-phoneme counts."""
        text = report.render(verbosity=Verbosity.NORMAL)
        assert "p" in text
        assert "50.0%" in text

    def test_verbose_report_includes_sentences(self, report):
        """Verbose output includes per-sentence breakdown."""
        text = report.render(verbosity=Verbosity.VERBOSE)
        assert "A big tap." in text
        assert "The top bit." in text

    def test_verbose_report_shows_new_phonemes(self, report):
        """Verbose output shows which phonemes each sentence contributed."""
        text = report.render(verbosity=Verbosity.VERBOSE)
        # Sentence 0 introduced p, b, t
        assert "new" in text.lower() or "New" in text

    def test_verbose_report_shows_phoneme_sources(self, report):
        """Verbose output maps each phoneme to its source sentences."""
        text = report.render(verbosity=Verbosity.VERBOSE)
        assert "source" in text.lower() or "Source" in text


# --- Export methods ---


class TestReportExport:
    """Tests for exporting reports to different formats."""

    @pytest.fixture
    def report(self) -> EvaluationReport:
        return EvaluationReport(
            language="en-us",
            unit="phoneme",
            target_phonemes=["p", "b", "t"],
            covered_phonemes={"p", "b"},
            missing_phonemes={"t"},
            coverage=2 / 3,
            phoneme_counts={"p": 2, "b": 1},
            total_sentences=1,
            sentence_details=[
                SentenceDetail(
                    index=0,
                    text="A pub.",
                    phoneme_count=3,
                    new_phonemes=["p", "b"],
                    all_phonemes=["ə", "p", "ʌ", "b"],
                ),
            ],
            phoneme_sources={"p": [0], "b": [0]},
        )

    def test_to_dict(self, report):
        """to_dict returns a plain Python dict with all data."""
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["language"] == "en-us"
        assert d["coverage"] == pytest.approx(2 / 3)
        assert "p" in d["covered_phonemes"]
        assert "t" in d["missing_phonemes"]
        assert len(d["sentence_details"]) == 1
        assert d["phoneme_counts"]["p"] == 2

    def test_to_json(self, report):
        """to_json returns a valid JSON string."""
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["language"] == "en-us"
        assert parsed["coverage"] == pytest.approx(2 / 3)

    def test_to_json_roundtrip(self, report):
        """JSON export can be deserialized back to matching dict."""
        d = report.to_dict()
        j = report.to_json()
        parsed = json.loads(j)
        # Lists vs sets: to_dict uses lists for JSON compatibility
        assert parsed["language"] == d["language"]
        assert parsed["coverage"] == pytest.approx(d["coverage"])

    def test_to_json_indent(self, report):
        """to_json supports pretty-printing."""
        j = report.to_json(indent=2)
        assert "\n" in j
        assert "  " in j

    def test_to_jsonld_ex(self, report):
        """to_jsonld_ex returns a JSON-LD document with @context."""
        doc = report.to_jsonld_ex()
        assert isinstance(doc, dict)
        assert "@context" in doc
        # Should contain corpusgen-specific terms
        assert "coverage" in doc or "coverage" in str(doc.get("@context", ""))

    def test_to_jsonld_ex_has_type(self, report):
        """JSON-LD export includes a @type."""
        doc = report.to_jsonld_ex()
        assert "@type" in doc

    def test_to_jsonld_ex_valid_json(self, report):
        """JSON-LD export is serializable to valid JSON."""
        doc = report.to_jsonld_ex()
        j = json.dumps(doc)
        parsed = json.loads(j)
        assert parsed["@context"] is not None


# --- Verbosity enum ---


class TestVerbosityEnum:
    """Tests for the Verbosity enum."""

    def test_minimal_value(self):
        assert Verbosity.MINIMAL.value == "minimal"

    def test_normal_value(self):
        assert Verbosity.NORMAL.value == "normal"

    def test_verbose_value(self):
        assert Verbosity.VERBOSE.value == "verbose"

    def test_from_string(self):
        """Verbosity can be constructed from string."""
        assert Verbosity("minimal") == Verbosity.MINIMAL
        assert Verbosity("normal") == Verbosity.NORMAL
        assert Verbosity("verbose") == Verbosity.VERBOSE
