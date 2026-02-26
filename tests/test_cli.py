"""Tests for the corpusgen CLI.

Test tiers:
    - **Fast tests** (default): Use Click's CliRunner with mocks at the
      library boundary (PhoibleDataset, G2PManager). No espeak-ng or
      PHOIBLE data required.
    - **Slow tests** (@pytest.mark.slow): Real espeak-ng, real PHOIBLE.
      Skipped by default.

All subcommands are tested for:
    - Happy path with default and custom options
    - Error handling (missing args, invalid inputs)
    - Output formats (text, json)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from corpusgen.cli import main


# ===========================================================================
# Helpers
# ===========================================================================


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


def _mock_inventory(
    phonemes: list[str] | None = None,
    consonants: list[str] | None = None,
    vowels: list[str] | None = None,
    language_name: str = "English",
    iso639_3: str = "eng",
    glottocode: str = "stan1293",
    source: str = "spa",
    inventory_id: int = 1,
):
    """Create a mock Inventory object."""
    if phonemes is None:
        phonemes = ["p", "b", "t", "d", "k", "ɡ", "m", "n", "ŋ", "iː", "ɪ", "ɛ", "æ"]
    if consonants is None:
        consonants = [p for p in phonemes if p not in {"iː", "ɪ", "ɛ", "æ"}]
    if vowels is None:
        vowels = [p for p in phonemes if p in {"iː", "ɪ", "ɛ", "æ"}]

    inv = MagicMock()
    inv.phonemes = phonemes
    inv.consonants = consonants
    inv.vowels = vowels
    inv.language_name = language_name
    inv.iso639_3 = iso639_3
    inv.glottocode = glottocode
    inv.source = source
    inv.inventory_id = inventory_id
    inv.segments = [MagicMock(phoneme=p) for p in phonemes]
    return inv


# ===========================================================================
# corpusgen --help / --version
# ===========================================================================


class TestTopLevel:
    """Top-level CLI group."""

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "corpusgen" in result.output.lower()

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0." in result.output  # version string present


# ===========================================================================
# corpusgen inventory
# ===========================================================================


class TestInventoryCommand:
    """corpusgen inventory — show PHOIBLE phoneme inventory."""

    @patch("corpusgen.cli.inventory.get_inventory")
    def test_basic_invocation(self, mock_get, runner):
        mock_get.return_value = _mock_inventory()
        result = runner.invoke(main, ["inventory", "--language", "en-us"])
        assert result.exit_code == 0
        assert "English" in result.output

    @patch("corpusgen.cli.inventory.get_inventory")
    def test_shows_phonemes(self, mock_get, runner):
        mock_get.return_value = _mock_inventory()
        result = runner.invoke(main, ["inventory", "--language", "en-us"])
        assert result.exit_code == 0
        # Should show consonants and vowels
        assert "p" in result.output
        assert "iː" in result.output

    @patch("corpusgen.cli.inventory.get_inventory")
    def test_shows_consonant_and_vowel_counts(self, mock_get, runner):
        mock_get.return_value = _mock_inventory()
        result = runner.invoke(main, ["inventory", "--language", "en-us"])
        assert result.exit_code == 0
        # Should report counts
        assert "9" in result.output  # 9 consonants
        assert "4" in result.output  # 4 vowels

    @patch("corpusgen.cli.inventory.get_inventory")
    def test_json_format(self, mock_get, runner):
        mock_get.return_value = _mock_inventory()
        result = runner.invoke(main, ["inventory", "--language", "en-us", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "phonemes" in data
        assert "language_name" in data
        assert data["language_name"] == "English"

    @patch("corpusgen.cli.inventory.get_inventory")
    def test_source_filter(self, mock_get, runner):
        mock_get.return_value = _mock_inventory()
        result = runner.invoke(main, ["inventory", "--language", "en-us", "--source", "upsid"])
        assert result.exit_code == 0
        mock_get.assert_called_once_with("en-us", source="upsid")

    @patch("corpusgen.cli.inventory.get_inventory")
    def test_no_source_passes_none(self, mock_get, runner):
        mock_get.return_value = _mock_inventory()
        result = runner.invoke(main, ["inventory", "--language", "en-us"])
        assert result.exit_code == 0
        mock_get.assert_called_once_with("en-us", source=None)

    def test_missing_language_errors(self, runner):
        result = runner.invoke(main, ["inventory"])
        assert result.exit_code != 0

    @patch("corpusgen.cli.inventory.get_inventory")
    def test_unknown_language_reports_error(self, mock_get, runner):
        mock_get.side_effect = KeyError("not found")
        result = runner.invoke(main, ["inventory", "--language", "xx-nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()


# ===========================================================================
# corpusgen evaluate
# ===========================================================================


def _mock_evaluate(
    sentences: list[str],
    language: str = "en-us",
    target_phonemes=None,
    unit: str = "phoneme",
):
    """Return a mock EvaluationReport."""
    from corpusgen.evaluate.report import EvaluationReport

    return EvaluationReport(
        language=language,
        unit=unit,
        target_phonemes=["p", "b", "t", "d", "k"],
        covered_phonemes={"p", "t", "k"},
        missing_phonemes={"b", "d"},
        coverage=0.6,
        phoneme_counts={"p": 3, "t": 5, "k": 2},
        total_sentences=len(sentences),
    )


class TestEvaluateCommand:
    """corpusgen evaluate — evaluate text for phoneme coverage."""

    @patch("corpusgen.cli.evaluate.evaluate")
    def test_inline_text(self, mock_eval, runner):
        mock_eval.side_effect = _mock_evaluate
        result = runner.invoke(main, [
            "evaluate", "The cat sat on the mat.", "--language", "en-us",
        ])
        assert result.exit_code == 0
        assert "60.0%" in result.output

    @patch("corpusgen.cli.evaluate.evaluate")
    def test_multiple_inline_sentences(self, mock_eval, runner):
        mock_eval.side_effect = _mock_evaluate
        result = runner.invoke(main, [
            "evaluate",
            "The cat sat.",
            "The dog ran.",
            "--language", "en-us",
        ])
        assert result.exit_code == 0
        # evaluate should receive both sentences
        call_args = mock_eval.call_args
        assert len(call_args[0][0]) == 2  # two sentences

    @patch("corpusgen.cli.evaluate.evaluate")
    def test_file_input(self, mock_eval, runner, tmp_path):
        mock_eval.side_effect = _mock_evaluate
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("The cat sat.\nThe dog ran.\nBirds fly.\n")
        result = runner.invoke(main, [
            "evaluate", "--file", str(corpus), "--language", "en-us",
        ])
        assert result.exit_code == 0
        call_args = mock_eval.call_args
        assert len(call_args[0][0]) == 3

    @patch("corpusgen.cli.evaluate.evaluate")
    def test_file_skips_blank_lines(self, mock_eval, runner, tmp_path):
        mock_eval.side_effect = _mock_evaluate
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("Line one.\n\nLine two.\n\n")
        result = runner.invoke(main, [
            "evaluate", "--file", str(corpus), "--language", "en-us",
        ])
        assert result.exit_code == 0
        call_args = mock_eval.call_args
        assert len(call_args[0][0]) == 2

    @patch("corpusgen.cli.evaluate.evaluate")
    def test_phoible_target(self, mock_eval, runner):
        mock_eval.side_effect = _mock_evaluate
        result = runner.invoke(main, [
            "evaluate", "Hello world.",
            "--language", "en-us",
            "--target", "phoible",
        ])
        assert result.exit_code == 0
        call_args = mock_eval.call_args
        assert call_args[1]["target_phonemes"] == "phoible"

    @patch("corpusgen.cli.evaluate.evaluate")
    def test_unit_diphone(self, mock_eval, runner):
        mock_eval.side_effect = _mock_evaluate
        result = runner.invoke(main, [
            "evaluate", "Hello.", "--language", "en-us", "--unit", "diphone",
        ])
        assert result.exit_code == 0
        call_args = mock_eval.call_args
        assert call_args[1]["unit"] == "diphone"

    @patch("corpusgen.cli.evaluate.evaluate")
    def test_json_format(self, mock_eval, runner):
        mock_eval.side_effect = _mock_evaluate
        result = runner.invoke(main, [
            "evaluate", "Hello.", "--language", "en-us", "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "coverage" in data
        assert data["coverage"] == 0.6

    @patch("corpusgen.cli.evaluate.evaluate")
    def test_jsonld_format(self, mock_eval, runner):
        mock_eval.side_effect = _mock_evaluate
        result = runner.invoke(main, [
            "evaluate", "Hello.", "--language", "en-us", "--format", "jsonld",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "@context" in data
        assert "@type" in data

    @patch("corpusgen.cli.evaluate.evaluate")
    def test_verbose_flag(self, mock_eval, runner):
        mock_eval.side_effect = _mock_evaluate
        result = runner.invoke(main, [
            "evaluate", "Hello.", "--language", "en-us", "--verbosity", "verbose",
        ])
        assert result.exit_code == 0

    def test_no_input_errors(self, runner):
        """Neither inline text nor --file should error."""
        result = runner.invoke(main, ["evaluate", "--language", "en-us"])
        assert result.exit_code != 0

    def test_file_not_found_errors(self, runner):
        result = runner.invoke(main, [
            "evaluate", "--file", "nonexistent.txt", "--language", "en-us",
        ])
        assert result.exit_code != 0


# ===========================================================================
# corpusgen select
# ===========================================================================


def _mock_select_sentences(
    candidates,
    language="en-us",
    target_phonemes=None,
    unit="phoneme",
    algorithm="greedy",
    max_sentences=None,
    target_coverage=1.0,
    **kwargs,
):
    """Return a mock SelectionResult."""
    from corpusgen.select.result import SelectionResult

    n = min(len(candidates), max_sentences or len(candidates))
    return SelectionResult(
        selected_indices=list(range(n)),
        selected_sentences=candidates[:n],
        coverage=0.85,
        covered_units={"p", "t", "k", "s"},
        missing_units={"b", "d"},
        unit=unit,
        algorithm=algorithm,
        elapsed_seconds=0.01,
        iterations=n,
    )


class TestSelectCommand:
    """corpusgen select — select optimal sentences from a candidate file."""

    @patch("corpusgen.cli.select.select_sentences")
    def test_basic_file_input(self, mock_sel, runner, tmp_path):
        mock_sel.side_effect = _mock_select_sentences
        corpus = tmp_path / "candidates.txt"
        corpus.write_text("The cat sat.\nThe dog ran.\nBirds fly.\n")
        result = runner.invoke(main, [
            "select", "--file", str(corpus), "--language", "en-us",
        ])
        assert result.exit_code == 0
        assert "85.0%" in result.output

    @patch("corpusgen.cli.select.select_sentences")
    def test_shows_selected_count(self, mock_sel, runner, tmp_path):
        mock_sel.side_effect = _mock_select_sentences
        corpus = tmp_path / "candidates.txt"
        corpus.write_text("Sentence one.\nSentence two.\nSentence three.\n")
        result = runner.invoke(main, [
            "select", "--file", str(corpus), "--language", "en-us",
        ])
        assert result.exit_code == 0
        assert "3" in result.output  # 3 selected

    @patch("corpusgen.cli.select.select_sentences")
    def test_algorithm_option(self, mock_sel, runner, tmp_path):
        mock_sel.side_effect = _mock_select_sentences
        corpus = tmp_path / "candidates.txt"
        corpus.write_text("Hello.\n")
        result = runner.invoke(main, [
            "select", "--file", str(corpus), "--language", "en-us",
            "--algorithm", "celf",
        ])
        assert result.exit_code == 0
        call_kwargs = mock_sel.call_args[1]
        assert call_kwargs["algorithm"] == "celf"

    @patch("corpusgen.cli.select.select_sentences")
    def test_max_sentences_option(self, mock_sel, runner, tmp_path):
        mock_sel.side_effect = _mock_select_sentences
        corpus = tmp_path / "candidates.txt"
        corpus.write_text("A.\nB.\nC.\nD.\nE.\n")
        result = runner.invoke(main, [
            "select", "--file", str(corpus), "--language", "en-us",
            "--max-sentences", "3",
        ])
        assert result.exit_code == 0
        call_kwargs = mock_sel.call_args[1]
        assert call_kwargs["max_sentences"] == 3

    @patch("corpusgen.cli.select.select_sentences")
    def test_target_coverage_option(self, mock_sel, runner, tmp_path):
        mock_sel.side_effect = _mock_select_sentences
        corpus = tmp_path / "candidates.txt"
        corpus.write_text("Hello.\n")
        result = runner.invoke(main, [
            "select", "--file", str(corpus), "--language", "en-us",
            "--target-coverage", "0.9",
        ])
        assert result.exit_code == 0
        call_kwargs = mock_sel.call_args[1]
        assert call_kwargs["target_coverage"] == 0.9

    @patch("corpusgen.cli.select.select_sentences")
    def test_phoible_target(self, mock_sel, runner, tmp_path):
        mock_sel.side_effect = _mock_select_sentences
        corpus = tmp_path / "candidates.txt"
        corpus.write_text("Hello.\n")
        result = runner.invoke(main, [
            "select", "--file", str(corpus), "--language", "en-us",
            "--target", "phoible",
        ])
        assert result.exit_code == 0
        call_kwargs = mock_sel.call_args[1]
        assert call_kwargs["target_phonemes"] == "phoible"

    @patch("corpusgen.cli.select.select_sentences")
    def test_unit_diphone(self, mock_sel, runner, tmp_path):
        mock_sel.side_effect = _mock_select_sentences
        corpus = tmp_path / "candidates.txt"
        corpus.write_text("Hello.\n")
        result = runner.invoke(main, [
            "select", "--file", str(corpus), "--language", "en-us",
            "--unit", "diphone",
        ])
        assert result.exit_code == 0
        call_kwargs = mock_sel.call_args[1]
        assert call_kwargs["unit"] == "diphone"

    @patch("corpusgen.cli.select.select_sentences")
    def test_json_format(self, mock_sel, runner, tmp_path):
        mock_sel.side_effect = _mock_select_sentences
        corpus = tmp_path / "candidates.txt"
        corpus.write_text("Hello.\nWorld.\n")
        result = runner.invoke(main, [
            "select", "--file", str(corpus), "--language", "en-us",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "coverage" in data
        assert "selected_sentences" in data
        assert data["coverage"] == 0.85

    @patch("corpusgen.cli.select.select_sentences")
    def test_output_file(self, mock_sel, runner, tmp_path):
        mock_sel.side_effect = _mock_select_sentences
        corpus = tmp_path / "candidates.txt"
        corpus.write_text("Sentence one.\nSentence two.\n")
        out = tmp_path / "selected.txt"
        result = runner.invoke(main, [
            "select", "--file", str(corpus), "--language", "en-us",
            "--output", str(out),
        ])
        assert result.exit_code == 0
        assert out.exists()
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_missing_file_errors(self, runner):
        result = runner.invoke(main, ["select", "--language", "en-us"])
        assert result.exit_code != 0

    def test_file_not_found_errors(self, runner):
        result = runner.invoke(main, [
            "select", "--file", "nonexistent.txt", "--language", "en-us",
        ])
        assert result.exit_code != 0
