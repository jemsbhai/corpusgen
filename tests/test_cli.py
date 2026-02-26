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


# ===========================================================================
# corpusgen generate
# ===========================================================================


def _mock_generation_result(
    sentences: list[str] | None = None,
    coverage: float = 0.85,
    covered_units: set[str] | None = None,
    missing_units: set[str] | None = None,
    unit: str = "phoneme",
    backend: str = "repository",
    elapsed_seconds: float = 0.5,
    iterations: int = 3,
    stop_reason: str = "target_coverage",
):
    """Create a mock GenerationResult."""
    if sentences is None:
        sentences = ["The cat sat.", "Dogs bark loudly.", "Birds fly high."]
    if covered_units is None:
        covered_units = {"p", "t", "k", "s", "b"}
    if missing_units is None:
        missing_units = {"d", "ɡ"}

    result = MagicMock()
    result.generated_sentences = sentences
    result.generated_phonemes = [["ð", "ə"]] * len(sentences)  # placeholder
    result.coverage = coverage
    result.covered_units = covered_units
    result.missing_units = missing_units
    result.unit = unit
    result.backend = backend
    result.elapsed_seconds = elapsed_seconds
    result.iterations = iterations
    result.stop_reason = stop_reason
    result.num_generated = len(sentences)
    result.metadata = {}
    return result


# Common patch targets for generate tests.
# The generate CLI module will import these — we mock where they're used.
_GEN_PATCHES = {
    "get_inventory": "corpusgen.cli.generate.get_inventory",
    "GenerationLoop": "corpusgen.cli.generate.GenerationLoop",
    "RepositoryBackend": "corpusgen.cli.generate.RepositoryBackend",
    "LLMBackend": "corpusgen.cli.generate.LLMBackend",
    "LocalBackend": "corpusgen.cli.generate.LocalBackend",
}


class TestGenerateCommand:
    """corpusgen generate — generate sentences for phoneme coverage."""

    # ---------------------------------------------------------------
    # Repository backend: happy path
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_repository_basic(self, mock_inv, mock_repo, mock_loop, runner, tmp_path):
        """Basic repository backend invocation with a file."""
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("The cat sat.\nDogs bark.\nBirds fly.\n")

        result = runner.invoke(main, [
            "generate",
            "--backend", "repository",
            "--language", "en-us",
            "--file", str(corpus),
        ])
        assert result.exit_code == 0, result.output
        assert "85.0%" in result.output

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_repository_passes_language_to_from_texts(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "fr-fr",
            "--file", str(corpus),
        ])
        call_kwargs = mock_repo.from_texts.call_args
        assert call_kwargs[1]["language"] == "fr-fr"

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_repository_requires_file(self, mock_inv, mock_repo, mock_loop, runner):
        """Repository backend without --file should error."""
        mock_inv.return_value = _mock_inventory()
        result = runner.invoke(main, [
            "generate", "--backend", "repository", "--language", "en-us",
        ])
        assert result.exit_code != 0
        assert "--file" in result.output.lower() or "error" in result.output.lower()

    # ---------------------------------------------------------------
    # LLM API backend: happy path
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LLMBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_llm_api_basic(self, mock_inv, mock_llm, mock_loop, runner):
        """Basic llm_api backend invocation."""
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result(
            backend="llm_api"
        )

        result = runner.invoke(main, [
            "generate",
            "--backend", "llm_api",
            "--language", "en-us",
            "--model", "openai/gpt-4o-mini",
        ])
        assert result.exit_code == 0, result.output

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LLMBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_llm_api_passes_model_and_key(
        self, mock_inv, mock_llm, mock_loop, runner
    ):
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        runner.invoke(main, [
            "generate", "-b", "llm_api", "-l", "en-us",
            "--model", "openai/gpt-4o-mini",
            "--api-key", "sk-test123",
        ])
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs["api_key"] == "sk-test123"

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LLMBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_llm_api_passes_temperature_and_max_tokens(
        self, mock_inv, mock_llm, mock_loop, runner
    ):
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        runner.invoke(main, [
            "generate", "-b", "llm_api", "-l", "en-us",
            "--model", "openai/gpt-4o-mini",
            "--llm-temperature", "0.5",
            "--llm-max-tokens", "2048",
        ])
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 2048

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LLMBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_llm_api_requires_model(self, mock_inv, mock_llm, mock_loop, runner):
        """llm_api backend without --model should error."""
        mock_inv.return_value = _mock_inventory()
        result = runner.invoke(main, [
            "generate", "--backend", "llm_api", "--language", "en-us",
        ])
        assert result.exit_code != 0
        assert "--model" in result.output.lower() or "error" in result.output.lower()

    # ---------------------------------------------------------------
    # Local backend: happy path
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_local_basic(self, mock_inv, mock_local, mock_loop, runner):
        """Basic local backend invocation."""
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result(
            backend="local"
        )

        result = runner.invoke(main, [
            "generate",
            "--backend", "local",
            "--language", "en-us",
            "--model", "gpt2",
        ])
        assert result.exit_code == 0, result.output

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_local_passes_device_and_quantization(
        self, mock_inv, mock_local, mock_loop, runner
    ):
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--device", "cuda",
            "--quantization", "4bit",
        ])
        call_kwargs = mock_local.call_args[1]
        assert call_kwargs["device"] == "cuda"
        assert call_kwargs["quantization"] == "4bit"

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_local_passes_temperature_and_max_tokens(
        self, mock_inv, mock_local, mock_loop, runner
    ):
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--local-temperature", "0.6",
            "--local-max-tokens", "512",
        ])
        call_kwargs = mock_local.call_args[1]
        assert call_kwargs["temperature"] == 0.6
        assert call_kwargs["max_new_tokens"] == 512

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_local_requires_model(self, mock_inv, mock_local, mock_loop, runner):
        """local backend without --model should error."""
        mock_inv.return_value = _mock_inventory()
        result = runner.invoke(main, [
            "generate", "--backend", "local", "--language", "en-us",
        ])
        assert result.exit_code != 0
        assert "--model" in result.output.lower() or "error" in result.output.lower()

    # ---------------------------------------------------------------
    # Target inventory: PHOIBLE + additive phonemes
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_phoible_target_default(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """Default target should use PHOIBLE."""
        mock_inv.return_value = _mock_inventory(phonemes=["p", "b", "t"])
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
        ])
        mock_inv.assert_called_once()

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_additive_phonemes(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """--phonemes should add to the PHOIBLE inventory."""
        mock_inv.return_value = _mock_inventory(phonemes=["p", "b", "t"])
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--phonemes", "ʃ,ʒ,θ",
        ])
        # The PhoneticTargetInventory should receive all 6 phonemes
        loop_call = mock_loop.call_args
        targets_arg = loop_call[1]["targets"]
        # The target inventory should contain p, b, t (PHOIBLE) + ʃ, ʒ, θ (additive)
        tracker = targets_arg.tracker
        target_set = tracker.target_units
        assert "ʃ" in target_set
        assert "ʒ" in target_set
        assert "θ" in target_set
        assert "p" in target_set

    # ---------------------------------------------------------------
    # Weights: inline and file-based
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_weights_inline(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """Inline weights should be parsed and passed."""
        mock_inv.return_value = _mock_inventory(phonemes=["p", "b", "t"])
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--weights", "p:2.0,b:1.5",
        ])
        loop_call = mock_loop.call_args
        targets_arg = loop_call[1]["targets"]
        assert targets_arg._weights["p"] == 2.0
        assert targets_arg._weights["b"] == 1.5

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_weights_json_file(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """Weights from a JSON file should be loaded and passed."""
        mock_inv.return_value = _mock_inventory(phonemes=["p", "b", "t"])
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        weights_file = tmp_path / "weights.json"
        weights_file.write_text(json.dumps({"p": 3.0, "t": 2.0}))

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--weights", str(weights_file),
        ])
        loop_call = mock_loop.call_args
        targets_arg = loop_call[1]["targets"]
        assert targets_arg._weights["p"] == 3.0
        assert targets_arg._weights["t"] == 2.0

    # ---------------------------------------------------------------
    # Stopping criteria
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_stopping_criteria_passed(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--target-coverage", "0.9",
            "--max-sentences", "50",
            "--max-iterations", "100",
            "--timeout", "60.0",
        ])
        loop_call = mock_loop.call_args
        stopping = loop_call[1]["stopping_criteria"]
        assert stopping.target_coverage == 0.9
        assert stopping.max_sentences == 50
        assert stopping.max_iterations == 100
        assert stopping.timeout_seconds == 60.0

    # ---------------------------------------------------------------
    # Candidates per iteration
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_candidates_per_iteration(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--candidates", "10",
        ])
        loop_call = mock_loop.call_args
        assert loop_call[1]["candidates_per_iteration"] == 10

    # ---------------------------------------------------------------
    # Coverage unit
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_unit_diphone(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--unit", "diphone",
        ])
        loop_call = mock_loop.call_args
        targets_arg = loop_call[1]["targets"]
        assert targets_arg.tracker.unit == "diphone"

    # ---------------------------------------------------------------
    # Output formats
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_text_output_format(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--format", "text",
        ])
        assert result.exit_code == 0
        assert "85.0%" in result.output
        assert "3" in result.output  # num_generated

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_json_output_format(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--format", "json",
            "--max-sentences", "50",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "coverage" in data
        assert data["coverage"] == 0.85
        assert "generated_sentences" in data
        assert "stop_reason" in data

    # ---------------------------------------------------------------
    # Output file
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_output_file(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")
        out = tmp_path / "generated.txt"

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--output", str(out),
        ])
        assert result.exit_code == 0
        assert out.exists()
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    # ---------------------------------------------------------------
    # Validation: cross-backend flag restrictions
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_api_key_invalid_for_repository(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """--api-key should be rejected for repository backend."""
        mock_inv.return_value = _mock_inventory()
        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--api-key", "sk-test",
        ])
        assert result.exit_code != 0

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_device_invalid_for_llm_api(
        self, mock_inv, mock_local, mock_loop, runner
    ):
        """--device should be rejected for llm_api backend."""
        mock_inv.return_value = _mock_inventory()
        result = runner.invoke(main, [
            "generate", "-b", "llm_api", "-l", "en-us",
            "--model", "openai/gpt-4o-mini",
            "--device", "cuda",
        ])
        assert result.exit_code != 0

    # ---------------------------------------------------------------
    # Missing required flags
    # ---------------------------------------------------------------

    def test_missing_backend_errors(self, runner):
        result = runner.invoke(main, [
            "generate", "--language", "en-us",
        ])
        assert result.exit_code != 0

    def test_missing_language_errors(self, runner):
        result = runner.invoke(main, [
            "generate", "--backend", "repository",
        ])
        assert result.exit_code != 0

    # ---------------------------------------------------------------
    # Safety warning for unbounded generation
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_warns_no_safety_limit(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """Should warn on stderr when no safety stopping limit is set."""
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
        ])
        # Should still succeed, but with a warning
        assert result.exit_code == 0
        # Click CliRunner captures stderr in output by default
        assert "warning" in result.output.lower() or "no safety" in result.output.lower()

    # ---------------------------------------------------------------
    # Scorer weights and built-in scorers
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_scorer_weights_passed(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """Scorer weights should be forwarded to PhoneticScorer."""
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "10",
            "--coverage-weight", "0.5",
            "--phonotactic-weight", "0.3",
            "--phonotactic-scorer", "ngram",
            "--fluency-weight", "0.2",
            "--fluency-scorer", "perplexity",
            "--fluency-model", "gpt2",
        ])
        loop_call = mock_loop.call_args
        scorer = loop_call[1]["scorer"]
        assert scorer._coverage_weight == 0.5
        assert scorer._phonotactic_weight == 0.3
        assert scorer._fluency_weight == 0.2

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_phonotactic_weight_without_scorer_errors(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """Non-zero phonotactic weight without --phonotactic-scorer should error."""
        mock_inv.return_value = _mock_inventory()
        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "10",
            "--phonotactic-weight", "0.3",
        ])
        assert result.exit_code != 0

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_fluency_weight_without_scorer_errors(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """Non-zero fluency weight without --fluency-scorer should error."""
        mock_inv.return_value = _mock_inventory()
        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "10",
            "--fluency-weight", "0.2",
        ])
        assert result.exit_code != 0

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_fluency_scorer_without_model_and_non_local_errors(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """Fluency scorer with non-local backend must specify --fluency-model."""
        mock_inv.return_value = _mock_inventory()
        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "10",
            "--fluency-weight", "0.2",
            "--fluency-scorer", "perplexity",
        ])
        assert result.exit_code != 0

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_phonotactic_ngram_scorer_created(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """--phonotactic-scorer ngram should wire up NgramPhonotacticScorer."""
        mock_inv.return_value = _mock_inventory(phonemes=["p", "a", "t", "k"])
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "10",
            "--phonotactic-weight", "0.3",
            "--phonotactic-scorer", "ngram",
        ])
        loop_call = mock_loop.call_args
        scorer = loop_call[1]["scorer"]
        # The phonotactic hook should be set (not None)
        assert scorer._phonotactic_scorer is not None
        # It should be callable
        result = scorer._phonotactic_scorer(["p", "a", "t"])
        assert isinstance(result, float)

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_phonotactic_ngram_order(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """--phonotactic-n should set the n-gram order."""
        mock_inv.return_value = _mock_inventory(phonemes=["p", "a", "t", "k"])
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "10",
            "--phonotactic-weight", "0.3",
            "--phonotactic-scorer", "ngram",
            "--phonotactic-n", "3",
        ])
        loop_call = mock_loop.call_args
        scorer = loop_call[1]["scorer"]
        # The underlying NgramPhonotacticScorer should have n=3
        assert scorer._phonotactic_scorer.n == 3

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_phonotactic_corpus_file(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """--phonotactic-corpus should train from corpus via G2P."""
        mock_inv.return_value = _mock_inventory(phonemes=["p", "a", "t", "k"])
        mock_repo.from_texts.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        ref_corpus = tmp_path / "reference.txt"
        ref_corpus.write_text("The cat sat.\nDogs bark.\n")

        with patch("corpusgen.cli.generate._phonemize_corpus") as mock_g2p:
            mock_g2p.return_value = [
                ["p", "a", "t"],
                ["k", "a", "t"],
            ]
            runner.invoke(main, [
                "generate", "-b", "repository", "-l", "en-us",
                "--file", str(corpus),
                "--max-sentences", "10",
                "--phonotactic-weight", "0.3",
                "--phonotactic-scorer", "ngram",
                "--phonotactic-corpus", str(ref_corpus),
            ])

        loop_call = mock_loop.call_args
        scorer = loop_call[1]["scorer"]
        assert scorer._phonotactic_scorer is not None

    # ---------------------------------------------------------------
    # Guidance strategies (local backend only)
    # ---------------------------------------------------------------

    @patch("corpusgen.cli.generate.DATGStrategy")
    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_guidance_datg_basic(
        self, mock_inv, mock_local, mock_loop, mock_datg, runner
    ):
        """--guidance datg should create DATGStrategy and pass to LocalBackend."""
        mock_inv.return_value = _mock_inventory(phonemes=["p", "a", "t", "k"])
        mock_loop.return_value.run.return_value = _mock_generation_result(
            backend="local"
        )

        result = runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--guidance", "datg",
            "--max-sentences", "10",
        ])
        assert result.exit_code == 0, result.output
        mock_datg.assert_called_once()
        # DATGStrategy should be passed to LocalBackend
        local_kwargs = mock_local.call_args[1]
        assert local_kwargs["guidance_strategy"] is not None

    @patch("corpusgen.cli.generate.DATGStrategy")
    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_guidance_datg_flat_flags(
        self, mock_inv, mock_local, mock_loop, mock_datg, runner
    ):
        """DATG flat flags should be forwarded to DATGStrategy."""
        mock_inv.return_value = _mock_inventory(phonemes=["p", "a", "t", "k"])
        mock_loop.return_value.run.return_value = _mock_generation_result()

        runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--guidance", "datg",
            "--datg-boost", "3.0",
            "--datg-penalty", "-3.0",
            "--datg-anti-mode", "frequency",
            "--datg-freq-threshold", "20",
            "--datg-batch-size", "256",
            "--max-sentences", "10",
        ])
        datg_kwargs = mock_datg.call_args[1]
        assert datg_kwargs["boost_strength"] == 3.0
        assert datg_kwargs["penalty_strength"] == -3.0
        assert datg_kwargs["anti_attribute_mode"] == "frequency"
        assert datg_kwargs["frequency_threshold"] == 20
        assert datg_kwargs["batch_size"] == 256

    @patch("corpusgen.cli.generate.DATGStrategy")
    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_guidance_datg_config_file(
        self, mock_inv, mock_local, mock_loop, mock_datg, runner, tmp_path
    ):
        """--guidance-config should override flat flags."""
        mock_inv.return_value = _mock_inventory(phonemes=["p", "a", "t", "k"])
        mock_loop.return_value.run.return_value = _mock_generation_result()

        config = tmp_path / "datg_config.json"
        config.write_text(json.dumps({
            "boost_strength": 7.0,
            "penalty_strength": -7.0,
            "anti_attribute_mode": "frequency",
        }))

        runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--guidance", "datg",
            "--guidance-config", str(config),
            "--datg-boost", "3.0",  # should be overridden by config
            "--max-sentences", "10",
        ])
        datg_kwargs = mock_datg.call_args[1]
        assert datg_kwargs["boost_strength"] == 7.0  # config wins
        assert datg_kwargs["penalty_strength"] == -7.0

    @patch("corpusgen.cli.generate.PhonRLStrategy")
    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_guidance_rl_basic(
        self, mock_inv, mock_local, mock_loop, mock_rl, runner
    ):
        """--guidance rl should create PhonRLStrategy."""
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result(
            backend="local"
        )

        result = runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--guidance", "rl",
            "--max-sentences", "10",
        ])
        assert result.exit_code == 0, result.output
        mock_rl.assert_called_once()

    @patch("corpusgen.cli.generate.PhonRLStrategy")
    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_guidance_rl_adapter_path(
        self, mock_inv, mock_local, mock_loop, mock_rl, runner, tmp_path
    ):
        """--rl-adapter-path should be forwarded to PhonRLStrategy."""
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--guidance", "rl",
            "--rl-adapter-path", str(adapter_dir),
            "--max-sentences", "10",
        ])
        rl_kwargs = mock_rl.call_args[1]
        assert rl_kwargs["adapter_path"] == str(adapter_dir)

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LLMBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_guidance_invalid_for_non_local(
        self, mock_inv, mock_llm, mock_loop, runner
    ):
        """--guidance should be rejected for non-local backends."""
        mock_inv.return_value = _mock_inventory()
        result = runner.invoke(main, [
            "generate", "-b", "llm_api", "-l", "en-us",
            "--model", "openai/gpt-4o-mini",
            "--guidance", "datg",
            "--max-sentences", "10",
        ])
        assert result.exit_code != 0

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_guidance_none_no_strategy(
        self, mock_inv, mock_local, mock_loop, runner
    ):
        """--guidance none should not pass a strategy to LocalBackend."""
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--guidance", "none",
            "--max-sentences", "10",
        ])
        local_kwargs = mock_local.call_args[1]
        assert local_kwargs.get("guidance_strategy") is None

    # ---------------------------------------------------------------
    # HuggingFace dataset source (repository backend)
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_dataset_basic(
        self, mock_inv, mock_repo, mock_loop, runner
    ):
        """--dataset should use RepositoryBackend.from_huggingface."""
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_huggingface.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--dataset", "wikitext",
            "--max-sentences", "10",
        ])
        assert result.exit_code == 0, result.output
        mock_repo.from_huggingface.assert_called_once()

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_dataset_passes_options(
        self, mock_inv, mock_repo, mock_loop, runner
    ):
        """Dataset options should be forwarded to from_huggingface."""
        mock_inv.return_value = _mock_inventory()
        mock_repo.from_huggingface.return_value = MagicMock()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--dataset", "wikitext",
            "--text-column", "sentence",
            "--split", "train",
            "--max-samples", "500",
            "--max-sentences", "10",
        ])
        call_kwargs = mock_repo.from_huggingface.call_args[1]
        assert call_kwargs["dataset_name"] == "wikitext"
        assert call_kwargs["text_column"] == "sentence"
        assert call_kwargs["split"] == "train"
        assert call_kwargs["max_samples"] == 500
        assert call_kwargs["language"] == "en-us"

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_dataset_and_file_mutually_exclusive(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """--dataset and --file cannot be used together."""
        mock_inv.return_value = _mock_inventory()
        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--dataset", "wikitext",
            "--max-sentences", "10",
        ])
        assert result.exit_code != 0

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LLMBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_dataset_invalid_for_non_repository(
        self, mock_inv, mock_llm, mock_loop, runner
    ):
        """--dataset should be rejected for non-repository backends."""
        mock_inv.return_value = _mock_inventory()
        result = runner.invoke(main, [
            "generate", "-b", "llm_api", "-l", "en-us",
            "--model", "openai/gpt-4o-mini",
            "--dataset", "wikitext",
            "--max-sentences", "10",
        ])
        assert result.exit_code != 0

    # ---------------------------------------------------------------
    # Custom prompt templates
    # ---------------------------------------------------------------

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LLMBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_prompt_template_inline_llm(
        self, mock_inv, mock_llm, mock_loop, runner
    ):
        """Inline --prompt-template should be passed to LLMBackend."""
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        tpl = "Generate {k} sentences with {target_units} in {language}"
        runner.invoke(main, [
            "generate", "-b", "llm_api", "-l", "en-us",
            "--model", "openai/gpt-4o-mini",
            "--prompt-template", tpl,
            "--max-sentences", "10",
        ])
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["prompt_template"] == tpl

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LLMBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_prompt_template_from_file(
        self, mock_inv, mock_llm, mock_loop, runner, tmp_path
    ):
        """--prompt-template pointing to a file should load its contents."""
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        tpl_file = tmp_path / "prompt.txt"
        tpl_content = "Custom: {target_units} for {language}, {k} sentences."
        tpl_file.write_text(tpl_content)

        runner.invoke(main, [
            "generate", "-b", "llm_api", "-l", "en-us",
            "--model", "openai/gpt-4o-mini",
            "--prompt-template", str(tpl_file),
            "--max-sentences", "10",
        ])
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["prompt_template"] == tpl_content

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_prompt_template_local_backend(
        self, mock_inv, mock_local, mock_loop, runner
    ):
        """--prompt-template should also work for local backend."""
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        tpl = "Generate {k} sentences with {target_units}"
        runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--prompt-template", tpl,
            "--max-sentences", "10",
        ])
        call_kwargs = mock_local.call_args[1]
        assert call_kwargs["prompt_template"] == tpl

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["RepositoryBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_prompt_template_invalid_for_repository(
        self, mock_inv, mock_repo, mock_loop, runner, tmp_path
    ):
        """--prompt-template should be rejected for repository backend."""
        mock_inv.return_value = _mock_inventory()
        corpus = tmp_path / "pool.txt"
        corpus.write_text("Hello.\n")

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--prompt-template", "Some template {target_units}",
            "--max-sentences", "10",
        ])
        assert result.exit_code != 0

    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LLMBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_prompt_template_must_contain_target_units(
        self, mock_inv, mock_llm, mock_loop, runner
    ):
        """Prompt template without {target_units} should error."""
        mock_inv.return_value = _mock_inventory()
        result = runner.invoke(main, [
            "generate", "-b", "llm_api", "-l", "en-us",
            "--model", "openai/gpt-4o-mini",
            "--prompt-template", "No placeholder here",
            "--max-sentences", "10",
        ])
        assert result.exit_code != 0

    # ---------------------------------------------------------------
    # Model sharing: fluency scorer + local backend
    # ---------------------------------------------------------------

    @patch("corpusgen.cli.generate.PerplexityFluencyScorer")
    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_fluency_shares_model_with_local_backend(
        self, mock_inv, mock_local, mock_loop, mock_fluency, runner
    ):
        """When fluency model matches local backend model, should share via from_model."""
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()

        # Set up the mock backend with _model and _tokenizer
        mock_backend_instance = MagicMock()
        mock_backend_instance._model = MagicMock()
        mock_backend_instance._tokenizer = MagicMock()
        mock_backend_instance.is_loaded = True
        mock_local.return_value = mock_backend_instance

        mock_fluency.from_model.return_value = MagicMock()

        runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--fluency-weight", "0.2",
            "--fluency-scorer", "perplexity",
            "--max-sentences", "10",
        ])
        # Backend should have been forced to load early
        mock_backend_instance._ensure_loaded.assert_called_once()
        # Fluency scorer should use from_model with the backend's model
        mock_fluency.from_model.assert_called_once_with(
            mock_backend_instance._model,
            mock_backend_instance._tokenizer,
        )

    @patch("corpusgen.cli.generate.PerplexityFluencyScorer")
    @patch(_GEN_PATCHES["GenerationLoop"])
    @patch(_GEN_PATCHES["LocalBackend"])
    @patch(_GEN_PATCHES["get_inventory"])
    def test_fluency_separate_model_when_different(
        self, mock_inv, mock_local, mock_loop, mock_fluency, runner
    ):
        """When --fluency-model differs from --model, should load independently."""
        mock_inv.return_value = _mock_inventory()
        mock_loop.return_value.run.return_value = _mock_generation_result()
        mock_local.return_value = MagicMock()

        runner.invoke(main, [
            "generate", "-b", "local", "-l", "en-us",
            "--model", "gpt2",
            "--fluency-weight", "0.2",
            "--fluency-scorer", "perplexity",
            "--fluency-model", "gpt2-medium",
            "--max-sentences", "10",
        ])
        # from_model should NOT be called — different models
        mock_fluency.from_model.assert_not_called()
        # Regular constructor should be used
        mock_fluency.assert_called_once()


# ===========================================================================
# Slow integration tests — real espeak-ng, real PHOIBLE, no mocks
# ===========================================================================


@pytest.mark.slow
class TestGenerateIntegration:
    """End-to-end CLI tests using real G2P and PHOIBLE.

    These require espeak-ng installed and PHOIBLE data available.
    Run with: pytest -m slow
    """

    def test_repository_real_english(self, runner, tmp_path):
        """Repository backend with real English sentences should produce output."""
        corpus = tmp_path / "pool.txt"
        corpus.write_text(
            "The cat sat on the mat.\n"
            "Dogs bark loudly at night.\n"
            "Birds fly high in the sky.\n"
            "She sells sea shells by the shore.\n"
            "Peter Piper picked a peck of pickled peppers.\n"
            "The quick brown fox jumps over the lazy dog.\n"
            "How much wood would a woodchuck chuck.\n"
            "Red lorry yellow lorry.\n"
            "Unique New York.\n"
            "Freshly fried fresh fish.\n"
        )

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "5",
            "--format", "text",
        ])
        assert result.exit_code == 0, result.output
        assert "Generated" in result.output
        assert "Coverage" in result.output

    def test_repository_json_output(self, runner, tmp_path):
        """JSON output should be valid and contain expected fields."""
        corpus = tmp_path / "pool.txt"
        corpus.write_text(
            "The cat sat on the mat.\n"
            "Dogs bark loudly at night.\n"
            "She sells sea shells by the shore.\n"
        )

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "3",
            "--format", "json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "generated_sentences" in data
        assert "coverage" in data
        assert isinstance(data["coverage"], float)
        assert 0.0 <= data["coverage"] <= 1.0
        assert len(data["generated_sentences"]) <= 3

    def test_repository_output_file(self, runner, tmp_path):
        """--output should write sentences to a file."""
        corpus = tmp_path / "pool.txt"
        corpus.write_text(
            "The quick brown fox.\n"
            "Jumps over the lazy dog.\n"
        )
        out = tmp_path / "generated.txt"

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "2",
            "--output", str(out),
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1

    def test_repository_additive_phonemes(self, runner, tmp_path):
        """--phonemes should expand target inventory beyond PHOIBLE."""
        corpus = tmp_path / "pool.txt"
        corpus.write_text(
            "The cat sat on the mat.\n"
            "Dogs bark loudly.\n"
        )

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "2",
            "--phonemes", "ʃ,ʒ",
            "--format", "json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        # The target set should include the additive phonemes
        all_units = set(data["covered_units"]) | set(data["missing_units"])
        assert "ʃ" in all_units or "ʒ" in all_units

    def test_repository_diphone_unit(self, runner, tmp_path):
        """--unit diphone should track diphone coverage."""
        corpus = tmp_path / "pool.txt"
        corpus.write_text(
            "The cat sat on the mat.\n"
            "Dogs bark loudly at night.\n"
            "She sells sea shells.\n"
        )

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "3",
            "--unit", "diphone",
            "--format", "json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["unit"] == "diphone"

    def test_repository_with_phonotactic_scorer(self, runner, tmp_path):
        """Phonotactic scorer should integrate without errors."""
        corpus = tmp_path / "pool.txt"
        corpus.write_text(
            "The cat sat on the mat.\n"
            "Dogs bark loudly at night.\n"
            "She sells sea shells by the shore.\n"
        )

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "3",
            "--coverage-weight", "0.7",
            "--phonotactic-weight", "0.3",
            "--phonotactic-scorer", "ngram",
            "--format", "json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["coverage"] >= 0.0

    def test_repository_stops_at_max_sentences(self, runner, tmp_path):
        """Should respect --max-sentences limit."""
        corpus = tmp_path / "pool.txt"
        # Write many sentences
        corpus.write_text("\n".join(
            [f"Sentence number {i} with words." for i in range(50)]
        ) + "\n")

        result = runner.invoke(main, [
            "generate", "-b", "repository", "-l", "en-us",
            "--file", str(corpus),
            "--max-sentences", "5",
            "--format", "json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["num_generated"] <= 5


@pytest.mark.slow
class TestInventoryIntegration:
    """End-to-end inventory command with real PHOIBLE."""

    def test_english_inventory(self, runner):
        """Should display real English phoneme inventory."""
        result = runner.invoke(main, ["inventory", "--language", "en-us"])
        assert result.exit_code == 0
        # English should have common phonemes
        assert "p" in result.output

    def test_english_json(self, runner):
        """JSON output should be valid."""
        result = runner.invoke(main, [
            "inventory", "--language", "en-us", "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "phonemes" in data
        assert len(data["phonemes"]) > 10  # English has plenty


@pytest.mark.slow
class TestEvaluateIntegration:
    """End-to-end evaluate command with real G2P."""

    def test_evaluate_english_sentence(self, runner):
        """Should evaluate coverage of a real English sentence."""
        result = runner.invoke(main, [
            "evaluate",
            "The quick brown fox jumps over the lazy dog.",
            "--language", "en-us",
        ])
        assert result.exit_code == 0
        assert "Coverage" in result.output or "coverage" in result.output

    def test_evaluate_json(self, runner):
        """JSON output should contain coverage data."""
        result = runner.invoke(main, [
            "evaluate",
            "She sells sea shells by the shore.",
            "--language", "en-us",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "coverage" in data
        assert 0.0 < data["coverage"] <= 1.0
