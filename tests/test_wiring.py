"""Tests for the wiring layer: convenience API that connects all components.

TDD RED phase. Tests:
1. PhoibleDataset.get_inventory_for_espeak() — espeak code → Inventory
2. evaluate(target_phonemes="phoible") — magic string shortcut
3. Top-level corpusgen.evaluate() and corpusgen.get_inventory()
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from corpusgen.inventory.models import Inventory
from corpusgen.inventory.phoible import PhoibleDataset
from corpusgen.inventory.mapping import EspeakMapping
from corpusgen.evaluate.report import EvaluationReport


# Skip if PHOIBLE not cached (these are integration-level tests)
PHOIBLE_PATH = Path.home() / ".corpusgen" / "phoible.csv"
pytestmark = pytest.mark.skipif(
    not PHOIBLE_PATH.is_file(),
    reason="PHOIBLE CSV not cached",
)


@pytest.fixture(scope="module")
def phoible() -> PhoibleDataset:
    ds = PhoibleDataset()
    ds.load()
    return ds


# ---------------------------------------------------------------------------
# 1. PhoibleDataset.get_inventory_for_espeak()
# ---------------------------------------------------------------------------


class TestGetInventoryForEspeak:
    """PhoibleDataset should accept espeak codes directly."""

    def test_english_us(self, phoible):
        inv = phoible.get_inventory_for_espeak("en-us")
        assert isinstance(inv, Inventory)
        assert inv.iso639_3 == "eng"

    def test_english_gb(self, phoible):
        inv = phoible.get_inventory_for_espeak("en-gb")
        assert inv.iso639_3 == "eng"

    def test_french(self, phoible):
        inv = phoible.get_inventory_for_espeak("fr-fr")
        assert inv.iso639_3 == "fra"

    def test_arabic(self, phoible):
        inv = phoible.get_inventory_for_espeak("ar")
        assert inv.iso639_3 == "arb"

    def test_mandarin(self, phoible):
        inv = phoible.get_inventory_for_espeak("cmn")
        assert inv.iso639_3 == "cmn"

    def test_japanese(self, phoible):
        inv = phoible.get_inventory_for_espeak("ja")
        assert inv.iso639_3 == "jpn"

    def test_case_insensitive(self, phoible):
        inv = phoible.get_inventory_for_espeak("EN-US")
        assert inv.iso639_3 == "eng"

    def test_with_source(self, phoible):
        inv = phoible.get_inventory_for_espeak("en-us", source="spa")
        assert inv.source == "spa"

    def test_unknown_espeak_raises(self, phoible):
        with pytest.raises(KeyError):
            phoible.get_inventory_for_espeak("xxx-unknown")

    def test_returns_segments(self, phoible):
        inv = phoible.get_inventory_for_espeak("de")
        assert inv.size > 0
        assert inv.consonant_count > 0
        assert inv.vowel_count > 0

    def test_union_for_espeak(self, phoible):
        """get_union_inventory_for_espeak also works."""
        inv = phoible.get_union_inventory_for_espeak("en-us")
        assert isinstance(inv, Inventory)
        assert inv.source == "union"
        assert inv.iso639_3 == "eng"


# ---------------------------------------------------------------------------
# 2. evaluate(target_phonemes="phoible")
# ---------------------------------------------------------------------------


class TestEvaluatePhoibleShortcut:
    """evaluate() accepts 'phoible' as target_phonemes value."""

    def test_phoible_string_returns_report(self):
        from corpusgen.evaluate import evaluate
        report = evaluate(
            ["The quick brown fox jumps over the lazy dog."],
            language="en-us",
            target_phonemes="phoible",
        )
        assert isinstance(report, EvaluationReport)

    def test_phoible_string_uses_phoible_inventory(self):
        """Coverage should differ from derived (target=None) because
        PHOIBLE inventory is typically larger than what's in the corpus."""
        from corpusgen.evaluate import evaluate
        report_phoible = evaluate(
            ["The cat sat on the mat."],
            language="en-us",
            target_phonemes="phoible",
        )
        report_derived = evaluate(
            ["The cat sat on the mat."],
            language="en-us",
            target_phonemes=None,
        )
        # Derived = 100%, PHOIBLE target should be < 100% for a short sentence
        assert report_derived.coverage == pytest.approx(1.0)
        assert report_phoible.coverage < 1.0

    def test_phoible_string_has_missing_phonemes(self):
        from corpusgen.evaluate import evaluate
        report = evaluate(
            ["Hello world."],
            language="en-us",
            target_phonemes="phoible",
        )
        assert len(report.missing_phonemes) > 0

    def test_phoible_string_invariants(self):
        from corpusgen.evaluate import evaluate
        report = evaluate(
            ["The quick brown fox.", "She sells seashells."],
            language="en-us",
            target_phonemes="phoible",
        )
        assert report.covered_phonemes | report.missing_phonemes == set(report.target_phonemes)
        assert report.covered_phonemes & report.missing_phonemes == set()
        assert 0.0 < report.coverage < 1.0

    def test_phoible_french(self):
        from corpusgen.evaluate import evaluate
        report = evaluate(
            ["Bonjour le monde."],
            language="fr-fr",
            target_phonemes="phoible",
        )
        assert isinstance(report, EvaluationReport)
        assert report.language == "fr-fr"
        assert report.coverage > 0.0

    def test_phoible_arabic(self):
        from corpusgen.evaluate import evaluate
        report = evaluate(
            ["مرحبا بالعالم"],
            language="ar",
            target_phonemes="phoible",
        )
        assert isinstance(report, EvaluationReport)
        assert report.coverage > 0.0

    def test_phoible_with_unit_diphone(self):
        from corpusgen.evaluate import evaluate
        report = evaluate(
            ["The quick brown fox jumps."],
            language="en-us",
            target_phonemes="phoible",
            unit="diphone",
        )
        assert report.unit == "diphone"
        assert isinstance(report.coverage, float)

    def test_phoible_case_insensitive(self):
        """'PHOIBLE' and 'Phoible' should also work."""
        from corpusgen.evaluate import evaluate
        report = evaluate(
            ["Hello world."],
            language="en-us",
            target_phonemes="PHOIBLE",
        )
        assert isinstance(report, EvaluationReport)


# ---------------------------------------------------------------------------
# 3. Top-level convenience: corpusgen.evaluate, corpusgen.get_inventory
# ---------------------------------------------------------------------------


class TestTopLevelConvenience:
    """corpusgen package exposes evaluate() and get_inventory() at top level."""

    def test_import_evaluate(self):
        from corpusgen import evaluate
        assert callable(evaluate)

    def test_import_get_inventory(self):
        from corpusgen import get_inventory
        assert callable(get_inventory)

    def test_get_inventory_espeak_code(self):
        from corpusgen import get_inventory
        inv = get_inventory("en-us")
        assert isinstance(inv, Inventory)
        assert inv.iso639_3 == "eng"

    def test_get_inventory_iso_code(self):
        from corpusgen import get_inventory
        inv = get_inventory("eng")
        assert isinstance(inv, Inventory)
        assert inv.iso639_3 == "eng"

    def test_get_inventory_with_source(self):
        from corpusgen import get_inventory
        inv = get_inventory("en-us", source="spa")
        assert inv.source == "spa"

    def test_top_level_evaluate_with_phoible(self):
        from corpusgen import evaluate
        report = evaluate(
            ["The cat sat on the mat."],
            language="en-us",
            target_phonemes="phoible",
        )
        assert isinstance(report, EvaluationReport)
        assert report.coverage > 0.0

    def test_top_level_evaluate_with_get_inventory(self):
        from corpusgen import evaluate, get_inventory
        inv = get_inventory("en-us")
        report = evaluate(
            ["The cat sat on the mat."],
            language="en-us",
            target_phonemes=inv.phonemes,
        )
        assert isinstance(report, EvaluationReport)

    def test_top_level_evaluate_none_target(self):
        """Existing behavior still works."""
        from corpusgen import evaluate
        report = evaluate(
            ["Hello world."],
            language="en-us",
            target_phonemes=None,
        )
        assert report.coverage == pytest.approx(1.0)

    def test_top_level_evaluate_list_target(self):
        """Existing behavior with explicit list still works."""
        from corpusgen import evaluate
        report = evaluate(
            ["Hello world."],
            language="en-us",
            target_phonemes=["p", "b", "t"],
        )
        assert isinstance(report, EvaluationReport)
