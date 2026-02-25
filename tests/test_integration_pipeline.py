"""Integration tests: real PHOIBLE data + espeak mapping + evaluate().

These tests use the actual PHOIBLE CSV cached at ~/.corpusgen/phoible.csv.
They verify the full pipeline: espeak code → ISO 639-3 → PHOIBLE inventory
→ evaluate() with real target phonemes.

Marked with a custom marker so they can be skipped in CI where PHOIBLE
or espeak-ng may not be available.
"""

import pytest
from pathlib import Path

from corpusgen.evaluate import evaluate
from corpusgen.evaluate.report import EvaluationReport, Verbosity
from corpusgen.inventory import PhoibleDataset, EspeakMapping, Inventory


# Skip entire module if PHOIBLE CSV not cached
PHOIBLE_PATH = Path.home() / ".corpusgen" / "phoible.csv"
pytestmark = pytest.mark.skipif(
    not PHOIBLE_PATH.is_file(),
    reason="PHOIBLE CSV not cached — run PhoibleDataset().download() first",
)


@pytest.fixture(scope="module")
def phoible() -> PhoibleDataset:
    """Module-scoped: load PHOIBLE once for all tests."""
    ds = PhoibleDataset()
    ds.load()
    return ds


@pytest.fixture(scope="module")
def mapping() -> EspeakMapping:
    return EspeakMapping()


# ---------------------------------------------------------------------------
# 1. PHOIBLE real data sanity checks
# ---------------------------------------------------------------------------


class TestPhoibleRealData:
    """Verify PHOIBLE loads correctly with real data."""

    def test_inventory_count(self, phoible):
        assert phoible.inventory_count == 3020

    def test_language_count(self, phoible):
        assert phoible.language_count >= 2000

    def test_segment_count(self, phoible):
        assert phoible.segment_count > 100000

    def test_english_has_multiple_inventories(self, phoible):
        invs = phoible.get_all_inventories("eng")
        assert len(invs) >= 9

    def test_english_default_reasonable_size(self, phoible):
        inv = phoible.get_inventory("eng")
        assert 30 <= inv.size <= 50

    def test_mandarin_has_tones(self, phoible):
        inv = phoible.get_inventory("cmn")
        assert inv.has_tones

    def test_korean_inventory(self, phoible):
        inv = phoible.get_inventory("kor")
        assert inv.consonant_count > 10
        assert inv.vowel_count > 5

    def test_arabic_inventory(self, phoible):
        inv = phoible.get_inventory("arb")
        assert inv.size > 20

    def test_hindi_inventory(self, phoible):
        inv = phoible.get_inventory("hin")
        assert inv.size > 30

    def test_search_english(self, phoible):
        results = phoible.search("English")
        assert any(r["iso639_3"] == "eng" for r in results)

    def test_available_languages_large(self, phoible):
        langs = phoible.available_languages()
        assert len(langs) >= 2000


# ---------------------------------------------------------------------------
# 2. Espeak → PHOIBLE pipeline
# ---------------------------------------------------------------------------

# Languages where both espeak and PHOIBLE have data
PIPELINE_LANGUAGES = [
    ("en-us", "eng", "English"),
    ("en-gb", "eng", "English"),
    ("fr-fr", "fra", "French"),
    ("de", "deu", "German"),
    ("es", "spa", "Spanish"),
    ("it", "ita", "Italian"),
    ("pt", "por", "Portuguese"),
    ("pt-br", "por", "Portuguese"),
    ("nl", "nld", "Dutch"),
    ("pl", "pol", "Polish"),
    ("ru", "rus", "Russian"),
    ("uk", "ukr", "Ukrainian"),
    ("hi", "hin", "Hindi"),
    ("bn", "ben", "Bengali"),
    ("ta", "tam", "Tamil"),
    ("te", "tel", "Telugu"),
    ("ko", "kor", "Korean"),
    ("ja", "jpn", "Japanese"),
    ("ar", "arb", "Arabic"),
    ("he", "heb", "Hebrew"),
    ("tr", "tur", "Turkish"),
    ("fi", "fin", "Finnish"),
    ("hu", "hun", "Hungarian"),
    ("sv", "swe", "Swedish"),
    ("da", "dan", "Danish"),
    ("el", "ell", "Greek"),
    ("ka", "kat", "Georgian"),
    ("th", "tha", "Thai"),
    ("vi", "vie", "Vietnamese"),
    ("id", "ind", "Indonesian"),
    ("sw", "swh", "Swahili"),
    ("cmn", "cmn", "Mandarin"),
]


@pytest.mark.parametrize(
    "espeak_code, iso_code, label",
    PIPELINE_LANGUAGES,
    ids=[t[2] + f" ({t[0]})" for t in PIPELINE_LANGUAGES],
)
class TestEspeakToPhoiblePipeline:
    """Test espeak code → ISO → PHOIBLE inventory pipeline."""

    def test_mapping_resolves(self, mapping, espeak_code, iso_code, label):
        assert mapping.to_iso(espeak_code) == iso_code

    def test_phoible_has_inventory(self, phoible, espeak_code, iso_code, label):
        inv = phoible.get_inventory(iso_code)
        assert inv.size > 0, f"No segments for {label} ({iso_code})"

    def test_inventory_has_consonants_and_vowels(self, phoible, espeak_code, iso_code, label):
        inv = phoible.get_inventory(iso_code)
        assert inv.consonant_count > 0, f"{label}: no consonants"
        assert inv.vowel_count > 0, f"{label}: no vowels"

    def test_features_populated(self, phoible, espeak_code, iso_code, label):
        inv = phoible.get_inventory(iso_code)
        for seg in inv.segments:
            assert len(seg.features) == 38, (
                f"{label}: segment {seg.phoneme!r} has {len(seg.features)} features"
            )


# ---------------------------------------------------------------------------
# 3. Full pipeline: espeak → PHOIBLE → evaluate()
# ---------------------------------------------------------------------------

FULL_PIPELINE_CASES = [
    ("en-us", "eng", [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "Pack my box with five dozen liquor jugs.",
    ], "English"),
    ("fr-fr", "fra", [
        "Bonjour le monde, comment allez-vous?",
        "Le chat mange le poisson dans la cuisine.",
        "Il fait beau aujourd'hui en France.",
    ], "French"),
    ("de", "deu", [
        "Guten Morgen, wie geht es Ihnen?",
        "Der Hund läuft schnell über die Straße.",
        "Die Blumen sind wunderschön im Garten.",
    ], "German"),
    ("ar", "arb", [
        "مرحبا بالعالم اليوم جميل",
        "الكتاب على الطاولة",
        "الطقس جميل في الربيع",
    ], "Arabic"),
    ("hi", "hin", [
        "नमस्ते दुनिया आज मौसम अच्छा है",
        "कुत्ता तेज दौड़ता है",
        "फूल बहुत सुंदर हैं",
    ], "Hindi"),
    ("ja", "jpn", [
        "おはようございます。",
        "今日は天気がいいですね。",
        "犬が走っています。",
    ], "Japanese"),
    ("ko", "kor", [
        "안녕하세요 세계",
        "오늘 날씨가 좋습니다",
        "개가 빨리 달립니다",
    ], "Korean"),
    ("tr", "tur", [
        "Günaydın dünya çok güzel.",
        "Köpek hızlı koşuyor.",
        "Çiçekler bahçede güzel.",
    ], "Turkish"),
    ("cmn", "cmn", [
        "你好世界今天天气很好。",
        "狗跑得很快。",
        "花在花园里很漂亮。",
    ], "Mandarin"),
    ("sw", "swh", [
        "Habari za asubuhi dunia ni nzuri.",
        "Mbwa anakimbia haraka sana.",
        "Maua ni mazuri bustanini.",
    ], "Swahili"),
]


@pytest.mark.parametrize(
    "espeak_code, iso_code, sentences, label",
    FULL_PIPELINE_CASES,
    ids=[t[3] for t in FULL_PIPELINE_CASES],
)
class TestFullPipeline:
    """End-to-end: espeak G2P + PHOIBLE inventory → evaluate()."""

    def test_evaluate_with_phoible_target(
        self, phoible, mapping, espeak_code, iso_code, sentences, label
    ):
        """evaluate() produces a valid report using PHOIBLE inventory as target."""
        inv = phoible.get_inventory(iso_code)
        report = evaluate(
            sentences,
            language=espeak_code,
            target_phonemes=inv.phonemes,
        )
        assert isinstance(report, EvaluationReport)
        assert report.total_sentences == len(sentences)
        assert 0.0 <= report.coverage <= 1.0

    def test_coverage_is_partial(
        self, phoible, mapping, espeak_code, iso_code, sentences, label
    ):
        """With a full PHOIBLE inventory as target, a small corpus
        should have partial coverage (not 0%, unlikely 100%)."""
        inv = phoible.get_inventory(iso_code)
        report = evaluate(
            sentences,
            language=espeak_code,
            target_phonemes=inv.phonemes,
        )
        # Should cover at least something
        assert report.coverage > 0.0, (
            f"{label}: 0% coverage — G2P may have produced no matching phonemes"
        )

    def test_missing_phonemes_exist(
        self, phoible, mapping, espeak_code, iso_code, sentences, label
    ):
        """A small corpus likely won't cover the full inventory."""
        inv = phoible.get_inventory(iso_code)
        report = evaluate(
            sentences,
            language=espeak_code,
            target_phonemes=inv.phonemes,
        )
        # With only 3 sentences, there should be some missing
        # (unless it's a very small inventory)
        if inv.size > 10:
            assert len(report.missing_phonemes) > 0, (
                f"{label}: full coverage with 3 sentences is suspicious"
            )

    def test_invariants_hold(
        self, phoible, mapping, espeak_code, iso_code, sentences, label
    ):
        """Core invariants must hold with real data."""
        inv = phoible.get_inventory(iso_code)
        report = evaluate(
            sentences,
            language=espeak_code,
            target_phonemes=inv.phonemes,
        )
        # covered + missing = target
        assert report.covered_phonemes | report.missing_phonemes == set(inv.phonemes)
        assert report.covered_phonemes & report.missing_phonemes == set()
        # Coverage ratio
        if len(inv.phonemes) > 0:
            expected = len(report.covered_phonemes) / len(inv.phonemes)
            assert report.coverage == pytest.approx(expected)

    def test_report_renders_all_verbosities(
        self, phoible, mapping, espeak_code, iso_code, sentences, label
    ):
        """Report rendering should work at all verbosity levels."""
        inv = phoible.get_inventory(iso_code)
        report = evaluate(
            sentences,
            language=espeak_code,
            target_phonemes=inv.phonemes,
        )
        for v in Verbosity:
            text = report.render(verbosity=v)
            assert len(text) > 0, f"{label}: empty render at {v}"

    def test_report_exports(
        self, phoible, mapping, espeak_code, iso_code, sentences, label
    ):
        """JSON and JSON-LD export work with real data."""
        import json
        inv = phoible.get_inventory(iso_code)
        report = evaluate(
            sentences,
            language=espeak_code,
            target_phonemes=inv.phonemes,
        )
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["language"] == espeak_code

        doc = report.to_jsonld_ex()
        assert "@context" in doc

    def test_union_inventory_pipeline(
        self, phoible, mapping, espeak_code, iso_code, sentences, label
    ):
        """Pipeline also works with union inventory."""
        union = phoible.get_union_inventory(iso_code)
        report = evaluate(
            sentences,
            language=espeak_code,
            target_phonemes=union.phonemes,
        )
        assert isinstance(report, EvaluationReport)
        assert report.coverage > 0.0
