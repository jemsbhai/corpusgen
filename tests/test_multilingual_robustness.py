"""Multilingual robustness tests for evaluate().

Verifies that the G2P → CoverageTracker → EvaluationReport pipeline
produces structurally valid output across typologically diverse languages.

These tests do NOT assert specific phoneme values (those are espeak-ng's
responsibility). They assert invariants that must hold for ANY language.
"""

import pytest

from corpusgen.evaluate import evaluate
from corpusgen.evaluate.report import EvaluationReport


# ---------------------------------------------------------------------------
# Language test data: (language_code, sentences, label)
#
# Selection criteria:
#   - Typological diversity (isolating, agglutinative, fusional, polysynthetic)
#   - Script diversity (Latin, Cyrillic, Arabic, Devanagari, CJK, Georgian, etc.)
#   - Phonological diversity (tonal, non-tonal, click, rich consonant clusters)
#   - Geographic spread
# ---------------------------------------------------------------------------

LANGUAGE_TEST_CASES = [
    # --- Indo-European ---
    ("en-us", ["The quick brown fox jumps over the lazy dog."], "English (US)"),
    ("en-gb", ["The quick brown fox jumps over the lazy dog."], "English (GB)"),
    ("fr-fr", ["Bonjour le monde, comment allez-vous aujourd'hui?"], "French"),
    ("de", ["Guten Morgen, wie geht es Ihnen heute?"], "German"),
    ("es", ["Buenos días, el mundo es hermoso."], "Spanish"),
    ("pt", ["Bom dia, o mundo é maravilhoso."], "Portuguese (EU)"),
    ("pt-br", ["Bom dia, o mundo é maravilhoso."], "Portuguese (BR)"),
    ("it", ["Buongiorno, il mondo è bellissimo."], "Italian"),
    ("nl", ["Goedemorgen, de wereld is prachtig."], "Dutch"),
    ("pl", ["Dzień dobry, świat jest piękny."], "Polish"),
    ("ru", ["Доброе утро, мир прекрасен."], "Russian"),
    ("uk", ["Доброго ранку, світ прекрасний."], "Ukrainian"),
    ("hi", ["नमस्ते दुनिया, आज मौसम अच्छा है।"], "Hindi"),
    ("bn", ["শুভ সকাল, পৃথিবী সুন্দর।"], "Bengali"),
    ("el", ["Καλημέρα κόσμε, πώς είσαι σήμερα;"], "Greek"),
    ("sv", ["God morgon, världen är vacker."], "Swedish"),
    ("da", ["Godmorgen, verden er smuk."], "Danish"),
    ("nb", ["God morgen, verden er vakker."], "Norwegian Bokmål"),
    ("fa", ["سلام دنیا، امروز هوا خوب است."], "Persian"),
    ("gu", ["નમસ્તે દુનિયા, આજે હવામાન સારું છે."], "Gujarati"),

    # --- Uralic ---
    ("fi", ["Hyvää huomenta, maailma on kaunis."], "Finnish"),
    ("hu", ["Jó reggelt, a világ gyönyörű."], "Hungarian"),
    ("et", ["Tere hommikust, maailm on ilus."], "Estonian"),

    # --- Turkic ---
    ("tr", ["Günaydın, dünya çok güzel."], "Turkish"),

    # --- Semitic ---
    ("ar", ["مرحبا بالعالم اليوم جميل"], "Arabic"),
    ("he", ["שלום עולם, היום יפה מאוד."], "Hebrew"),

    # --- Sino-Tibetan ---
    ("cmn", ["你好世界今天天气很好。"], "Mandarin Chinese"),
    ("my", ["မင်္ဂလာပါ ကမ္ဘာကြီး။"], "Myanmar (Burmese)"),

    # --- Tai-Kadai ---
    ("th", ["สวัสดีชาวโลก วันนี้อากาศดี"], "Thai"),

    # --- Austronesian ---
    ("ms", ["Selamat pagi, dunia indah sekali."], "Malay"),
    ("id", ["Selamat pagi, dunia sangat indah."], "Indonesian"),

    # --- Dravidian ---
    ("ta", ["வணக்கம் உலகம், இன்று வானிலை நன்றாக உள்ளது."], "Tamil"),
    ("te", ["నమస్కారం ప్రపంచం, ఈరోజు వాతావరణం బాగుంది."], "Telugu"),
    ("kn", ["ನಮಸ್ಕಾರ ಪ್ರಪಂಚ, ಇಂದು ಹವಾಮಾನ ಚೆನ್ನಾಗಿದೆ."], "Kannada"),

    # --- Kartvelian ---
    ("ka", ["გამარჯობა მსოფლიო, დღეს ამინდი კარგია."], "Georgian"),

    # --- Niger-Congo ---
    ("sw", ["Habari za asubuhi, dunia ni nzuri sana."], "Swahili"),

    # --- Japonic ---
    ("ja", ["おはようございます、今日は良い天気です。"], "Japanese"),

    # --- Korean ---
    ("ko", ["안녕하세요 세계, 오늘 날씨가 좋습니다."], "Korean"),

    # --- Constructed ---
    ("eo", ["Bonan matenon, la mondo estas bela."], "Esperanto"),

    # --- Vietnamese (tonal, Latin script with diacritics) ---
    ("vi", ["Xin chào thế giới, hôm nay thời tiết đẹp."], "Vietnamese"),
]


# ---------------------------------------------------------------------------
# Parametrized structural invariant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lang, sentences, label",
    LANGUAGE_TEST_CASES,
    ids=[tc[2] for tc in LANGUAGE_TEST_CASES],
)
class TestMultilingualInvariants:
    """Structural invariants that must hold for any language."""

    def test_returns_report(self, lang, sentences, label):
        """evaluate() returns an EvaluationReport for every supported language."""
        report = evaluate(sentences, language=lang, target_phonemes=None)
        assert isinstance(report, EvaluationReport), f"Failed for {label}"

    def test_language_preserved(self, lang, sentences, label):
        report = evaluate(sentences, language=lang, target_phonemes=None)
        assert report.language == lang

    def test_produces_phonemes(self, lang, sentences, label):
        """G2P should produce at least some phonemes for real text."""
        report = evaluate(sentences, language=lang, target_phonemes=None)
        assert len(report.target_phonemes) > 0, (
            f"{label}: no phonemes produced — G2P may have failed silently"
        )

    def test_full_coverage_with_derived_inventory(self, lang, sentences, label):
        """With target=None (derived), coverage must be exactly 1.0."""
        report = evaluate(sentences, language=lang, target_phonemes=None)
        assert report.coverage == pytest.approx(1.0), (
            f"{label}: derived inventory should give 100% coverage"
        )

    def test_no_missing_with_derived_inventory(self, lang, sentences, label):
        report = evaluate(sentences, language=lang, target_phonemes=None)
        assert report.missing_phonemes == set(), (
            f"{label}: derived inventory should have no missing phonemes"
        )

    def test_covered_equals_target(self, lang, sentences, label):
        report = evaluate(sentences, language=lang, target_phonemes=None)
        assert report.covered_phonemes == set(report.target_phonemes)

    def test_sentence_details_count(self, lang, sentences, label):
        report = evaluate(sentences, language=lang, target_phonemes=None)
        assert len(report.sentence_details) == len(sentences)

    def test_sentence_details_have_phonemes(self, lang, sentences, label):
        """Every non-empty sentence should produce at least one phoneme."""
        report = evaluate(sentences, language=lang, target_phonemes=None)
        for sd in report.sentence_details:
            assert sd.phoneme_count > 0, (
                f"{label}: sentence '{sd.text[:30]}...' produced 0 phonemes"
            )

    def test_phoneme_counts_all_positive(self, lang, sentences, label):
        report = evaluate(sentences, language=lang, target_phonemes=None)
        for ph, count in report.phoneme_counts.items():
            assert count > 0, f"{label}: phoneme {ph!r} has count 0"

    def test_sources_valid(self, lang, sentences, label):
        report = evaluate(sentences, language=lang, target_phonemes=None)
        for ph, indices in report.phoneme_sources.items():
            for idx in indices:
                assert 0 <= idx < len(sentences), (
                    f"{label}: phoneme {ph!r} has invalid source index {idx}"
                )

    def test_new_phonemes_no_cross_sentence_duplicates(self, lang, sentences, label):
        """A phoneme should be 'new' in at most one sentence."""
        report = evaluate(sentences, language=lang, target_phonemes=None)
        all_new = []
        for sd in report.sentence_details:
            all_new.extend(sd.new_phonemes)
        assert len(all_new) == len(set(all_new)), (
            f"{label}: duplicate new_phonemes across sentences"
        )

    def test_union_new_equals_covered(self, lang, sentences, label):
        report = evaluate(sentences, language=lang, target_phonemes=None)
        all_new = set()
        for sd in report.sentence_details:
            all_new.update(sd.new_phonemes)
        assert all_new == report.covered_phonemes

    def test_report_renders(self, lang, sentences, label):
        """Report rendering should not crash for any language."""
        report = evaluate(sentences, language=lang, target_phonemes=None)
        text = report.render()
        assert len(text) > 0

    def test_report_exports_json(self, lang, sentences, label):
        """JSON export should not crash for any language."""
        import json
        report = evaluate(sentences, language=lang, target_phonemes=None)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["language"] == lang

    def test_deterministic(self, lang, sentences, label):
        """Same input must produce identical output."""
        r1 = evaluate(sentences, language=lang, target_phonemes=None)
        r2 = evaluate(sentences, language=lang, target_phonemes=None)
        assert r1.coverage == r2.coverage
        assert r1.covered_phonemes == r2.covered_phonemes
        assert r1.phoneme_counts == r2.phoneme_counts


# ---------------------------------------------------------------------------
# Multi-sentence per language (tests sentence interaction)
# ---------------------------------------------------------------------------

MULTI_SENTENCE_CASES = [
    ("en-us", [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "Pack my box with five dozen liquor jugs.",
    ], "English multi-sentence"),
    ("fr-fr", [
        "Bonjour le monde.",
        "Le chat mange le poisson.",
        "Il fait beau aujourd'hui.",
    ], "French multi-sentence"),
    ("de", [
        "Guten Morgen.",
        "Der Hund läuft schnell.",
        "Die Blumen sind schön.",
    ], "German multi-sentence"),
    ("ru", [
        "Доброе утро.",
        "Собака бежит быстро.",
        "Цветы красивые.",
    ], "Russian multi-sentence"),
    ("hi", [
        "नमस्ते दुनिया।",
        "आज मौसम बहुत अच्छा है।",
        "कुत्ता तेज दौड़ता है।",
    ], "Hindi multi-sentence"),
    ("ja", [
        "おはようございます。",
        "今日は天気がいいですね。",
        "犬が走っています。",
    ], "Japanese multi-sentence"),
    ("ar", [
        "مرحبا بالعالم",
        "اليوم جميل جدا",
        "الكلب يركض بسرعة",
    ], "Arabic multi-sentence"),
]


@pytest.mark.parametrize(
    "lang, sentences, label",
    MULTI_SENTENCE_CASES,
    ids=[tc[2] for tc in MULTI_SENTENCE_CASES],
)
class TestMultiSentenceInteraction:
    """Tests that sentence-level tracking works correctly across languages."""

    def test_new_phonemes_appear_in_first_occurrence_only(self, lang, sentences, label):
        """Each phoneme's 'new' attribution should match the earliest sentence
        where it actually appears."""
        report = evaluate(sentences, language=lang, target_phonemes=None)

        # Build expected first-occurrence map from sentence details
        first_seen: dict[str, int] = {}
        for sd in report.sentence_details:
            for ph in sd.all_phonemes:
                if ph not in first_seen:
                    first_seen[ph] = sd.index

        # Check new_phonemes aligns with first_seen
        for sd in report.sentence_details:
            for ph in sd.new_phonemes:
                assert first_seen[ph] == sd.index, (
                    f"{label}: phoneme {ph!r} marked new in sentence {sd.index} "
                    f"but first appeared in sentence {first_seen[ph]}"
                )

    def test_later_sentences_may_have_fewer_new(self, lang, sentences, label):
        """The first sentence should generally have the most new phonemes."""
        report = evaluate(sentences, language=lang, target_phonemes=None)
        if len(report.sentence_details) >= 2:
            first_new = len(report.sentence_details[0].new_phonemes)
            # First sentence should have at least some new phonemes
            assert first_new > 0, (
                f"{label}: first sentence contributed no new phonemes"
            )

    def test_phoneme_count_sum_consistency(self, lang, sentences, label):
        """Sum of per-phoneme counts must equal total target-phoneme occurrences
        across all sentence details."""
        report = evaluate(sentences, language=lang, target_phonemes=None)
        target_set = set(report.target_phonemes)

        total_from_counts = sum(report.phoneme_counts.values())

        total_from_details = 0
        for sd in report.sentence_details:
            for ph in sd.all_phonemes:
                if ph in target_set:
                    total_from_details += 1

        assert total_from_counts == total_from_details, (
            f"{label}: count mismatch — "
            f"phoneme_counts sum={total_from_counts}, "
            f"sentence details sum={total_from_details}"
        )

    def test_sources_consistent_with_details(self, lang, sentences, label):
        """If phoneme P lists sentence S as a source, P must appear in S's all_phonemes."""
        report = evaluate(sentences, language=lang, target_phonemes=None)
        for ph, indices in report.phoneme_sources.items():
            for idx in indices:
                sd = report.sentence_details[idx]
                assert ph in sd.all_phonemes, (
                    f"{label}: phoneme {ph!r} claims source {idx} "
                    f"but not in sentence's all_phonemes"
                )
