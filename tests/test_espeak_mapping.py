"""Tests for espeak ↔ ISO 639-3 language code mapping.

TDD RED phase. Tests the mapping layer that connects espeak-ng voice
codes to PHOIBLE ISO 639-3 identifiers.
"""

import pytest

from corpusgen.inventory.mapping import EspeakMapping


# ---------------------------------------------------------------------------
# 1. Mapping data integrity
# ---------------------------------------------------------------------------


class TestMappingData:
    """Tests that the bundled mapping data is well-formed."""

    def test_mapping_loads(self):
        m = EspeakMapping()
        assert m is not None

    def test_mapping_nonempty(self):
        m = EspeakMapping()
        assert m.size > 100  # We have ~131 espeak voices

    def test_all_values_are_3_letter_or_short_codes(self):
        """ISO 639-3 codes are 3-4 characters."""
        m = EspeakMapping()
        for espeak, iso in m.items():
            assert 2 <= len(iso) <= 4, (
                f"espeak {espeak!r} maps to {iso!r} which is not a valid ISO code"
            )

    def test_all_keys_are_lowercase(self):
        m = EspeakMapping()
        for key in m.espeak_codes():
            assert key == key.lower(), f"Key {key!r} is not lowercase"


# ---------------------------------------------------------------------------
# 2. Forward lookup: espeak → ISO 639-3
# ---------------------------------------------------------------------------


class TestEspeakToISO:
    """Tests for espeak code → ISO 639-3 lookup."""

    def test_english_us(self):
        m = EspeakMapping()
        assert m.to_iso("en-us") == "eng"

    def test_english_gb(self):
        m = EspeakMapping()
        assert m.to_iso("en-gb") == "eng"

    def test_french(self):
        m = EspeakMapping()
        assert m.to_iso("fr-fr") == "fra"

    def test_arabic(self):
        m = EspeakMapping()
        assert m.to_iso("ar") == "arb"

    def test_mandarin(self):
        m = EspeakMapping()
        assert m.to_iso("cmn") == "cmn"

    def test_korean(self):
        m = EspeakMapping()
        assert m.to_iso("ko") == "kor"

    def test_japanese(self):
        m = EspeakMapping()
        assert m.to_iso("ja") == "jpn"

    def test_hindi(self):
        m = EspeakMapping()
        assert m.to_iso("hi") == "hin"

    def test_german(self):
        m = EspeakMapping()
        assert m.to_iso("de") == "deu"

    def test_spanish(self):
        m = EspeakMapping()
        assert m.to_iso("es") == "spa"

    def test_portuguese_brazil(self):
        m = EspeakMapping()
        assert m.to_iso("pt-br") == "por"

    def test_case_insensitive(self):
        """Lookup should be case-insensitive."""
        m = EspeakMapping()
        assert m.to_iso("EN-US") == "eng"
        assert m.to_iso("Fr-Fr") == "fra"

    def test_unknown_raises(self):
        m = EspeakMapping()
        with pytest.raises(KeyError, match="xxx"):
            m.to_iso("xxx")


# ---------------------------------------------------------------------------
# 3. Reverse lookup: ISO 639-3 → espeak codes
# ---------------------------------------------------------------------------


class TestISOToEspeak:
    """Tests for ISO 639-3 → espeak code(s) lookup."""

    def test_english_returns_multiple(self):
        """English has many espeak variants."""
        m = EspeakMapping()
        codes = m.to_espeak("eng")
        assert isinstance(codes, list)
        assert "en-us" in codes
        assert "en-gb" in codes
        assert len(codes) >= 2

    def test_korean_returns_single(self):
        m = EspeakMapping()
        codes = m.to_espeak("kor")
        assert codes == ["ko"]

    def test_french_returns_multiple(self):
        m = EspeakMapping()
        codes = m.to_espeak("fra")
        assert "fr-fr" in codes

    def test_mandarin_returns_multiple(self):
        m = EspeakMapping()
        codes = m.to_espeak("cmn")
        assert "cmn" in codes
        assert "cmn-latn-pinyin" in codes

    def test_unknown_raises(self):
        m = EspeakMapping()
        with pytest.raises(KeyError, match="xxx"):
            m.to_espeak("xxx")

    def test_results_sorted(self):
        m = EspeakMapping()
        codes = m.to_espeak("eng")
        assert codes == sorted(codes)


# ---------------------------------------------------------------------------
# 4. Macrolanguage resolution
# ---------------------------------------------------------------------------


class TestMacrolanguageResolution:
    """Tests that macrolanguage ISO codes are resolved correctly."""

    def test_malay_macro_to_standard(self):
        """ms (Malay macro) → zsm (Standard Malay) in PHOIBLE."""
        m = EspeakMapping()
        assert m.to_iso("ms") == "zsm"

    def test_swahili_macro_to_individual(self):
        m = EspeakMapping()
        assert m.to_iso("sw") == "swh"

    def test_nepali_macro_to_individual(self):
        m = EspeakMapping()
        assert m.to_iso("ne") == "npi"

    def test_albanian_macro_to_tosk(self):
        m = EspeakMapping()
        assert m.to_iso("sq") == "als"

    def test_latvian_macro_to_standard(self):
        m = EspeakMapping()
        assert m.to_iso("lv") == "lvs"

    def test_uzbek_macro_to_northern(self):
        m = EspeakMapping()
        assert m.to_iso("uz") == "uzn"


# ---------------------------------------------------------------------------
# 5. Dialect variant grouping
# ---------------------------------------------------------------------------


class TestDialectGrouping:
    """Tests that dialect variants map to the same ISO code."""

    def test_english_dialects_same_iso(self):
        m = EspeakMapping()
        en_variants = ["en-us", "en-gb", "en-gb-scotland",
                        "en-gb-x-rp", "en-029"]
        for v in en_variants:
            assert m.to_iso(v) == "eng", f"{v} should map to eng"

    def test_french_dialects_same_iso(self):
        m = EspeakMapping()
        for v in ["fr-fr", "fr-be", "fr-ch"]:
            assert m.to_iso(v) == "fra", f"{v} should map to fra"

    def test_vietnamese_dialects_same_iso(self):
        m = EspeakMapping()
        for v in ["vi", "vi-vn-x-central", "vi-vn-x-south"]:
            assert m.to_iso(v) == "vie", f"{v} should map to vie"

    def test_spanish_dialects_same_iso(self):
        m = EspeakMapping()
        for v in ["es", "es-419"]:
            assert m.to_iso(v) == "spa", f"{v} should map to spa"

    def test_russian_dialects_same_iso(self):
        m = EspeakMapping()
        for v in ["ru", "ru-lv"]:
            assert m.to_iso(v) == "rus", f"{v} should map to rus"


# ---------------------------------------------------------------------------
# 6. Listing
# ---------------------------------------------------------------------------


class TestMappingListing:
    """Tests for enumeration methods."""

    def test_espeak_codes(self):
        m = EspeakMapping()
        codes = m.espeak_codes()
        assert isinstance(codes, list)
        assert "en-us" in codes
        assert "fr-fr" in codes

    def test_iso_codes(self):
        m = EspeakMapping()
        isos = m.iso_codes()
        assert isinstance(isos, list)
        assert "eng" in isos
        assert "fra" in isos

    def test_iso_codes_no_duplicates(self):
        m = EspeakMapping()
        isos = m.iso_codes()
        assert len(isos) == len(set(isos))

    def test_items(self):
        m = EspeakMapping()
        items = list(m.items())
        assert len(items) > 100
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in items)
