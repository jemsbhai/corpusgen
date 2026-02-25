"""Tests for PHOIBLE dataset loader, cache, and query API.

TDD RED phase. Uses a small fixture CSV that mirrors PHOIBLE's exact
49-column structure. No network access required.
"""

import csv
import json
import os
import textwrap
from pathlib import Path

import pytest

from corpusgen.inventory.models import Segment, Inventory, FEATURE_NAMES
from corpusgen.inventory.phoible import PhoibleDataset


# ---------------------------------------------------------------------------
# Fixture CSV — miniature PHOIBLE with known data
# ---------------------------------------------------------------------------

# 3 languages, 5 inventories, mix of consonants/vowels/tones/marginals/allophones
FIXTURE_HEADER = (
    "InventoryID,Glottocode,ISO6393,LanguageName,SpecificDialect,"
    "GlyphID,Phoneme,Allophones,Marginal,SegmentClass,Source,"
    + ",".join(FEATURE_NAMES)
)

# Feature defaults: all "0"
_Z = ",".join(["0"] * 38)


def _row(inv_id, glotto, iso, name, dialect, glyph, phoneme, allophones,
         marginal, seg_class, source, **feat_overrides):
    """Build a CSV row string with feature overrides."""
    feats = {f: "0" for f in FEATURE_NAMES}
    feats.update(feat_overrides)
    feat_str = ",".join(feats[f] for f in FEATURE_NAMES)
    dialect_str = dialect if dialect else "NA"
    allo_str = allophones if allophones else "NA"
    marginal_str = "TRUE" if marginal else "FALSE"
    return (
        f"{inv_id},{glotto},{iso},{name},{dialect_str},"
        f"{glyph},{phoneme},{allo_str},{marginal_str},{seg_class},{source},"
        f"{feat_str}"
    )


FIXTURE_ROWS = [
    # --- Korean (inv 1, source=spa) ---
    _row(1, "kore1280", "kor", "Korean", None, "0068", "h", "ç h ɦ",
         False, "consonant", "spa", consonantal="-", continuant="+"),
    _row(1, "kore1280", "kor", "Korean", None, "006B", "k", "k̚ ɡ k",
         False, "consonant", "spa", consonantal="+"),
    _row(1, "kore1280", "kor", "Korean", None, "0061", "a", "a",
         False, "vowel", "spa", syllabic="+", low="+"),
    _row(1, "kore1280", "kor", "Korean", None, "0069", "i", "i",
         False, "vowel", "spa", syllabic="+", high="+", front="+"),

    # --- English inv 160, source=spa (General) ---
    _row(160, "stan1293", "eng", "English", None, "0070", "p", "p pʰ",
         False, "consonant", "spa", consonantal="+", labial="+"),
    _row(160, "stan1293", "eng", "English", None, "0074", "t", "t tʰ",
         False, "consonant", "spa", consonantal="+", coronal="+"),
    _row(160, "stan1293", "eng", "English", None, "0073", "s", "s",
         False, "consonant", "spa", consonantal="+", coronal="+",
         continuant="+", strident="+"),
    _row(160, "stan1293", "eng", "English", None, "0292", "ʒ", "ʒ",
         True, "consonant", "spa", consonantal="+", continuant="+"),
    _row(160, "stan1293", "eng", "English", None, "0061", "æ", "æ",
         False, "vowel", "spa", syllabic="+", front="+", low="+"),
    _row(160, "stan1293", "eng", "English", None, "0259", "ə", "ə",
         False, "vowel", "spa", syllabic="+"),

    # --- English inv 2175, source=ph (Western US dialect) ---
    _row(2175, "stan1293", "eng", "English",
         "Western and Mid-Western US", "0070", "p", "p pʰ",
         False, "consonant", "ph", consonantal="+", labial="+"),
    _row(2175, "stan1293", "eng", "English",
         "Western and Mid-Western US", "0074", "t", "t tʰ ɾ",
         False, "consonant", "ph", consonantal="+", coronal="+"),
    _row(2175, "stan1293", "eng", "English",
         "Western and Mid-Western US", "0073", "s", "s",
         False, "consonant", "ph", consonantal="+", coronal="+",
         continuant="+", strident="+"),
    _row(2175, "stan1293", "eng", "English",
         "Western and Mid-Western US", "00E6", "æ", "æ",
         False, "vowel", "ph", syllabic="+", front="+", low="+"),
    _row(2175, "stan1293", "eng", "English",
         "Western and Mid-Western US", "0259", "ə", "ə ʌ",
         False, "vowel", "ph", syllabic="+"),

    # --- Mandarin (inv 2000, source=spa) — with tones ---
    _row(2000, "mand1415", "cmn", "Mandarin Chinese", None,
         "0070", "p", "p",
         False, "consonant", "spa", consonantal="+", labial="+"),
    _row(2000, "mand1415", "cmn", "Mandarin Chinese", None,
         "0074", "t", "t",
         False, "consonant", "spa", consonantal="+", coronal="+"),
    _row(2000, "mand1415", "cmn", "Mandarin Chinese", None,
         "0061", "a", "a",
         False, "vowel", "spa", syllabic="+", low="+"),
    _row(2000, "mand1415", "cmn", "Mandarin Chinese", None,
         "0069", "i", "i",
         False, "vowel", "spa", syllabic="+", high="+", front="+"),
    _row(2000, "mand1415", "cmn", "Mandarin Chinese", None,
         "02E5", "˥", "NA",
         False, "tone", "spa", tone="+"),
    _row(2000, "mand1415", "cmn", "Mandarin Chinese", None,
         "02E5+02E9", "˥˩", "NA",
         False, "tone", "spa", tone="+"),

    # --- Mandarin (inv 2001, source=ph) — second inventory, fewer segments ---
    _row(2001, "mand1415", "cmn", "Mandarin Chinese", "Beijing",
         "0070", "p", "p",
         False, "consonant", "ph", consonantal="+", labial="+"),
    _row(2001, "mand1415", "cmn", "Mandarin Chinese", "Beijing",
         "0061", "a", "a",
         False, "vowel", "ph", syllabic="+", low="+"),
    _row(2001, "mand1415", "cmn", "Mandarin Chinese", "Beijing",
         "02E5", "˥", "NA",
         False, "tone", "ph", tone="+"),
]

FIXTURE_CSV = FIXTURE_HEADER + "\n" + "\n".join(FIXTURE_ROWS) + "\n"


@pytest.fixture
def fixture_csv_path(tmp_path) -> Path:
    """Write the fixture CSV to a temp file and return its path."""
    csv_path = tmp_path / "phoible.csv"
    csv_path.write_text(FIXTURE_CSV, encoding="utf-8")
    return csv_path


@pytest.fixture
def dataset(fixture_csv_path) -> PhoibleDataset:
    """A PhoibleDataset loaded from the fixture CSV."""
    ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
    ds.load()
    return ds


# ---------------------------------------------------------------------------
# 1. Construction and loading
# ---------------------------------------------------------------------------


class TestPhoibleDatasetLoading:
    """Tests for dataset construction and CSV parsing."""

    def test_construction(self, fixture_csv_path):
        ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
        assert ds is not None

    def test_load_succeeds(self, fixture_csv_path):
        ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
        ds.load()

    def test_is_loaded_property(self, fixture_csv_path):
        ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
        assert ds.is_loaded is False
        ds.load()
        assert ds.is_loaded is True

    def test_load_parses_correct_inventory_count(self, dataset):
        """Fixture has 5 inventories: 1, 160, 2175, 2000, 2001."""
        assert dataset.inventory_count == 5

    def test_load_parses_correct_language_count(self, dataset):
        """Fixture has 3 languages: kor, eng, cmn."""
        assert dataset.language_count == 3

    def test_load_parses_correct_segment_count(self, dataset):
        """Fixture has 24 total segment rows."""
        assert dataset.segment_count == 24

    def test_load_idempotent(self, fixture_csv_path):
        """Calling load() twice should not duplicate data."""
        ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
        ds.load()
        count1 = ds.inventory_count
        ds.load()
        count2 = ds.inventory_count
        assert count1 == count2

    def test_missing_csv_raises(self, tmp_path):
        """If CSV doesn't exist and no download, raise FileNotFoundError."""
        ds = PhoibleDataset(cache_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            ds.load()


# ---------------------------------------------------------------------------
# 2. Query by ISO 639-3
# ---------------------------------------------------------------------------


class TestQueryByISO:
    """Tests for looking up inventories by ISO 639-3 code."""

    def test_get_inventory_by_iso(self, dataset):
        inv = dataset.get_inventory("kor")
        assert isinstance(inv, Inventory)
        assert inv.iso639_3 == "kor"
        assert inv.language_name == "Korean"

    def test_get_inventory_english_default(self, dataset):
        """With multiple inventories, default returns the largest."""
        inv = dataset.get_inventory("eng")
        assert inv.iso639_3 == "eng"
        # inv 160 has 6 segments, inv 2175 has 5 → should return 160
        assert inv.inventory_id == 160
        assert inv.size == 6

    def test_get_inventory_with_source(self, dataset):
        """Can request a specific source."""
        inv = dataset.get_inventory("eng", source="ph")
        assert inv.source == "ph"
        assert inv.inventory_id == 2175

    def test_get_inventory_unknown_iso_raises(self, dataset):
        with pytest.raises(KeyError, match="xyz"):
            dataset.get_inventory("xyz")

    def test_get_inventory_unknown_source_raises(self, dataset):
        with pytest.raises(KeyError, match="upsid"):
            dataset.get_inventory("eng", source="upsid")

    def test_get_all_inventories_by_iso(self, dataset):
        invs = dataset.get_all_inventories("eng")
        assert isinstance(invs, list)
        assert len(invs) == 2
        assert all(isinstance(i, Inventory) for i in invs)
        ids = {i.inventory_id for i in invs}
        assert ids == {160, 2175}

    def test_get_all_inventories_single(self, dataset):
        invs = dataset.get_all_inventories("kor")
        assert len(invs) == 1

    def test_get_all_inventories_unknown_raises(self, dataset):
        with pytest.raises(KeyError, match="xyz"):
            dataset.get_all_inventories("xyz")


# ---------------------------------------------------------------------------
# 3. Query by Glottocode
# ---------------------------------------------------------------------------


class TestQueryByGlottocode:
    """Tests for looking up inventories by Glottocode."""

    def test_get_inventory_by_glottocode(self, dataset):
        inv = dataset.get_inventory("kore1280")
        assert inv.glottocode == "kore1280"
        assert inv.language_name == "Korean"

    def test_get_all_inventories_by_glottocode(self, dataset):
        invs = dataset.get_all_inventories("mand1415")
        assert len(invs) == 2

    def test_glottocode_and_iso_give_same_result(self, dataset):
        by_iso = dataset.get_inventory("kor")
        by_glotto = dataset.get_inventory("kore1280")
        assert by_iso.inventory_id == by_glotto.inventory_id


# ---------------------------------------------------------------------------
# 4. Union inventory
# ---------------------------------------------------------------------------


class TestUnionInventory:
    """Tests for merging all inventories into a maximally inclusive set."""

    def test_union_returns_inventory(self, dataset):
        inv = dataset.get_union_inventory("eng")
        assert isinstance(inv, Inventory)

    def test_union_has_superset_of_phonemes(self, dataset):
        """Union phonemes should be a superset of each individual inventory."""
        union = dataset.get_union_inventory("eng")
        union_set = set(union.phonemes)

        for inv in dataset.get_all_inventories("eng"):
            assert set(inv.phonemes) <= union_set, (
                f"Inventory {inv.inventory_id} has phonemes not in union"
            )

    def test_union_no_duplicate_phonemes(self, dataset):
        union = dataset.get_union_inventory("eng")
        assert len(union.phonemes) == len(set(union.phonemes))

    def test_union_preserves_metadata(self, dataset):
        union = dataset.get_union_inventory("eng")
        assert union.iso639_3 == "eng"
        assert union.language_name == "English"
        assert union.source == "union"

    def test_union_single_inventory_language(self, dataset):
        """For a language with one inventory, union = that inventory."""
        union = dataset.get_union_inventory("kor")
        single = dataset.get_inventory("kor")
        assert set(union.phonemes) == set(single.phonemes)

    def test_union_unknown_raises(self, dataset):
        with pytest.raises(KeyError):
            dataset.get_union_inventory("xyz")

    def test_union_english_includes_marginal(self, dataset):
        """Union should include marginal phonemes from any source."""
        union = dataset.get_union_inventory("eng")
        # ʒ is marginal in inv 160
        assert "ʒ" in union.phonemes

    def test_union_mandarin_includes_tones(self, dataset):
        """Union should include tones from all sources."""
        union = dataset.get_union_inventory("cmn")
        assert "˥" in union.phonemes
        assert "˥˩" in union.phonemes


# ---------------------------------------------------------------------------
# 5. Parsed segment fidelity
# ---------------------------------------------------------------------------


class TestSegmentParsing:
    """Tests that CSV data is parsed into Segment objects correctly."""

    def test_phoneme_parsed(self, dataset):
        inv = dataset.get_inventory("kor")
        assert "h" in inv.phonemes
        assert "k" in inv.phonemes
        assert "a" in inv.phonemes
        assert "i" in inv.phonemes

    def test_segment_class_parsed(self, dataset):
        inv = dataset.get_inventory("kor")
        assert inv.consonant_count == 2
        assert inv.vowel_count == 2

    def test_segment_class_tone(self, dataset):
        inv = dataset.get_inventory("cmn")
        assert inv.tone_count == 2

    def test_allophones_parsed(self, dataset):
        inv = dataset.get_inventory("kor")
        allo = inv.all_allophones
        assert allo["h"] == ["ç", "h", "ɦ"]
        assert allo["k"] == ["k̚", "ɡ", "k"]

    def test_allophones_na_parsed_as_empty(self, dataset):
        """PHOIBLE 'NA' allophones should become empty list."""
        inv = dataset.get_inventory("cmn")
        # Tones have NA allophones
        tone_seg = [s for s in inv.segments if s.phoneme == "˥"][0]
        assert tone_seg.allophones == []

    def test_marginal_parsed(self, dataset):
        inv = dataset.get_inventory("eng")
        zh_seg = [s for s in inv.segments if s.phoneme == "ʒ"][0]
        assert zh_seg.marginal is True
        p_seg = [s for s in inv.segments if s.phoneme == "p"][0]
        assert p_seg.marginal is False

    def test_features_parsed(self, dataset):
        inv = dataset.get_inventory("eng")
        p_seg = [s for s in inv.segments if s.phoneme == "p"][0]
        assert p_seg.features["consonantal"] == "+"
        assert p_seg.features["labial"] == "+"

    def test_features_count(self, dataset):
        inv = dataset.get_inventory("kor")
        for seg in inv.segments:
            assert len(seg.features) == 38

    def test_glyph_id_parsed(self, dataset):
        inv = dataset.get_inventory("kor")
        h_seg = [s for s in inv.segments if s.phoneme == "h"][0]
        assert h_seg.glyph_id == "0068"

    def test_dialect_parsed(self, dataset):
        inv = dataset.get_inventory("eng", source="ph")
        assert inv.specific_dialect == "Western and Mid-Western US"

    def test_dialect_na_is_none(self, dataset):
        inv = dataset.get_inventory("kor")
        assert inv.specific_dialect is None

    def test_glottocode_parsed(self, dataset):
        inv = dataset.get_inventory("kor")
        assert inv.glottocode == "kore1280"


# ---------------------------------------------------------------------------
# 6. Search and listing
# ---------------------------------------------------------------------------


class TestSearchAndListing:
    """Tests for language search and listing."""

    def test_search_by_name_exact(self, dataset):
        results = dataset.search("Korean")
        assert len(results) >= 1
        assert any(r["iso639_3"] == "kor" for r in results)

    def test_search_by_name_case_insensitive(self, dataset):
        results = dataset.search("korean")
        assert len(results) >= 1

    def test_search_by_name_partial(self, dataset):
        results = dataset.search("Engl")
        assert any(r["iso639_3"] == "eng" for r in results)

    def test_search_no_results(self, dataset):
        results = dataset.search("Klingon")
        assert results == []

    def test_search_result_structure(self, dataset):
        results = dataset.search("Korean")
        r = results[0]
        assert "iso639_3" in r
        assert "glottocode" in r
        assert "language_name" in r
        assert "inventory_count" in r
        assert "sources" in r

    def test_available_languages(self, dataset):
        langs = dataset.available_languages()
        assert isinstance(langs, list)
        assert len(langs) == 3  # kor, eng, cmn

    def test_available_languages_structure(self, dataset):
        langs = dataset.available_languages()
        for lang in langs:
            assert "iso639_3" in lang
            assert "glottocode" in lang
            assert "language_name" in lang
            assert "inventory_count" in lang
            assert "sources" in lang

    def test_available_languages_sorted(self, dataset):
        langs = dataset.available_languages()
        names = [l["language_name"] for l in langs]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# 7. Inventory sources listing
# ---------------------------------------------------------------------------


class TestInventorySources:
    """Tests for listing available sources per language."""

    def test_sources_for_language(self, dataset):
        sources = dataset.sources_for("eng")
        assert isinstance(sources, list)
        assert set(sources) == {"spa", "ph"}

    def test_sources_for_single_inventory(self, dataset):
        sources = dataset.sources_for("kor")
        assert sources == ["spa"]

    def test_sources_for_unknown_raises(self, dataset):
        with pytest.raises(KeyError):
            dataset.sources_for("xyz")


# ---------------------------------------------------------------------------
# 8. Cache path management
# ---------------------------------------------------------------------------


class TestCacheManagement:
    """Tests for cache directory and file path handling."""

    def test_cache_dir_property(self, fixture_csv_path):
        ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
        assert ds.cache_dir == fixture_csv_path.parent

    def test_csv_path_property(self, fixture_csv_path):
        ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
        assert ds.csv_path == fixture_csv_path

    def test_default_cache_dir(self):
        """Default cache dir should be ~/.corpusgen/."""
        ds = PhoibleDataset()
        expected = Path.home() / ".corpusgen"
        assert ds.cache_dir == expected

    def test_csv_exists_property(self, fixture_csv_path):
        ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
        assert ds.csv_exists is True

    def test_csv_exists_false(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        ds = PhoibleDataset(cache_dir=empty_dir)
        assert ds.csv_exists is False


# ---------------------------------------------------------------------------
# 9. Auto-load on query
# ---------------------------------------------------------------------------


class TestAutoLoad:
    """Dataset should auto-load from cache when queried if not yet loaded."""

    def test_auto_load_on_get_inventory(self, fixture_csv_path):
        ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
        assert ds.is_loaded is False
        inv = ds.get_inventory("kor")
        assert ds.is_loaded is True
        assert inv.language_name == "Korean"

    def test_auto_load_on_search(self, fixture_csv_path):
        ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
        results = ds.search("Korean")
        assert ds.is_loaded is True
        assert len(results) >= 1

    def test_auto_load_on_available_languages(self, fixture_csv_path):
        ds = PhoibleDataset(cache_dir=fixture_csv_path.parent)
        langs = ds.available_languages()
        assert ds.is_loaded is True
