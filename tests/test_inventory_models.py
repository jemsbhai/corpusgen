"""Tests for inventory data models: Segment and Inventory.

TDD RED phase — these tests define the contract for the pure data
containers that represent PHOIBLE phonological segments and inventories.

No I/O or network access. All data is constructed inline.
"""

import pytest

from corpusgen.inventory.models import Segment, Inventory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "tone", "stress", "syllabic", "short", "long",
    "consonantal", "sonorant", "continuant", "delayedRelease",
    "approximant", "tap", "trill", "nasal", "lateral",
    "labial", "round", "labiodental", "coronal", "anterior",
    "distributed", "strident", "dorsal", "high", "low",
    "front", "back", "tense", "retractedTongueRoot",
    "advancedTongueRoot", "periodicGlottalSource",
    "epilaryngealSource", "spreadGlottis", "constrictedGlottis",
    "fortis", "lenis", "raisedLarynxEjective",
    "loweredLarynxImplosive", "click",
]


def _make_features(**overrides) -> dict[str, str]:
    """Build a full feature dict with defaults of '0', applying overrides."""
    features = {f: "0" for f in FEATURE_COLUMNS}
    features.update(overrides)
    return features


def _consonant(phoneme: str, **feat_overrides) -> Segment:
    """Shortcut to build a consonant Segment."""
    return Segment(
        phoneme=phoneme,
        segment_class="consonant",
        marginal=False,
        allophones=[],
        features=_make_features(consonantal="+", **feat_overrides),
        glyph_id="",
    )


def _vowel(phoneme: str, **feat_overrides) -> Segment:
    """Shortcut to build a vowel Segment."""
    return Segment(
        phoneme=phoneme,
        segment_class="vowel",
        marginal=False,
        allophones=[],
        features=_make_features(syllabic="+", **feat_overrides),
        glyph_id="",
    )


def _tone(phoneme: str) -> Segment:
    """Shortcut to build a tone Segment."""
    return Segment(
        phoneme=phoneme,
        segment_class="tone",
        marginal=False,
        allophones=[],
        features=_make_features(tone="+"),
        glyph_id="",
    )


@pytest.fixture
def sample_segments() -> list[Segment]:
    """A small English-like inventory."""
    return [
        _consonant("p", labial="+"),
        _consonant("b", labial="+", sonorant="-"),
        _consonant("t", coronal="+"),
        _consonant("d", coronal="+"),
        _consonant("k", dorsal="+"),
        _consonant("s", coronal="+", continuant="+", strident="+"),
        _consonant("z", coronal="+", continuant="+", strident="+"),
        _consonant("m", labial="+", nasal="+", sonorant="+"),
        _consonant("n", coronal="+", nasal="+", sonorant="+"),
        _vowel("iː", high="+", front="+", long="+"),
        _vowel("ɪ", high="+", front="+"),
        _vowel("ɛ", front="+", low="-"),
        _vowel("æ", front="+", low="+"),
        _vowel("ɑː", low="+", back="+", long="+"),
        _vowel("ə"),
    ]


@pytest.fixture
def sample_inventory(sample_segments) -> Inventory:
    """A minimal English-like inventory."""
    return Inventory(
        inventory_id=160,
        language_name="English",
        iso639_3="eng",
        glottocode="stan1293",
        specific_dialect=None,
        source="spa",
        segments=sample_segments,
    )


@pytest.fixture
def tonal_inventory() -> Inventory:
    """An inventory with tone segments (e.g., Mandarin-like)."""
    return Inventory(
        inventory_id=999,
        language_name="Mandarin",
        iso639_3="cmn",
        glottocode="mand1415",
        specific_dialect=None,
        source="spa",
        segments=[
            _consonant("p"),
            _consonant("t"),
            _consonant("k"),
            _vowel("a"),
            _vowel("i"),
            _vowel("u"),
            _tone("˥"),
            _tone("˧˥"),
            _tone("˨˩˦"),
            _tone("˥˩"),
        ],
    )


@pytest.fixture
def inventory_with_marginals() -> Inventory:
    """An inventory with some marginal phonemes."""
    return Inventory(
        inventory_id=500,
        language_name="TestLang",
        iso639_3="tst",
        glottocode="test1234",
        specific_dialect="Northern",
        source="ph",
        segments=[
            _consonant("p"),
            _consonant("t"),
            _consonant("k"),
            Segment(
                phoneme="ʒ",
                segment_class="consonant",
                marginal=True,
                allophones=["ʒ", "dʒ"],
                features=_make_features(consonantal="+", continuant="+"),
                glyph_id="0292",
            ),
            Segment(
                phoneme="x",
                segment_class="consonant",
                marginal=True,
                allophones=[],
                features=_make_features(consonantal="+", continuant="+", dorsal="+"),
                glyph_id="0078",
            ),
            _vowel("a"),
            _vowel("i"),
        ],
    )


# ---------------------------------------------------------------------------
# 1. Segment dataclass
# ---------------------------------------------------------------------------


class TestSegment:
    """Tests for the Segment data container."""

    def test_construction(self):
        s = _consonant("p")
        assert s.phoneme == "p"
        assert s.segment_class == "consonant"
        assert s.marginal is False

    def test_phoneme_field(self):
        s = _consonant("tʃ")
        assert s.phoneme == "tʃ"

    def test_segment_class_consonant(self):
        assert _consonant("p").segment_class == "consonant"

    def test_segment_class_vowel(self):
        assert _vowel("a").segment_class == "vowel"

    def test_segment_class_tone(self):
        assert _tone("˥").segment_class == "tone"

    def test_marginal_default_false(self):
        assert _consonant("p").marginal is False

    def test_marginal_true(self):
        s = Segment(
            phoneme="ʒ",
            segment_class="consonant",
            marginal=True,
            allophones=[],
            features=_make_features(),
            glyph_id="",
        )
        assert s.marginal is True

    def test_allophones_empty(self):
        assert _consonant("p").allophones == []

    def test_allophones_populated(self):
        s = Segment(
            phoneme="k",
            segment_class="consonant",
            marginal=False,
            allophones=["k̚", "ɡ", "k"],
            features=_make_features(),
            glyph_id="006B",
        )
        assert s.allophones == ["k̚", "ɡ", "k"]
        assert len(s.allophones) == 3

    def test_features_dict(self):
        s = _consonant("p", labial="+")
        assert isinstance(s.features, dict)
        assert s.features["labial"] == "+"
        assert s.features["consonantal"] == "+"

    def test_features_has_all_38_keys(self):
        """Every Segment should carry all 38 distinctive features."""
        s = _consonant("p")
        assert len(s.features) == 38
        for col in FEATURE_COLUMNS:
            assert col in s.features

    def test_features_values_valid(self):
        """Feature values must be '+', '-', or '0'."""
        s = _consonant("p", labial="+", sonorant="-")
        for val in s.features.values():
            assert val in ("+", "-", "0"), f"Invalid feature value: {val!r}"

    def test_glyph_id(self):
        s = Segment(
            phoneme="p",
            segment_class="consonant",
            marginal=False,
            allophones=[],
            features=_make_features(),
            glyph_id="0070",
        )
        assert s.glyph_id == "0070"

    def test_equality(self):
        """Two Segments with identical data should be equal."""
        s1 = _consonant("p", labial="+")
        s2 = _consonant("p", labial="+")
        assert s1 == s2

    def test_inequality_different_phoneme(self):
        s1 = _consonant("p")
        s2 = _consonant("b")
        assert s1 != s2

    def test_inequality_different_marginal(self):
        s1 = Segment("p", "consonant", False, [], _make_features(), "")
        s2 = Segment("p", "consonant", True, [], _make_features(), "")
        assert s1 != s2

    def test_repr_contains_phoneme(self):
        s = _consonant("p")
        r = repr(s)
        assert "p" in r

    def test_hashable(self):
        """Segments should be usable in sets/dicts for deduplication."""
        s = _consonant("p", labial="+")
        # frozen dataclass or __hash__ needed
        {s}  # Should not raise


# ---------------------------------------------------------------------------
# 2. Inventory dataclass — metadata fields
# ---------------------------------------------------------------------------


class TestInventoryMetadata:
    """Tests for Inventory metadata fields."""

    def test_construction(self, sample_inventory):
        assert sample_inventory.inventory_id == 160
        assert sample_inventory.language_name == "English"

    def test_iso639_3(self, sample_inventory):
        assert sample_inventory.iso639_3 == "eng"

    def test_glottocode(self, sample_inventory):
        assert sample_inventory.glottocode == "stan1293"

    def test_specific_dialect_none(self, sample_inventory):
        assert sample_inventory.specific_dialect is None

    def test_specific_dialect_set(self):
        inv = Inventory(
            inventory_id=2175,
            language_name="English",
            iso639_3="eng",
            glottocode="stan1293",
            specific_dialect="Western and Mid-Western US",
            source="ph",
            segments=[_consonant("p")],
        )
        assert inv.specific_dialect == "Western and Mid-Western US"

    def test_source(self, sample_inventory):
        assert sample_inventory.source == "spa"

    def test_segments_list(self, sample_inventory):
        assert isinstance(sample_inventory.segments, list)
        assert all(isinstance(s, Segment) for s in sample_inventory.segments)

    def test_segments_length(self, sample_segments, sample_inventory):
        assert len(sample_inventory.segments) == len(sample_segments)


# ---------------------------------------------------------------------------
# 3. Inventory computed properties — phoneme lists
# ---------------------------------------------------------------------------


class TestInventoryPhonemes:
    """Tests for computed phoneme list properties."""

    def test_phonemes_returns_all_ipa(self, sample_inventory):
        """phonemes property returns all IPA symbols as a list."""
        phonemes = sample_inventory.phonemes
        assert isinstance(phonemes, list)
        assert len(phonemes) == len(sample_inventory.segments)

    def test_phonemes_preserves_order(self, sample_inventory):
        """Phoneme order matches segment order."""
        expected = [s.phoneme for s in sample_inventory.segments]
        assert sample_inventory.phonemes == expected

    def test_phonemes_are_strings(self, sample_inventory):
        for p in sample_inventory.phonemes:
            assert isinstance(p, str)
            assert len(p) > 0

    def test_consonants(self, sample_inventory):
        """consonants property filters to consonant segments only."""
        cons = sample_inventory.consonants
        assert isinstance(cons, list)
        assert all(isinstance(c, str) for c in cons)
        # Our sample has p, b, t, d, k, s, z, m, n = 9 consonants
        assert len(cons) == 9

    def test_consonants_subset_of_phonemes(self, sample_inventory):
        assert set(sample_inventory.consonants) <= set(sample_inventory.phonemes)

    def test_vowels(self, sample_inventory):
        """vowels property filters to vowel segments only."""
        vows = sample_inventory.vowels
        assert isinstance(vows, list)
        # Our sample has iː, ɪ, ɛ, æ, ɑː, ə = 6 vowels
        assert len(vows) == 6

    def test_vowels_subset_of_phonemes(self, sample_inventory):
        assert set(sample_inventory.vowels) <= set(sample_inventory.phonemes)

    def test_tones_empty_for_non_tonal(self, sample_inventory):
        """Non-tonal language inventory has empty tones list."""
        assert sample_inventory.tones == []

    def test_tones_for_tonal_language(self, tonal_inventory):
        """Tonal language inventory returns tone segments."""
        tones = tonal_inventory.tones
        assert len(tones) == 4
        assert "˥" in tones
        assert "˥˩" in tones

    def test_tones_subset_of_phonemes(self, tonal_inventory):
        assert set(tonal_inventory.tones) <= set(tonal_inventory.phonemes)

    def test_consonants_vowels_tones_partition(self, tonal_inventory):
        """Consonants + vowels + tones should equal all phonemes."""
        all_parts = (
            tonal_inventory.consonants
            + tonal_inventory.vowels
            + tonal_inventory.tones
        )
        assert sorted(all_parts) == sorted(tonal_inventory.phonemes)

    def test_partition_non_tonal(self, sample_inventory):
        """For non-tonal: consonants + vowels = all phonemes."""
        all_parts = sample_inventory.consonants + sample_inventory.vowels
        assert sorted(all_parts) == sorted(sample_inventory.phonemes)


# ---------------------------------------------------------------------------
# 4. Inventory — marginal phoneme filtering
# ---------------------------------------------------------------------------


class TestInventoryMarginal:
    """Tests for marginal/non-marginal phoneme filtering."""

    def test_marginal_phonemes(self, inventory_with_marginals):
        marginal = inventory_with_marginals.marginal_phonemes
        assert isinstance(marginal, list)
        assert set(marginal) == {"ʒ", "x"}

    def test_non_marginal_phonemes(self, inventory_with_marginals):
        non_marginal = inventory_with_marginals.non_marginal_phonemes
        assert isinstance(non_marginal, list)
        assert set(non_marginal) == {"p", "t", "k", "a", "i"}

    def test_marginal_plus_non_marginal_equals_all(self, inventory_with_marginals):
        all_ph = set(inventory_with_marginals.phonemes)
        marginal = set(inventory_with_marginals.marginal_phonemes)
        non_marginal = set(inventory_with_marginals.non_marginal_phonemes)
        assert marginal | non_marginal == all_ph
        assert marginal & non_marginal == set()

    def test_no_marginals_gives_empty_list(self, sample_inventory):
        assert sample_inventory.marginal_phonemes == []

    def test_all_non_marginal_when_no_marginals(self, sample_inventory):
        assert len(sample_inventory.non_marginal_phonemes) == len(sample_inventory.phonemes)


# ---------------------------------------------------------------------------
# 5. Inventory — segment access by class
# ---------------------------------------------------------------------------


class TestInventorySegmentAccess:
    """Tests for accessing full Segment objects by class."""

    def test_consonant_segments(self, sample_inventory):
        """consonant_segments returns Segment objects, not just strings."""
        segs = sample_inventory.consonant_segments
        assert all(isinstance(s, Segment) for s in segs)
        assert all(s.segment_class == "consonant" for s in segs)
        assert len(segs) == 9

    def test_vowel_segments(self, sample_inventory):
        segs = sample_inventory.vowel_segments
        assert all(isinstance(s, Segment) for s in segs)
        assert all(s.segment_class == "vowel" for s in segs)
        assert len(segs) == 6

    def test_tone_segments(self, tonal_inventory):
        segs = tonal_inventory.tone_segments
        assert all(isinstance(s, Segment) for s in segs)
        assert all(s.segment_class == "tone" for s in segs)
        assert len(segs) == 4

    def test_marginal_segments(self, inventory_with_marginals):
        segs = inventory_with_marginals.marginal_segments
        assert all(isinstance(s, Segment) for s in segs)
        assert all(s.marginal is True for s in segs)
        assert len(segs) == 2

    def test_non_marginal_segments(self, inventory_with_marginals):
        segs = inventory_with_marginals.non_marginal_segments
        assert all(isinstance(s, Segment) for s in segs)
        assert all(s.marginal is False for s in segs)


# ---------------------------------------------------------------------------
# 6. Inventory — allophone access
# ---------------------------------------------------------------------------


class TestInventoryAllophones:
    """Tests for allophone data preservation."""

    def test_allophones_preserved(self, inventory_with_marginals):
        """Segments retain their allophone lists."""
        zh_seg = [s for s in inventory_with_marginals.segments if s.phoneme == "ʒ"][0]
        assert zh_seg.allophones == ["ʒ", "dʒ"]

    def test_all_allophones(self, inventory_with_marginals):
        """all_allophones returns a dict mapping phoneme → allophone list."""
        allo = inventory_with_marginals.all_allophones
        assert isinstance(allo, dict)
        assert allo["ʒ"] == ["ʒ", "dʒ"]
        # Segments with no allophones map to empty list
        assert allo["p"] == []

    def test_all_allophones_keys_equal_phonemes(self, inventory_with_marginals):
        allo = inventory_with_marginals.all_allophones
        assert set(allo.keys()) == set(inventory_with_marginals.phonemes)


# ---------------------------------------------------------------------------
# 7. Inventory — feature queries
# ---------------------------------------------------------------------------


class TestInventoryFeatureQueries:
    """Tests for querying segments by distinctive features."""

    def test_segments_with_feature(self, sample_inventory):
        """Can query segments that have a specific feature value."""
        nasals = sample_inventory.segments_with_feature("nasal", "+")
        assert isinstance(nasals, list)
        assert all(isinstance(s, Segment) for s in nasals)
        nasal_phonemes = [s.phoneme for s in nasals]
        assert "m" in nasal_phonemes
        assert "n" in nasal_phonemes
        assert "p" not in nasal_phonemes

    def test_segments_with_feature_labial(self, sample_inventory):
        labials = sample_inventory.segments_with_feature("labial", "+")
        labial_phonemes = [s.phoneme for s in labials]
        assert "p" in labial_phonemes
        assert "b" in labial_phonemes
        assert "m" in labial_phonemes
        assert "t" not in labial_phonemes

    def test_segments_with_feature_invalid_feature(self, sample_inventory):
        """Querying an invalid feature name raises ValueError."""
        with pytest.raises(ValueError, match="feature"):
            sample_inventory.segments_with_feature("nonexistent", "+")

    def test_segments_with_feature_invalid_value(self, sample_inventory):
        """Querying with invalid feature value raises ValueError."""
        with pytest.raises(ValueError, match="value"):
            sample_inventory.segments_with_feature("nasal", "yes")

    def test_segments_with_features_multi(self, sample_inventory):
        """Can query with multiple feature constraints."""
        # coronal + nasal = n
        results = sample_inventory.segments_with_features(
            {"coronal": "+", "nasal": "+"}
        )
        phonemes = [s.phoneme for s in results]
        assert "n" in phonemes
        assert "m" not in phonemes  # labial nasal, not coronal

    def test_segments_with_features_empty_dict(self, sample_inventory):
        """Empty feature dict returns all segments."""
        results = sample_inventory.segments_with_features({})
        assert len(results) == len(sample_inventory.segments)


# ---------------------------------------------------------------------------
# 8. Inventory — summary statistics
# ---------------------------------------------------------------------------


class TestInventorySummary:
    """Tests for inventory summary properties."""

    def test_size(self, sample_inventory):
        assert sample_inventory.size == 15

    def test_consonant_count(self, sample_inventory):
        assert sample_inventory.consonant_count == 9

    def test_vowel_count(self, sample_inventory):
        assert sample_inventory.vowel_count == 6

    def test_tone_count_non_tonal(self, sample_inventory):
        assert sample_inventory.tone_count == 0

    def test_tone_count_tonal(self, tonal_inventory):
        assert tonal_inventory.tone_count == 4

    def test_has_tones(self, tonal_inventory):
        assert tonal_inventory.has_tones is True

    def test_has_tones_false(self, sample_inventory):
        assert sample_inventory.has_tones is False

    def test_marginal_count(self, inventory_with_marginals):
        assert inventory_with_marginals.marginal_count == 2

    def test_marginal_count_zero(self, sample_inventory):
        assert sample_inventory.marginal_count == 0


# ---------------------------------------------------------------------------
# 9. Inventory — serialization
# ---------------------------------------------------------------------------


class TestInventorySerialization:
    """Tests for export to dict and compatibility with evaluate()."""

    def test_to_dict(self, sample_inventory):
        d = sample_inventory.to_dict()
        assert isinstance(d, dict)
        assert d["inventory_id"] == 160
        assert d["language_name"] == "English"
        assert d["iso639_3"] == "eng"
        assert d["glottocode"] == "stan1293"
        assert d["source"] == "spa"
        assert len(d["segments"]) == 15

    def test_to_dict_segment_structure(self, sample_inventory):
        d = sample_inventory.to_dict()
        seg = d["segments"][0]
        assert "phoneme" in seg
        assert "segment_class" in seg
        assert "marginal" in seg
        assert "allophones" in seg
        assert "features" in seg
        assert "glyph_id" in seg

    def test_to_dict_features_preserved(self, sample_inventory):
        d = sample_inventory.to_dict()
        seg = d["segments"][0]
        assert len(seg["features"]) == 38

    def test_phonemes_compatible_with_evaluate(self, sample_inventory):
        """phonemes property returns a list[str] that can be passed
        directly to evaluate(target_phonemes=...)."""
        phonemes = sample_inventory.phonemes
        assert isinstance(phonemes, list)
        assert all(isinstance(p, str) for p in phonemes)
        # No duplicates
        assert len(phonemes) == len(set(phonemes))


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


class TestInventoryEdgeCases:
    """Boundary conditions."""

    def test_empty_inventory(self):
        inv = Inventory(
            inventory_id=0,
            language_name="Empty",
            iso639_3="xxx",
            glottocode="empt0000",
            specific_dialect=None,
            source="test",
            segments=[],
        )
        assert inv.size == 0
        assert inv.phonemes == []
        assert inv.consonants == []
        assert inv.vowels == []
        assert inv.tones == []
        assert inv.marginal_phonemes == []
        assert inv.non_marginal_phonemes == []
        assert inv.has_tones is False
        assert inv.all_allophones == {}

    def test_single_segment_inventory(self):
        inv = Inventory(
            inventory_id=1,
            language_name="Minimal",
            iso639_3="min",
            glottocode="mini0001",
            specific_dialect=None,
            source="test",
            segments=[_vowel("a")],
        )
        assert inv.size == 1
        assert inv.phonemes == ["a"]
        assert inv.consonants == []
        assert inv.vowels == ["a"]

    def test_only_tones_inventory(self):
        """An inventory with only tone segments (degenerate case)."""
        inv = Inventory(
            inventory_id=2,
            language_name="ToneOnly",
            iso639_3="ton",
            glottocode="tone0001",
            specific_dialect=None,
            source="test",
            segments=[_tone("˥"), _tone("˩")],
        )
        assert inv.consonants == []
        assert inv.vowels == []
        assert inv.tones == ["˥", "˩"]
        assert inv.has_tones is True

    def test_unicode_phonemes(self):
        """Segments with complex Unicode IPA symbols."""
        segments = [
            _consonant("t͡ʃ"),   # affricate with tie bar
            _consonant("d͡ʒ"),
            _consonant("ɲ"),    # palatal nasal
            _consonant("ŋ"),    # velar nasal
            _vowel("ɛ̃"),       # nasalized vowel
            _vowel("ɔ̃"),
        ]
        inv = Inventory(
            inventory_id=3,
            language_name="UnicodeTest",
            iso639_3="uni",
            glottocode="unic0001",
            specific_dialect=None,
            source="test",
            segments=segments,
        )
        assert "t͡ʃ" in inv.phonemes
        assert "ɛ̃" in inv.vowels
        assert inv.size == 6

    def test_inventory_repr(self, sample_inventory):
        r = repr(sample_inventory)
        assert "English" in r or "eng" in r or "160" in r
