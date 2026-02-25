"""Data models for phonological segments and inventories.

Pure data containers with no I/O or network access. These represent
PHOIBLE phonological inventory data with full fidelity: 38 distinctive
features, allophones, marginal status, and segment classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FEATURE_NAMES: tuple[str, ...] = (
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
)
"""All 38 PHOIBLE distinctive feature column names, in canonical order."""

_VALID_FEATURE_VALUES = ("+", "-", "0")
_VALID_SEGMENT_CLASSES = ("consonant", "vowel", "tone")


@dataclass(eq=False)
class Segment:
    """A single phonological segment with full PHOIBLE metadata.

    Hashable so segments are usable in sets and as dict keys.

    Attributes:
        phoneme: IPA symbol (e.g., 'p', 'tʃ', 'ɛ̃', '˥˩').
        segment_class: One of 'consonant', 'vowel', 'tone'.
        marginal: Whether the segment is marginal in this inventory.
        allophones: List of allophonic variants (IPA strings).
        features: Dict of 38 distinctive features, each '+', '-', or '0'.
        glyph_id: Unicode glyph ID string from PHOIBLE.
    """

    phoneme: str
    segment_class: str
    marginal: bool
    allophones: list[str]
    features: dict[str, str]
    glyph_id: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return NotImplemented
        return (
            self.phoneme == other.phoneme
            and self.segment_class == other.segment_class
            and self.marginal == other.marginal
            and self.allophones == other.allophones
            and self.features == other.features
            and self.glyph_id == other.glyph_id
        )

    def __hash__(self) -> int:
        return hash((
            self.phoneme,
            self.segment_class,
            self.marginal,
            tuple(self.allophones),
            tuple(sorted(self.features.items())),
            self.glyph_id,
        ))

    def __repr__(self) -> str:
        return (
            f"Segment(phoneme={self.phoneme!r}, "
            f"class={self.segment_class!r}, "
            f"marginal={self.marginal})"
        )


@dataclass
class Inventory:
    """A single phonological inventory from one PHOIBLE source.

    Represents a complete phoneme inventory for a language/dialect as
    documented by a specific source. Preserves all PHOIBLE metadata
    including segments, allophones, distinctive features, and provenance.

    Attributes:
        inventory_id: PHOIBLE InventoryID.
        language_name: Human-readable language name.
        iso639_3: ISO 639-3 language code.
        glottocode: Glottolog code.
        specific_dialect: Dialect specification, or None.
        source: PHOIBLE source identifier (e.g., 'spa', 'upsid', 'ph').
        segments: List of Segment objects in this inventory.
    """

    inventory_id: int
    language_name: str
    iso639_3: str
    glottocode: str
    specific_dialect: str | None
    source: str
    segments: list[Segment]

    # --- Phoneme lists (IPA strings) ---

    @property
    def phonemes(self) -> list[str]:
        """All IPA symbols in segment order."""
        return [s.phoneme for s in self.segments]

    @property
    def consonants(self) -> list[str]:
        """IPA symbols for consonant segments."""
        return [s.phoneme for s in self.segments if s.segment_class == "consonant"]

    @property
    def vowels(self) -> list[str]:
        """IPA symbols for vowel segments."""
        return [s.phoneme for s in self.segments if s.segment_class == "vowel"]

    @property
    def tones(self) -> list[str]:
        """IPA symbols for tone segments."""
        return [s.phoneme for s in self.segments if s.segment_class == "tone"]

    @property
    def marginal_phonemes(self) -> list[str]:
        """IPA symbols for marginal segments."""
        return [s.phoneme for s in self.segments if s.marginal]

    @property
    def non_marginal_phonemes(self) -> list[str]:
        """IPA symbols for non-marginal segments."""
        return [s.phoneme for s in self.segments if not s.marginal]

    # --- Full Segment access by class ---

    @property
    def consonant_segments(self) -> list[Segment]:
        """Consonant Segment objects."""
        return [s for s in self.segments if s.segment_class == "consonant"]

    @property
    def vowel_segments(self) -> list[Segment]:
        """Vowel Segment objects."""
        return [s for s in self.segments if s.segment_class == "vowel"]

    @property
    def tone_segments(self) -> list[Segment]:
        """Tone Segment objects."""
        return [s for s in self.segments if s.segment_class == "tone"]

    @property
    def marginal_segments(self) -> list[Segment]:
        """Marginal Segment objects."""
        return [s for s in self.segments if s.marginal]

    @property
    def non_marginal_segments(self) -> list[Segment]:
        """Non-marginal Segment objects."""
        return [s for s in self.segments if not s.marginal]

    # --- Allophone access ---

    @property
    def all_allophones(self) -> dict[str, list[str]]:
        """Map each phoneme to its allophone list."""
        return {s.phoneme: list(s.allophones) for s in self.segments}

    # --- Feature queries ---

    def segments_with_feature(
        self, feature: str, value: str
    ) -> list[Segment]:
        """Return segments matching a single feature constraint.

        Args:
            feature: Distinctive feature name (e.g., 'nasal', 'labial').
            value: Required value ('+', '-', or '0').

        Returns:
            List of matching Segment objects.

        Raises:
            ValueError: If feature name or value is invalid.
        """
        if feature not in FEATURE_NAMES:
            raise ValueError(
                f"Invalid feature: {feature!r}. "
                f"Must be one of: {', '.join(FEATURE_NAMES[:5])}... "
                f"({len(FEATURE_NAMES)} total)"
            )
        if value not in _VALID_FEATURE_VALUES:
            raise ValueError(
                f"Invalid feature value: {value!r}. "
                f"Must be one of {_VALID_FEATURE_VALUES}"
            )
        return [s for s in self.segments if s.features.get(feature) == value]

    def segments_with_features(
        self, constraints: dict[str, str]
    ) -> list[Segment]:
        """Return segments matching multiple feature constraints.

        Args:
            constraints: Dict of {feature_name: required_value}.

        Returns:
            List of Segment objects matching ALL constraints.

        Raises:
            ValueError: If any feature name or value is invalid.
        """
        if not constraints:
            return list(self.segments)

        for feat, val in constraints.items():
            if feat not in FEATURE_NAMES:
                raise ValueError(
                    f"Invalid feature: {feat!r}. "
                    f"Must be one of: {', '.join(FEATURE_NAMES[:5])}... "
                    f"({len(FEATURE_NAMES)} total)"
                )
            if val not in _VALID_FEATURE_VALUES:
                raise ValueError(
                    f"Invalid feature value: {val!r} for feature {feat!r}. "
                    f"Must be one of {_VALID_FEATURE_VALUES}"
                )

        return [
            s for s in self.segments
            if all(s.features.get(f) == v for f, v in constraints.items())
        ]

    # --- Summary statistics ---

    @property
    def size(self) -> int:
        """Total number of segments."""
        return len(self.segments)

    @property
    def consonant_count(self) -> int:
        """Number of consonant segments."""
        return sum(1 for s in self.segments if s.segment_class == "consonant")

    @property
    def vowel_count(self) -> int:
        """Number of vowel segments."""
        return sum(1 for s in self.segments if s.segment_class == "vowel")

    @property
    def tone_count(self) -> int:
        """Number of tone segments."""
        return sum(1 for s in self.segments if s.segment_class == "tone")

    @property
    def has_tones(self) -> bool:
        """Whether this inventory includes tone segments."""
        return self.tone_count > 0

    @property
    def marginal_count(self) -> int:
        """Number of marginal segments."""
        return sum(1 for s in self.segments if s.marginal)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Export as a plain Python dict.

        Returns:
            Dict with all inventory metadata and segment data,
            suitable for JSON serialization.
        """
        return {
            "inventory_id": self.inventory_id,
            "language_name": self.language_name,
            "iso639_3": self.iso639_3,
            "glottocode": self.glottocode,
            "specific_dialect": self.specific_dialect,
            "source": self.source,
            "segments": [
                {
                    "phoneme": s.phoneme,
                    "segment_class": s.segment_class,
                    "marginal": s.marginal,
                    "allophones": list(s.allophones),
                    "features": dict(s.features),
                    "glyph_id": s.glyph_id,
                }
                for s in self.segments
            ],
        }

    def __repr__(self) -> str:
        dialect = f", dialect={self.specific_dialect!r}" if self.specific_dialect else ""
        return (
            f"Inventory(id={self.inventory_id}, "
            f"lang={self.language_name!r} [{self.iso639_3}], "
            f"source={self.source!r}, "
            f"segments={self.size}"
            f"{dialect})"
        )
