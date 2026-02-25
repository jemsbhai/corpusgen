"""EvaluationReport: structured output with verbosity levels and export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Verbosity(Enum):
    """Report verbosity levels."""

    MINIMAL = "minimal"
    NORMAL = "normal"
    VERBOSE = "verbose"


@dataclass
class SentenceDetail:
    """Per-sentence coverage breakdown (used in verbose output).

    Attributes:
        index: Sentence position in the corpus.
        text: The original sentence text.
        phoneme_count: Total phonemes in this sentence.
        new_phonemes: Phonemes first covered by this sentence.
        all_phonemes: All phonemes found in this sentence.
    """

    index: int
    text: str
    phoneme_count: int
    new_phonemes: list[str]
    all_phonemes: list[str]


@dataclass
class EvaluationReport:
    """Complete evaluation result with multi-level rendering and export.

    This object always holds the full data internally. The verbosity level
    only controls what ``.render()`` outputs as human-readable text.

    Attributes:
        language: Language code used for evaluation.
        unit: Coverage unit type (phoneme, diphone, triphone).
        target_phonemes: Full list of target phonemes.
        covered_phonemes: Set of phonemes that were covered.
        missing_phonemes: Set of phonemes not yet covered.
        coverage: Coverage fraction (0.0 to 1.0).
        phoneme_counts: Per-phoneme occurrence counts.
        total_sentences: Number of sentences evaluated.
        sentence_details: Per-sentence breakdown (for verbose output).
        phoneme_sources: Maps each phoneme to source sentence indices.
    """

    language: str
    unit: str
    target_phonemes: list[str]
    covered_phonemes: set[str]
    missing_phonemes: set[str]
    coverage: float
    phoneme_counts: dict[str, int]
    total_sentences: int
    sentence_details: list[SentenceDetail] = field(default_factory=list)
    phoneme_sources: dict[str, list[int]] = field(default_factory=dict)

    # --- Rendering ---

    def render(self, verbosity: Verbosity = Verbosity.NORMAL) -> str:
        """Render a human-readable report at the specified verbosity level.

        Args:
            verbosity: One of Verbosity.MINIMAL, NORMAL, or VERBOSE.

        Returns:
            Formatted string report.
        """
        if verbosity == Verbosity.MINIMAL:
            return self._render_minimal()
        elif verbosity == Verbosity.NORMAL:
            return self._render_normal()
        elif verbosity == Verbosity.VERBOSE:
            return self._render_verbose()
        else:
            raise ValueError(f"Unknown verbosity: {verbosity!r}")

    def _render_minimal(self) -> str:
        covered = len(self.covered_phonemes)
        total = len(self.target_phonemes)
        pct = self.coverage * 100
        missing_str = ", ".join(sorted(self.missing_phonemes))

        lines = [
            f"Coverage: {pct:.1f}% ({covered}/{total} {self.unit}s)",
            f"Missing: {missing_str}" if missing_str else "Missing: none",
        ]
        return "\n".join(lines)

    def _render_normal(self) -> str:
        parts = [self._render_minimal(), ""]

        # Per-phoneme counts
        parts.append("Per-phoneme counts:")
        for phoneme in sorted(self.phoneme_counts.keys()):
            count = self.phoneme_counts[phoneme]
            parts.append(f"  {phoneme}: {count}")

        return "\n".join(parts)

    def _render_verbose(self) -> str:
        parts = [self._render_normal(), ""]

        # Per-sentence breakdown
        parts.append("Sentence details:")
        for sd in self.sentence_details:
            new_str = ", ".join(sd.new_phonemes) if sd.new_phonemes else "none"
            parts.append(
                f"  [{sd.index}] \"{sd.text}\" — "
                f"{sd.phoneme_count} phonemes, "
                f"new: [{new_str}]"
            )

        parts.append("")

        # Phoneme source mapping
        parts.append("Phoneme sources:")
        for phoneme in sorted(self.phoneme_sources.keys()):
            indices = self.phoneme_sources[phoneme]
            parts.append(f"  {phoneme}: sentences {indices}")

        return "\n".join(parts)

    # --- Export ---

    def to_dict(self) -> dict[str, Any]:
        """Export as a plain Python dict.

        Sets are converted to sorted lists for JSON compatibility.
        """
        return {
            "language": self.language,
            "unit": self.unit,
            "target_phonemes": list(self.target_phonemes),
            "covered_phonemes": sorted(self.covered_phonemes),
            "missing_phonemes": sorted(self.missing_phonemes),
            "coverage": self.coverage,
            "phoneme_counts": dict(self.phoneme_counts),
            "total_sentences": self.total_sentences,
            "sentence_details": [
                {
                    "index": sd.index,
                    "text": sd.text,
                    "phoneme_count": sd.phoneme_count,
                    "new_phonemes": sd.new_phonemes,
                    "all_phonemes": sd.all_phonemes,
                }
                for sd in self.sentence_details
            ],
            "phoneme_sources": dict(self.phoneme_sources),
        }

    def to_json(self, indent: int | None = None) -> str:
        """Export as a JSON string.

        Args:
            indent: JSON indentation level. None for compact output.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_jsonld_ex(self) -> dict[str, Any]:
        """Export as a JSON-LD document compatible with jsonld-ex.

        Returns a JSON-LD document with @context, @type, and all
        evaluation data mapped to linked data terms.
        """
        base = self.to_dict()

        return {
            "@context": {
                "@vocab": "https://corpusgen.dev/ns/",
                "corpusgen": "https://corpusgen.dev/ns/",
                "ipa": "https://www.internationalphoneticassociation.org/ns/",
                "schema": "https://schema.org/",
                "language": "schema:inLanguage",
                "coverage": "corpusgen:coverageRatio",
                "unit": "corpusgen:coverageUnit",
                "target_phonemes": "corpusgen:targetPhonemes",
                "covered_phonemes": "corpusgen:coveredPhonemes",
                "missing_phonemes": "corpusgen:missingPhonemes",
                "phoneme_counts": "corpusgen:phonemeCounts",
                "total_sentences": "corpusgen:totalSentences",
                "sentence_details": "corpusgen:sentenceDetails",
                "phoneme_sources": "corpusgen:phonemeSources",
            },
            "@type": "corpusgen:EvaluationReport",
            **base,
        }
