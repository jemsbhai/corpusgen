"""G2PResult: data container for grapheme-to-phoneme conversion output."""

from dataclasses import dataclass


@dataclass(frozen=True)
class G2PResult:
    """Result of a grapheme-to-phoneme conversion.

    Attributes:
        text: Original input text.
        ipa: IPA transcription string.
        phonemes: List of individual phoneme segments.
        language: Language code used for conversion.
    """

    text: str
    ipa: str
    phonemes: list[str]
    language: str

    @property
    def diphones(self) -> list[str]:
        """Adjacent phoneme pairs (bigrams)."""
        return [
            f"{self.phonemes[i]}-{self.phonemes[i + 1]}"
            for i in range(len(self.phonemes) - 1)
        ]

    @property
    def triphones(self) -> list[str]:
        """Adjacent phoneme triples (trigrams)."""
        return [
            f"{self.phonemes[i]}-{self.phonemes[i + 1]}-{self.phonemes[i + 2]}"
            for i in range(len(self.phonemes) - 2)
        ]

    @property
    def phoneme_count(self) -> int:
        """Total number of phonemes."""
        return len(self.phonemes)

    @property
    def unique_phonemes(self) -> set[str]:
        """Set of distinct phonemes."""
        return set(self.phonemes)
