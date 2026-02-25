"""EspeakMapping: bidirectional mapping between espeak-ng and ISO 639-3 codes.

Loads a bundled JSON mapping that connects espeak-ng voice identifiers
to ISO 639-3 codes used by PHOIBLE. Handles macrolanguage resolution,
dialect grouping, and case-insensitive lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


_MAPPING_FILE = Path(__file__).parent.parent / "data" / "espeak_iso_mapping.json"


class EspeakMapping:
    """Bidirectional mapping between espeak-ng voice codes and ISO 639-3.

    Loads the bundled ``espeak_iso_mapping.json`` on construction.
    Macrolanguage codes are pre-resolved to the specific variety
    present in PHOIBLE (e.g., ``ms`` → ``zsm`` for Standard Malay).

    Example::

        m = EspeakMapping()
        m.to_iso("en-us")    # → "eng"
        m.to_espeak("eng")   # → ["en-029", "en-gb", "en-gb-scotland", ..., "en-us", ...]
    """

    def __init__(self, mapping_file: Path | None = None) -> None:
        if mapping_file is None:
            mapping_file = _MAPPING_FILE

        with open(mapping_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Remove metadata keys
        self._forward: dict[str, str] = {
            k: v for k, v in raw.items() if not k.startswith("_")
        }

        # Build reverse index: ISO 639-3 → sorted list of espeak codes
        self._reverse: dict[str, list[str]] = {}
        for espeak_code, iso_code in self._forward.items():
            if iso_code not in self._reverse:
                self._reverse[iso_code] = []
            self._reverse[iso_code].append(espeak_code)

        for iso_code in self._reverse:
            self._reverse[iso_code].sort()

    @property
    def size(self) -> int:
        """Number of espeak → ISO mappings."""
        return len(self._forward)

    def to_iso(self, espeak_code: str) -> str:
        """Convert an espeak-ng voice code to an ISO 639-3 code.

        Args:
            espeak_code: espeak-ng voice identifier (e.g., 'en-us', 'fr-fr').
                Case-insensitive.

        Returns:
            ISO 639-3 code (e.g., 'eng', 'fra').

        Raises:
            KeyError: If the espeak code is not in the mapping.
        """
        key = espeak_code.lower()
        if key in self._forward:
            return self._forward[key]
        raise KeyError(
            f"Unknown espeak code: {espeak_code!r}. "
            f"Use espeak_codes() to list available codes."
        )

    def to_espeak(self, iso_code: str) -> list[str]:
        """Convert an ISO 639-3 code to espeak-ng voice code(s).

        A single ISO code may map to multiple espeak voices (dialects).

        Args:
            iso_code: ISO 639-3 code (e.g., 'eng', 'fra').

        Returns:
            Sorted list of espeak-ng voice codes.

        Raises:
            KeyError: If the ISO code is not in the reverse mapping.
        """
        if iso_code in self._reverse:
            return list(self._reverse[iso_code])
        raise KeyError(
            f"Unknown ISO 639-3 code: {iso_code!r}. "
            f"Use iso_codes() to list available codes."
        )

    def espeak_codes(self) -> list[str]:
        """List all espeak-ng voice codes in the mapping."""
        return sorted(self._forward.keys())

    def iso_codes(self) -> list[str]:
        """List all unique ISO 639-3 codes in the mapping (deduplicated)."""
        return sorted(self._reverse.keys())

    def items(self) -> Iterator[tuple[str, str]]:
        """Iterate over (espeak_code, iso_code) pairs."""
        return iter(sorted(self._forward.items()))
