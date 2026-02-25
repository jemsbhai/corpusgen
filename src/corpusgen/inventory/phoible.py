"""PhoibleDataset: load, cache, and query PHOIBLE phoneme inventory data.

Downloads the PHOIBLE CSV from GitHub on first use, caches locally,
and provides efficient lookup by ISO 639-3, Glottocode, or language name.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from corpusgen.inventory.models import Inventory, Segment, FEATURE_NAMES


_PHOIBLE_CSV_URL = (
    "https://github.com/phoible/dev/blob/master/data/phoible.csv?raw=true"
)
_CSV_FILENAME = "phoible.csv"


class PhoibleDataset:
    """Manages access to the PHOIBLE phonological inventory database.

    Loads the PHOIBLE CSV (105K+ segment rows, 3,020 inventories,
    2,186 languages) and provides efficient query methods.

    Data is parsed lazily on first query, or explicitly via ``load()``.

    Args:
        cache_dir: Directory to store/read the cached phoible.csv.
            Defaults to ``~/.corpusgen/``.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".corpusgen"
        self._cache_dir = Path(cache_dir)
        self._csv_path = self._cache_dir / _CSV_FILENAME

        # Internal state — populated by load()
        self._loaded = False
        self._inventories: dict[int, Inventory] = {}  # InventoryID → Inventory
        self._iso_index: dict[str, list[int]] = {}    # ISO 639-3 → [InventoryID]
        self._glotto_index: dict[str, list[int]] = {} # Glottocode → [InventoryID]
        self._language_meta: dict[str, dict[str, Any]] = {}  # ISO → metadata

    # --- Properties ---

    @property
    def cache_dir(self) -> Path:
        """Directory where phoible.csv is cached."""
        return self._cache_dir

    @property
    def csv_path(self) -> Path:
        """Full path to the cached phoible.csv file."""
        return self._csv_path

    @property
    def csv_exists(self) -> bool:
        """Whether the cached CSV file exists on disk."""
        return self._csv_path.is_file()

    @property
    def is_loaded(self) -> bool:
        """Whether data has been parsed into memory."""
        return self._loaded

    @property
    def inventory_count(self) -> int:
        """Number of inventories loaded."""
        return len(self._inventories)

    @property
    def language_count(self) -> int:
        """Number of distinct languages (by ISO 639-3)."""
        return len(self._iso_index)

    @property
    def segment_count(self) -> int:
        """Total number of segments across all inventories."""
        return sum(inv.size for inv in self._inventories.values())

    # --- Loading ---

    def load(self) -> None:
        """Parse the cached PHOIBLE CSV into memory.

        If already loaded, resets and reloads (idempotent on data).

        Raises:
            FileNotFoundError: If the CSV file does not exist.
                Call ``download()`` first or provide a valid cache_dir.
        """
        if not self._csv_path.is_file():
            raise FileNotFoundError(
                f"PHOIBLE CSV not found at {self._csv_path}. "
                f"Call download() first or provide a valid cache_dir."
            )

        # Reset state
        self._inventories.clear()
        self._iso_index.clear()
        self._glotto_index.clear()
        self._language_meta.clear()

        # Parse CSV — group rows by InventoryID
        raw_inventories: dict[int, list[dict[str, str]]] = {}

        with open(self._csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                inv_id = int(row["InventoryID"])
                if inv_id not in raw_inventories:
                    raw_inventories[inv_id] = []
                raw_inventories[inv_id].append(row)

        # Build Inventory objects
        for inv_id, rows in raw_inventories.items():
            first = rows[0]

            segments: list[Segment] = []
            for row in rows:
                segments.append(_parse_segment(row))

            dialect = first["SpecificDialect"]
            if dialect == "NA" or not dialect:
                dialect = None

            inv = Inventory(
                inventory_id=inv_id,
                language_name=first["LanguageName"],
                iso639_3=first["ISO6393"],
                glottocode=first["Glottocode"],
                specific_dialect=dialect,
                source=first["Source"],
                segments=segments,
            )
            self._inventories[inv_id] = inv

            # Build indices
            iso = inv.iso639_3
            if iso not in self._iso_index:
                self._iso_index[iso] = []
            self._iso_index[iso].append(inv_id)

            glotto = inv.glottocode
            if glotto not in self._glotto_index:
                self._glotto_index[glotto] = []
            self._glotto_index[glotto].append(inv_id)

        # Build language metadata
        for iso, inv_ids in self._iso_index.items():
            invs = [self._inventories[i] for i in inv_ids]
            self._language_meta[iso] = {
                "iso639_3": iso,
                "glottocode": invs[0].glottocode,
                "language_name": invs[0].language_name,
                "inventory_count": len(inv_ids),
                "sources": sorted({inv.source for inv in invs}),
            }

        self._loaded = True

    def _ensure_loaded(self) -> None:
        """Auto-load from cache if not yet loaded."""
        if not self._loaded:
            self.load()

    # --- Query by ISO 639-3 or Glottocode ---

    def _resolve_identifier(self, identifier: str) -> list[int]:
        """Resolve an ISO 639-3 or Glottocode to inventory IDs.

        Tries ISO first, then Glottocode.

        Raises:
            KeyError: If identifier is not found in either index.
        """
        if identifier in self._iso_index:
            return self._iso_index[identifier]
        if identifier in self._glotto_index:
            return self._glotto_index[identifier]
        raise KeyError(
            f"Language identifier {identifier!r} not found. "
            f"Try search() to find the correct code."
        )

    def get_inventory(
        self, identifier: str, source: str | None = None
    ) -> Inventory:
        """Get a single inventory for a language.

        Args:
            identifier: ISO 639-3 code or Glottocode.
            source: If specified, return the inventory from this source.
                If None, returns the inventory with the most segments.

        Returns:
            An Inventory object.

        Raises:
            KeyError: If the identifier or source is not found.
        """
        self._ensure_loaded()
        inv_ids = self._resolve_identifier(identifier)

        if source is not None:
            for inv_id in inv_ids:
                inv = self._inventories[inv_id]
                if inv.source == source:
                    return inv
            raise KeyError(
                f"No inventory with source {source!r} for {identifier!r}. "
                f"Available sources: {[self._inventories[i].source for i in inv_ids]}"
            )

        # Default: return the inventory with the most segments
        best_id = max(inv_ids, key=lambda i: self._inventories[i].size)
        return self._inventories[best_id]

    def get_all_inventories(self, identifier: str) -> list[Inventory]:
        """Get all inventories for a language.

        Args:
            identifier: ISO 639-3 code or Glottocode.

        Returns:
            List of Inventory objects.

        Raises:
            KeyError: If the identifier is not found.
        """
        self._ensure_loaded()
        inv_ids = self._resolve_identifier(identifier)
        return [self._inventories[i] for i in inv_ids]

    def get_union_inventory(self, identifier: str) -> Inventory:
        """Get the union of all inventories for a language.

        Merges all inventories into a single maximally inclusive set.
        If a phoneme appears in multiple inventories, the Segment from
        the largest inventory is preferred (preserving its features,
        allophones, etc.).

        Args:
            identifier: ISO 639-3 code or Glottocode.

        Returns:
            A synthetic Inventory with source="union".

        Raises:
            KeyError: If the identifier is not found.
        """
        self._ensure_loaded()
        inventories = self.get_all_inventories(identifier)

        # Sort inventories largest-first so the biggest one's segments
        # take priority for metadata (features, allophones, etc.)
        inventories_sorted = sorted(inventories, key=lambda i: i.size, reverse=True)

        seen_phonemes: set[str] = set()
        merged_segments: list[Segment] = []

        for inv in inventories_sorted:
            for seg in inv.segments:
                if seg.phoneme not in seen_phonemes:
                    seen_phonemes.add(seg.phoneme)
                    merged_segments.append(seg)

        first = inventories[0]
        return Inventory(
            inventory_id=0,
            language_name=first.language_name,
            iso639_3=first.iso639_3,
            glottocode=first.glottocode,
            specific_dialect=None,
            source="union",
            segments=merged_segments,
        )

    # --- Search and listing ---

    def search(self, name: str) -> list[dict[str, Any]]:
        """Search for languages by name (case-insensitive, partial match).

        Args:
            name: Search string to match against language names.

        Returns:
            List of dicts with language metadata for each match.
        """
        self._ensure_loaded()
        query = name.lower()
        results = []
        for meta in self._language_meta.values():
            if query in meta["language_name"].lower():
                results.append(dict(meta))
        return results

    def available_languages(self) -> list[dict[str, Any]]:
        """List all languages in the dataset.

        Returns:
            Sorted list of dicts with language metadata.
        """
        self._ensure_loaded()
        langs = list(self._language_meta.values())
        langs.sort(key=lambda x: x["language_name"])
        return [dict(m) for m in langs]

    def sources_for(self, identifier: str) -> list[str]:
        """List available PHOIBLE sources for a language.

        Args:
            identifier: ISO 639-3 code or Glottocode.

        Returns:
            Sorted list of source identifiers.

        Raises:
            KeyError: If the identifier is not found.
        """
        self._ensure_loaded()
        inv_ids = self._resolve_identifier(identifier)
        sources = {self._inventories[i].source for i in inv_ids}
        return sorted(sources)

    # --- Espeak convenience methods ---

    def get_inventory_for_espeak(
        self, espeak_code: str, source: str | None = None
    ) -> Inventory:
        """Get an inventory using an espeak-ng voice code.

        Maps the espeak code to ISO 639-3 via the bundled mapping,
        then looks up the PHOIBLE inventory.

        Args:
            espeak_code: espeak-ng voice identifier (e.g., 'en-us').
            source: Optional PHOIBLE source filter.

        Returns:
            An Inventory object.

        Raises:
            KeyError: If the espeak code or resulting ISO is not found.
        """
        from corpusgen.inventory.mapping import EspeakMapping
        mapping = EspeakMapping()
        iso = mapping.to_iso(espeak_code)
        return self.get_inventory(iso, source=source)

    def get_union_inventory_for_espeak(self, espeak_code: str) -> Inventory:
        """Get the union inventory using an espeak-ng voice code.

        Args:
            espeak_code: espeak-ng voice identifier (e.g., 'en-us').

        Returns:
            A synthetic union Inventory.

        Raises:
            KeyError: If the espeak code or resulting ISO is not found.
        """
        from corpusgen.inventory.mapping import EspeakMapping
        mapping = EspeakMapping()
        iso = mapping.to_iso(espeak_code)
        return self.get_union_inventory(iso)

    # --- Download ---

    def download(self) -> None:
        """Download phoible.csv from GitHub and cache locally.

        Creates the cache directory if it doesn't exist.
        """
        import urllib.request

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_PHOIBLE_CSV_URL, self._csv_path)


# ---------------------------------------------------------------------------
# Internal CSV parsing helpers
# ---------------------------------------------------------------------------


def _parse_segment(row: dict[str, str]) -> Segment:
    """Parse a single CSV row into a Segment object."""
    # Parse allophones: space-separated IPA, "NA" → empty
    raw_allo = row.get("Allophones", "")
    if raw_allo and raw_allo != "NA":
        allophones = raw_allo.split(" ")
    else:
        allophones = []

    # Parse marginal: "TRUE" → True, everything else → False
    marginal = row.get("Marginal", "FALSE") == "TRUE"

    # Parse features
    features = {}
    for feat_name in FEATURE_NAMES:
        features[feat_name] = row.get(feat_name, "0")

    return Segment(
        phoneme=row["Phoneme"],
        segment_class=row["SegmentClass"],
        marginal=marginal,
        allophones=allophones,
        features=features,
        glyph_id=row.get("GlyphID", ""),
    )
