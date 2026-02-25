"""RepositoryBackend: selects from a pre-existing sentence pool.

Bridges the selection paradigm with the Phon-CTG generation loop.
The user provides a pool of sentences (pre-phonemized, raw text,
or from a HuggingFace dataset), and the backend returns candidates
that best target the requested phonetic units.
"""

from __future__ import annotations

from typing import Any


from corpusgen.generate.phon_ctg.loop import GenerationBackend


def _load_hf_dataset(
    dataset_name: str,
    split: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load a HuggingFace dataset. Isolated for mockability.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    return load_dataset(dataset_name, split=split, **kwargs)


def _extract_units(phonemes: list[str], target_units: list[str]) -> set[str]:
    """Extract phoneme, diphone, and triphone units present in a phoneme list.

    Detects the unit type from the format of target_units (contains '-' or not)
    and extracts accordingly.
    """
    if not phonemes or not target_units:
        return set()

    # Determine unit types needed from target format
    target_set = set(target_units)
    found: set[str] = set()

    # Check plain phonemes
    phoneme_set = set(phonemes)
    found.update(phoneme_set & target_set)

    # Check diphones
    for i in range(len(phonemes) - 1):
        diphone = f"{phonemes[i]}-{phonemes[i + 1]}"
        if diphone in target_set:
            found.add(diphone)

    # Check triphones
    for i in range(len(phonemes) - 2):
        triphone = f"{phonemes[i]}-{phonemes[i + 1]}-{phonemes[i + 2]}"
        if triphone in target_set:
            found.add(triphone)

    return found


class RepositoryBackend(GenerationBackend):
    """Generation backend that selects from a sentence pool.

    Candidates are ranked by how many of the requested target units
    they contain. Used sentences can be removed via ``mark_used()``.

    Args:
        pool: List of dicts, each with ``"phonemes"`` (list[str])
            and optionally ``"text"`` (str).
    """

    def __init__(self, pool: list[dict]) -> None:
        # Validate pool entries
        for i, entry in enumerate(pool):
            if "phonemes" not in entry:
                raise ValueError(
                    f"Pool entry {i} missing 'phonemes' key. "
                    f"Each entry must have 'phonemes' (list[str])."
                )
        self._pool = [dict(entry) for entry in pool]

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "repository"

    @property
    def pool_size(self) -> int:
        """Number of sentences remaining in the pool."""
        return len(self._pool)

    @property
    def pool(self) -> list[dict]:
        """Copy of the current pool."""
        return [dict(entry) for entry in self._pool]

    # -------------------------------------------------------------------
    # Class methods: alternative constructors
    # -------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        language: str = "en-us",
    ) -> RepositoryBackend:
        """Create a backend from raw text by running G2P.

        Args:
            texts: List of sentences.
            language: Language code for G2P conversion.

        Returns:
            A RepositoryBackend with phonemized pool.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("Empty texts list.")

        from corpusgen.g2p.manager import G2PManager

        g2p = G2PManager()
        results = g2p.phonemize_batch(texts, language=language)

        pool = [
            {"text": text, "phonemes": result.phonemes}
            for text, result in zip(texts, results)
        ]
        return cls(pool=pool)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        text_column: str = "text",
        split: str | None = None,
        language: str = "en-us",
        max_samples: int | None = None,
        **dataset_kwargs: Any,
    ) -> RepositoryBackend:
        """Create a backend from a HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "wikitext").
            text_column: Column name containing text data.
            split: Dataset split (e.g., "train", "test").
            language: Language code for G2P conversion.
            max_samples: Maximum number of samples to load.
            **dataset_kwargs: Forwarded to ``datasets.load_dataset()``.

        Returns:
            A RepositoryBackend with phonemized pool.

        Raises:
            ImportError: If ``datasets`` package is not installed.
        """
        ds = _load_hf_dataset(dataset_name, split=split, **dataset_kwargs)

        texts: list[str] = []
        for row in ds:
            texts.append(row[text_column])
            if max_samples is not None and len(texts) >= max_samples:
                break

        return cls.from_texts(texts=texts, language=language)

    # -------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------

    def generate(
        self,
        target_units: list[str],
        k: int = 5,
        **kwargs: Any,
    ) -> list[dict]:
        """Return top-k pool sentences ranked by target unit overlap.

        Args:
            target_units: Phonetic units to target.
            k: Maximum candidates to return.

        Returns:
            List of candidate dicts with "text", "phonemes", and
            "_pool_index" (for ``mark_used``).
        """
        if k <= 0 or not self._pool:
            return []

        # Score each pool entry by how many target units it covers
        scored: list[tuple[int, int, dict]] = []
        for idx, entry in enumerate(self._pool):
            hits = _extract_units(entry["phonemes"], target_units)
            scored.append((len(hits), idx, entry))

        # Sort by hit count descending, stable by pool order
        scored.sort(key=lambda x: -x[0])

        # Return top-k with positive hits, or top-k overall if no hits
        candidates: list[dict] = []
        for hit_count, idx, entry in scored[:k]:
            candidates.append({
                "text": entry.get("text", ""),
                "phonemes": entry["phonemes"],
                "_pool_index": idx,
            })

        return candidates

    # -------------------------------------------------------------------
    # Deduplication
    # -------------------------------------------------------------------

    def mark_used(self, pool_index: int) -> None:
        """Remove a sentence from the pool by its index.

        Args:
            pool_index: Index into the current pool.

        Raises:
            IndexError: If index is out of range.
        """
        if pool_index < 0 or pool_index >= len(self._pool):
            raise IndexError(
                f"Pool index {pool_index} out of range "
                f"(pool size: {len(self._pool)})"
            )
        self._pool.pop(pool_index)
