"""corpusgen: Language-agnostic speech corpus generation with maximal phoneme coverage."""

__version__ = "0.1.0"

from corpusgen.evaluate import evaluate
from corpusgen.inventory.models import Inventory
from corpusgen.select import select_sentences


def get_inventory(
    language: str,
    source: str | None = None,
) -> Inventory:
    """Get a PHOIBLE phoneme inventory for a language.

    Accepts either an espeak-ng voice code (e.g., 'en-us', 'fr-fr')
    or an ISO 639-3 / Glottocode identifier (e.g., 'eng', 'stan1293').
    Tries espeak mapping first, then falls back to direct PHOIBLE lookup.

    Args:
        language: espeak-ng code, ISO 639-3 code, or Glottocode.
        source: Optional PHOIBLE source filter (e.g., 'spa', 'upsid').

    Returns:
        An Inventory object with phonemes, features, and metadata.

    Raises:
        KeyError: If the language identifier is not found.
    """
    from corpusgen.inventory.phoible import PhoibleDataset
    from corpusgen.inventory.mapping import EspeakMapping

    ds = PhoibleDataset()

    # Try espeak mapping first
    try:
        mapping = EspeakMapping()
        iso = mapping.to_iso(language)
        return ds.get_inventory(iso, source=source)
    except KeyError:
        pass

    # Fall back to direct PHOIBLE lookup (ISO 639-3 or Glottocode)
    return ds.get_inventory(language, source=source)


__all__ = ["evaluate", "get_inventory", "select_sentences", "Inventory", "__version__"]
