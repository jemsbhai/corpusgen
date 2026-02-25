"""Sentence selection algorithms: greedy, CELF, ILP, stochastic, distribution-aware, NSGA-II."""

from __future__ import annotations

from corpusgen.select.base import SelectorBase
from corpusgen.select.result import SelectionResult
from corpusgen.select.greedy import GreedySelector
from corpusgen.select.celf import CELFSelector
from corpusgen.select.stochastic import StochasticGreedySelector
from corpusgen.select.distribution import DistributionAwareSelector

# Optional dependency selectors
try:
    from corpusgen.select.ilp import ILPSelector
except ImportError:
    ILPSelector = None  # type: ignore[assignment, misc]

try:
    from corpusgen.select.nsga2 import NSGA2Selector
except ImportError:
    NSGA2Selector = None  # type: ignore[assignment, misc]


# Algorithms that need no optional dependencies
_CORE_ALGORITHMS = {
    "greedy": GreedySelector,
    "celf": CELFSelector,
    "stochastic": StochasticGreedySelector,
    "distribution": DistributionAwareSelector,
}

# Algorithms that need optional dependencies
_OPTIONAL_ALGORITHMS = {
    "ilp": ("ILPSelector", "pulp", "corpusgen[optimization]"),
    "nsga2": ("NSGA2Selector", "pymoo", "corpusgen[optimization]"),
}


def select_sentences(
    candidates: list[str],
    language: str = "en-us",
    target_phonemes: list[str] | str | None = None,
    unit: str = "phoneme",
    algorithm: str = "greedy",
    max_sentences: int | None = None,
    target_coverage: float = 1.0,
    candidate_phonemes: list[list[str]] | None = None,
    weights: dict[str, float] | None = None,
    **algorithm_kwargs,
) -> SelectionResult:
    """Select sentences from candidates for maximal phoneme coverage.

    This is the primary user-facing API for corpus selection. It handles
    G2P conversion, target inventory resolution, and algorithm dispatch.

    Args:
        candidates: List of candidate sentences (raw text).
        language: Language code for G2P conversion (e.g., 'en-us', 'fr-fr').
            Ignored if candidate_phonemes is provided.
        target_phonemes: Target phoneme inventory. If None, derived from all
            unique phonemes in candidates. If ``"phoible"``, fetched from
            the PHOIBLE database for the given language.
        unit: Coverage unit type — "phoneme", "diphone", or "triphone".
        algorithm: Selection algorithm — "greedy", "celf", "stochastic",
            "ilp", "distribution", or "nsga2".
        max_sentences: Maximum number of sentences to select (budget).
            None means no limit.
        target_coverage: Stop when this coverage fraction is reached.
        candidate_phonemes: Pre-phonemized candidates. If provided, G2P
            is skipped entirely. Must have same length as candidates.
        weights: Optional mapping from unit to weight for marginal gain.
            If None, all units are weighted equally (1.0).
        **algorithm_kwargs: Passed to the algorithm constructor (e.g.,
            epsilon and seed for stochastic, target_distribution for
            distribution, population_size/n_generations for nsga2).

    Returns:
        SelectionResult with selected sentences and coverage metrics.

    Raises:
        ValueError: If algorithm, unit, or inputs are invalid.
        ImportError: If the requested algorithm needs an uninstalled dependency.
    """
    valid_units = ("phoneme", "diphone", "triphone")
    if unit not in valid_units:
        raise ValueError(
            f"Invalid unit: {unit!r}. Must be one of {valid_units}"
        )

    # --- Validate candidate_phonemes length ---
    if candidate_phonemes is not None and len(candidate_phonemes) != len(candidates):
        raise ValueError(
            f"candidate_phonemes length ({len(candidate_phonemes)}) must match "
            f"candidates length ({len(candidates)})"
        )

    # --- Phonemize if needed ---
    if candidate_phonemes is None:
        from corpusgen.g2p.manager import G2PManager

        g2p = G2PManager()
        g2p_results = g2p.phonemize_batch(candidates, language=language)
        candidate_phonemes = [r.phonemes for r in g2p_results]

    # --- Resolve target_phonemes ---
    if isinstance(target_phonemes, str) and target_phonemes.lower() == "phoible":
        from corpusgen.inventory.phoible import PhoibleDataset

        ds = PhoibleDataset()
        inv = ds.get_inventory_for_espeak(language)
        target_phonemes = inv.phonemes

    if target_phonemes is None:
        # Derive from all unique phonemes in candidates
        all_unique: set[str] = set()
        for phonemes in candidate_phonemes:
            all_unique.update(phonemes)
        target_phonemes = sorted(all_unique)

    # --- Build target unit set ---
    from corpusgen.coverage.tracker import CoverageTracker

    tracker = CoverageTracker(target_phonemes=target_phonemes, unit=unit)
    target_units = tracker.target_units

    # --- Instantiate the selector ---
    selector = _make_selector(algorithm, unit, **algorithm_kwargs)

    # --- Run selection ---
    return selector.select(
        candidates=candidates,
        candidate_phonemes=candidate_phonemes,
        target_units=target_units,
        max_sentences=max_sentences,
        target_coverage=target_coverage,
        weights=weights,
    )


def _make_selector(algorithm: str, unit: str, **kwargs) -> SelectorBase:
    """Instantiate a selector by algorithm name.

    Args:
        algorithm: Algorithm name string.
        unit: Coverage unit type.
        **kwargs: Forwarded to the selector constructor.

    Returns:
        An instance of the requested SelectorBase subclass.

    Raises:
        ValueError: If algorithm name is unknown.
        ImportError: If the algorithm requires an uninstalled package.
    """
    if algorithm in _CORE_ALGORITHMS:
        cls = _CORE_ALGORITHMS[algorithm]
        return cls(unit=unit, **kwargs)

    if algorithm in _OPTIONAL_ALGORITHMS:
        class_name, pkg, install_hint = _OPTIONAL_ALGORITHMS[algorithm]
        # Check if the class is available in this module's namespace
        cls = globals().get(class_name)
        if cls is None:
            raise ImportError(
                f"Algorithm {algorithm!r} requires {pkg}. "
                f"Install with: pip install {install_hint}"
            )
        return cls(unit=unit, **kwargs)

    all_algorithms = sorted(set(_CORE_ALGORITHMS) | set(_OPTIONAL_ALGORITHMS))
    raise ValueError(
        f"Unknown algorithm: {algorithm!r}. "
        f"Must be one of {all_algorithms}"
    )


__all__ = [
    "SelectorBase",
    "SelectionResult",
    "GreedySelector",
    "CELFSelector",
    "StochasticGreedySelector",
    "DistributionAwareSelector",
    "select_sentences",
]
if ILPSelector is not None:
    __all__.append("ILPSelector")
if NSGA2Selector is not None:
    __all__.append("NSGA2Selector")
