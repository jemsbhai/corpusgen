"""evaluate(): top-level API that wires G2P + CoverageTracker + EvaluationReport."""

from __future__ import annotations

from corpusgen.coverage.tracker import CoverageTracker
from corpusgen.evaluate.report import EvaluationReport, SentenceDetail
from corpusgen.g2p.manager import G2PManager
from corpusgen.g2p.result import G2PResult


def evaluate(
    sentences: list[str],
    language: str = "en-us",
    target_phonemes: list[str] | str | None = None,
    unit: str = "phoneme",
) -> EvaluationReport:
    """Evaluate a corpus of sentences for phoneme coverage.

    This is the primary user-facing API. It phonemizes the input sentences,
    tracks coverage against a target phoneme inventory, and returns a
    structured report.

    Args:
        sentences: List of text sentences to evaluate.
        language: Language code for G2P conversion (e.g., 'en-us', 'fr-fr').
        target_phonemes: Target phoneme inventory to measure coverage against.
            If None, the inventory is derived from all unique phonemes found
            in the corpus (resulting in 100% coverage — useful for inventory
            discovery). If the string ``"phoible"``, the target is
            automatically fetched from the PHOIBLE database for the given
            language (requires cached PHOIBLE data).
        unit: Coverage unit type — "phoneme", "diphone", or "triphone".

    Returns:
        EvaluationReport with coverage metrics, per-sentence details,
        phoneme counts, and source provenance.

    Raises:
        ValueError: If *unit* is not one of "phoneme", "diphone", "triphone".
    """
    valid_units = ("phoneme", "diphone", "triphone")
    if unit not in valid_units:
        raise ValueError(
            f"Invalid unit: {unit!r}. Must be one of {valid_units}"
        )

    # --- Step 0: Resolve "phoible" shortcut ---
    if isinstance(target_phonemes, str) and target_phonemes.lower() == "phoible":
        from corpusgen.inventory.phoible import PhoibleDataset
        ds = PhoibleDataset()
        inv = ds.get_inventory_for_espeak(language)
        target_phonemes = inv.phonemes

    # --- Step 1: Phonemize all sentences ---
    g2p = G2PManager()
    g2p_results: list[G2PResult] = g2p.phonemize_batch(sentences, language=language)

    # --- Step 2: Derive target inventory if not provided ---
    if target_phonemes is None:
        all_unique: set[str] = set()
        for result in g2p_results:
            all_unique.update(result.phonemes)
        target_phonemes = sorted(all_unique)

    # Handle empty corpus + None target: target_phonemes is [] → coverage 1.0
    # Handle empty target list: coverage 1.0 by CoverageTracker convention

    # --- Step 3: Track coverage ---
    tracker = CoverageTracker(target_phonemes=target_phonemes, unit=unit)

    # For phoneme unit, preserve the user's original ordering.
    # For diphone/triphone, the target is a combinatorial set — sort it.
    if unit == "phoneme":
        target_units_list = list(target_phonemes)
    else:
        target_units_list = sorted(tracker._target_set)

    # Track which target-units are first seen per sentence (for new_phonemes)
    seen_target_units: set[str] = set()

    sentence_details: list[SentenceDetail] = []

    for idx, (sentence, g2p_result) in enumerate(zip(sentences, g2p_results)):
        phonemes = g2p_result.phonemes

        # Compute the units this sentence contains
        if unit == "phoneme":
            sentence_units = phonemes
        elif unit == "diphone":
            sentence_units = [
                f"{phonemes[i]}-{phonemes[i + 1]}"
                for i in range(len(phonemes) - 1)
            ]
        elif unit == "triphone":
            sentence_units = [
                f"{phonemes[i]}-{phonemes[i + 1]}-{phonemes[i + 2]}"
                for i in range(len(phonemes) - 2)
            ]
        else:
            sentence_units = []

        # Update the tracker
        tracker.update(phonemes, sentence_index=idx)

        # Determine which target-units are NEW from this sentence
        target_set = tracker._target_set
        new_units = []
        for u in sentence_units:
            if u in target_set and u not in seen_target_units:
                new_units.append(u)
                seen_target_units.add(u)

        # Deduplicate new_units while preserving order
        seen_in_new: set[str] = set()
        new_units_deduped: list[str] = []
        for u in new_units:
            if u not in seen_in_new:
                new_units_deduped.append(u)
                seen_in_new.add(u)

        sentence_details.append(
            SentenceDetail(
                index=idx,
                text=sentence,
                phoneme_count=len(phonemes),
                new_phonemes=new_units_deduped,
                all_phonemes=list(phonemes),
            )
        )

    # --- Step 4: Assemble report ---
    return EvaluationReport(
        language=language,
        unit=unit,
        target_phonemes=target_units_list,
        covered_phonemes=set(tracker._covered),
        missing_phonemes=set(tracker.missing),
        coverage=tracker.coverage,
        phoneme_counts=tracker.phoneme_counts,
        total_sentences=len(sentences),
        sentence_details=sentence_details,
        phoneme_sources=tracker.phoneme_sources,
    )
