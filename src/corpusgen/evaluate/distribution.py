"""Distribution quality metrics for phoneme corpus evaluation.

Quantifies how well-balanced a corpus's phoneme distribution is, going
beyond binary coverage to measure distributional quality.  Implements
metrics grounded in information theory and statistics that are standard
in the speech corpus literature.

Metrics provided:

* **Shannon entropy** and **normalized entropy** — measure how spread
  out the distribution is.  Normalized entropy of 1.0 means perfectly
  uniform across all target units.
* **Jensen-Shannon Divergence (JSD)** — symmetric, bounded [0, 1]
  divergence between the corpus distribution and a reference (uniform
  by default, or user-supplied).  Uses base-2 logarithms so the range
  is exactly [0, 1].  Recommended over KL-divergence for corpus work
  because it is symmetric and always finite.
* **Pearson correlation** — correlation between corpus unit frequencies
  and a reference distribution.  Values ≥ 0.99 indicate the corpus
  preserves natural language statistics (see Hindi code-mixed corpus
  design achieving r = 0.996).
* **Coefficient of variation (CV)** — σ/μ of counts across target
  units.  0.0 = perfectly balanced.
* **Min/max statistics** — extremes and their ratio.
* **PCD_uniform** — Phoneme Coverage Diversity composite:
  ``coverage × (1 - JSD_uniform)``.  Combines coverage completeness
  with distributional quality in a single score.

Future work (not yet implemented):
    * Full Phonetic Completeness Score (PCS) requiring syllable and
      positional coverage tracking.
    * Full Phonological Contextual Diversity (PCD) requiring word-
      position, stress, and prosodic context tracking.

References:
    Lin, J. (1991). Divergence measures based on the Shannon entropy.
        IEEE Transactions on Information Theory, 37(1), 145–151.
    Amrouche, A., et al. (2021). Balanced Arabic corpus design for
        speech synthesis. (distribution alignment methodology)
    Gupta, V., et al. (2020). Hindi code-mixed speech corpus with
        Pearson's r = 0.996 triphone correlation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class DistributionMetrics:
    """Immutable container for corpus distribution quality metrics.

    All metrics are computed over target phonetic units only; counts for
    units outside the target inventory are ignored.

    Attributes:
        entropy: Shannon entropy H(X) in bits (base-2 log).
            0.0 when all mass on one unit; log₂(N) when perfectly uniform.
        normalized_entropy: H(X) / log₂(N).  1.0 = perfectly uniform.
            Defined as 1.0 when N ≤ 1 (trivially uniform by convention).
        jsd_uniform: Jensen-Shannon Divergence vs. a uniform distribution
            over all target units.  0.0 = perfectly uniform; 1.0 = maximally
            divergent.  Uses base-2 log so the range is exactly [0, 1].
        coefficient_of_variation: Population standard deviation / mean of
            counts across *all* target units (including zeros).
            0.0 = all counts equal.
        min_count: Smallest count among target units (0 if any are missing).
        max_count: Largest count among target units.
        count_ratio: min_count / max_count.  0.0 if min is 0; 1.0 if
            all counts equal.  Defined as 1.0 when max_count is 0.
        zero_count: Number of target units with zero occurrences.
        pcd_uniform: Phoneme Coverage Diversity (uniform reference):
            ``coverage × (1 - jsd_uniform)`` where coverage is the
            fraction of target units with count > 0.
        jsd_reference: JSD vs. a user-supplied reference distribution.
            None when no reference is provided.
        pearson_correlation: Pearson's r between corpus counts and a
            reference distribution.  None when no reference is provided
            or when either distribution has zero variance (undefined).
    """

    entropy: float
    normalized_entropy: float
    jsd_uniform: float
    coefficient_of_variation: float
    min_count: int
    max_count: int
    count_ratio: float
    zero_count: int
    pcd_uniform: float
    jsd_reference: float | None
    pearson_correlation: float | None

    def to_dict(self) -> dict[str, Any]:
        """Export as a plain Python dict (JSON-safe)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kl_divergence(p: list[float], q: list[float]) -> float:
    """Compute KL(P || Q) using base-2 logarithms.

    Uses the convention 0 · log(0/q) = 0 (the mathematical limit).
    Requires q[i] > 0 whenever p[i] > 0.

    Args:
        p: Probability distribution P (sums to 1).
        q: Probability distribution Q (sums to 1).

    Returns:
        KL divergence in bits.
    """
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            total += pi * math.log2(pi / qi)
    return total


def _jsd(p: list[float], q: list[float]) -> float:
    """Compute Jensen-Shannon Divergence using base-2 logarithms.

    JSD(P || Q) = ½ KL(P || M) + ½ KL(Q || M)
    where M = ½(P + Q).

    Bounded [0, 1] with base-2 log.

    Args:
        p: Probability distribution P.
        q: Probability distribution Q.

    Returns:
        JSD in [0, 1].
    """
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _pearson_r(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient.

    Returns None if either vector has zero variance (undefined).

    Args:
        x: First vector of values.
        y: Second vector of values (same length as x).

    Returns:
        Pearson's r in [-1, 1], or None if undefined.
    """
    n = len(x)
    if n < 2:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Numerator: covariance (unnormalized)
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    # Denominators: standard deviations (unnormalized)
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    if var_x == 0.0 or var_y == 0.0:
        return None

    return cov / math.sqrt(var_x * var_y)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_distribution_metrics(
    phoneme_counts: dict[str, int],
    target_phonemes: list[str],
    reference_distribution: dict[str, float] | None = None,
) -> DistributionMetrics:
    """Compute distribution quality metrics for a phoneme corpus.

    Measures how well-balanced the corpus's phoneme distribution is
    across the target inventory, using information-theoretic and
    statistical metrics.

    Args:
        phoneme_counts: Mapping from phonetic unit to its occurrence
            count in the corpus.  Units not in ``target_phonemes`` are
            ignored.
        target_phonemes: The target inventory of phonetic units.  Only
            these units are considered.  May contain duplicates (they
            are deduplicated internally while preserving order).
        reference_distribution: Optional mapping from phonetic unit to
            its expected relative frequency.  Need not sum to 1.0 —
            values are normalized automatically.  Units in
            ``target_phonemes`` but missing from the reference are
            treated as 0.  When provided, ``jsd_reference`` and
            ``pearson_correlation`` are computed.

    Returns:
        DistributionMetrics with all computed fields.
    """
    # --- Deduplicate targets preserving order ---
    seen: set[str] = set()
    targets: list[str] = []
    for t in target_phonemes:
        if t not in seen:
            targets.append(t)
            seen.add(t)

    n = len(targets)

    # --- Extract counts for target units only ---
    counts = [phoneme_counts.get(t, 0) for t in targets]
    total = sum(counts)

    # --- Basic count statistics ---
    if n == 0:
        min_count = 0
        max_count = 0
        zero_count_val = 0
    else:
        min_count = min(counts)
        max_count = max(counts)
        zero_count_val = sum(1 for c in counts if c == 0)

    # count_ratio: min/max, with conventions for degenerate cases
    if max_count == 0:
        count_ratio = 1.0 if n == 0 else 0.0
    else:
        count_ratio = min_count / max_count

    # --- Corpus probability distribution ---
    if total > 0:
        corpus_dist = [c / total for c in counts]
    else:
        # All-zero corpus: no valid probability distribution
        corpus_dist = [0.0] * n

    # --- Coverage ---
    covered = sum(1 for c in counts if c > 0)
    coverage = covered / n if n > 0 else 1.0

    # --- Shannon entropy (bits) ---
    entropy = 0.0
    for p in corpus_dist:
        if p > 0:
            entropy -= p * math.log2(p)

    # --- Normalized entropy ---
    if n <= 1:
        normalized_entropy = 1.0  # trivially uniform by convention
    elif total == 0:
        normalized_entropy = 0.0  # no information
    else:
        max_entropy = math.log2(n)
        normalized_entropy = entropy / max_entropy

    # --- JSD vs uniform ---
    if n == 0:
        jsd_uniform = 0.0  # trivially no divergence
    elif total == 0:
        jsd_uniform = 1.0  # maximal divergence: empty vs uniform
    else:
        uniform_dist = [1.0 / n] * n
        jsd_uniform = _jsd(corpus_dist, uniform_dist)

    # --- Coefficient of variation ---
    if n == 0 or total == 0:
        cv = 0.0
    else:
        mean_count = total / n
        variance = sum((c - mean_count) ** 2 for c in counts) / n
        stdev = math.sqrt(variance)
        cv = stdev / mean_count if mean_count > 0 else 0.0

    # --- PCD_uniform ---
    pcd_uniform = coverage * (1.0 - jsd_uniform)

    # --- Reference-based metrics ---
    jsd_ref: float | None = None
    pearson: float | None = None

    if reference_distribution is not None and n > 0:
        # Build reference vector aligned to targets, fill missing with 0
        ref_raw = [reference_distribution.get(t, 0.0) for t in targets]
        ref_total = sum(ref_raw)

        if ref_total > 0:
            ref_dist = [r / ref_total for r in ref_raw]
        else:
            ref_dist = [0.0] * n

        # JSD vs reference
        if total == 0:
            # Empty corpus vs any reference → maximal divergence
            jsd_ref = 1.0
        elif ref_total == 0:
            # Empty reference vs any corpus → maximal divergence
            jsd_ref = 1.0
        else:
            jsd_ref = _jsd(corpus_dist, ref_dist)

        # Pearson correlation (on counts vs reference probabilities)
        # Using the corpus count vector and reference probability vector
        pearson = _pearson_r(
            [float(c) for c in counts],
            ref_dist,
        )

    return DistributionMetrics(
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        jsd_uniform=jsd_uniform,
        coefficient_of_variation=cv,
        min_count=min_count,
        max_count=max_count,
        count_ratio=count_ratio,
        zero_count=zero_count_val,
        pcd_uniform=pcd_uniform,
        jsd_reference=jsd_ref,
        pearson_correlation=pearson,
    )
