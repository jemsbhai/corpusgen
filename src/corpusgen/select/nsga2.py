"""NSGA2Selector: multi-objective Pareto optimization for sentence selection."""

from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np

from corpusgen.select.base import SelectorBase
from corpusgen.select.result import SelectionResult

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination
except ImportError:
    NSGA2 = None  # type: ignore[assignment, misc]

    class Problem:  # type: ignore[no-redef]
        """Stub when pymoo is not installed."""

        pass


class _CoverageSelectionProblem(Problem):
    """pymoo Problem: binary selection with multiple objectives.

    Objectives (all minimized):
        1. -coverage (negative, since pymoo minimizes)
        2. n_sentences (sentence count)
        3. kl_divergence (optional, if target_distribution given)

    Constraints:
        - max_sentences budget (if specified)
    """

    def __init__(
        self,
        candidate_units: list[set[str]],
        candidate_unit_lists: list[list[str]] | None,
        target_units: set[str],
        target_distribution: dict[str, float] | None,
        max_sentences: int | None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.candidate_units = candidate_units
        self.candidate_unit_lists = candidate_unit_lists
        self.target_units = target_units
        self.target_count = len(target_units)
        self.target_distribution = target_distribution
        self.max_sentences = max_sentences
        self.weights = weights
        # Precompute total weight for normalisation
        if weights is not None:
            self.total_weight = sum(weights.get(u, 1.0) for u in target_units)
        else:
            self.total_weight = float(len(target_units))

        n_obj = 3 if target_distribution else 2
        n_ieq_constr = 1 if max_sentences is not None else 0

        super().__init__(
            n_var=len(candidate_units),
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # X is (pop_size, n_candidates) binary matrix
        pop_size = X.shape[0]
        f = np.zeros((pop_size, self.n_obj))
        g = np.zeros((pop_size, self.n_ieq_constr)) if self.n_ieq_constr > 0 else None

        for i in range(pop_size):
            selected = np.where(X[i] > 0.5)[0]
            n_selected = len(selected)

            # Objective 1: -coverage (weighted if weights provided)
            covered = set()
            for idx in selected:
                covered |= self.candidate_units[idx]
            if self.weights is not None:
                wcov = sum(self.weights.get(u, 1.0) for u in covered)
                coverage = wcov / self.total_weight if self.total_weight > 0 else 1.0
            else:
                coverage = len(covered) / self.target_count if self.target_count > 0 else 1.0
            f[i, 0] = -coverage

            # Objective 2: sentence count
            f[i, 1] = n_selected

            # Objective 3 (optional): KL-divergence
            if self.target_distribution is not None and self.candidate_unit_lists is not None:
                counts: Counter[str] = Counter()
                for idx in selected:
                    counts.update(
                        u for u in self.candidate_unit_lists[idx]
                        if u in self.target_units
                    )
                f[i, 2] = self._kl_divergence(counts)

            # Constraint: max sentences
            if g is not None:
                g[i, 0] = n_selected - self.max_sentences

        out["F"] = f
        if g is not None:
            out["G"] = g

    def _kl_divergence(self, counts: Counter[str]) -> float:
        total = sum(counts.values())
        if total == 0:
            return 1e6  # Large penalty for empty selection
        units = set(self.target_distribution.keys())
        smoothed_total = total + len(units)
        kl = 0.0
        for u in units:
            p = self.target_distribution[u]
            q = (counts.get(u, 0) + 1) / smoothed_total
            if p > 0:
                kl += p * math.log(p / q)
        return kl


class NSGA2Selector(SelectorBase):
    """Multi-objective Pareto selector using NSGA-II.

    Simultaneously optimizes:
        1. Maximize phoneme coverage
        2. Minimize sentence count
        3. Minimize KL-divergence to target distribution (optional)

    Returns the Pareto-front solution with highest coverage (tie-break:
    fewest sentences). The full Pareto front is available in
    ``result.metadata["pareto_front"]``.

    Reference:
        Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
        A fast and elitist multiobjective genetic algorithm: NSGA-II.
        IEEE Trans. Evolutionary Computation.

    Requires: ``pip install corpusgen[optimization]`` or ``pip install pymoo``.

    Args:
        unit: Coverage unit type.
        target_distribution: Optional target frequency distribution.
            If provided, KL-divergence becomes the third objective.
        population_size: GA population size (default: 50).
        n_generations: Number of generations (default: 100).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        unit: str = "phoneme",
        target_distribution: dict[str, float] | None = None,
        population_size: int = 50,
        n_generations: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__(unit=unit)
        if NSGA2 is None:
            raise ImportError(
                "NSGA2Selector requires pymoo. Install with: "
                "pip install corpusgen[optimization] or pip install pymoo"
            )
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if n_generations < 1:
            raise ValueError("n_generations must be >= 1")

        self._target_distribution = None
        if target_distribution is not None:
            total = sum(target_distribution.values())
            self._target_distribution = {
                k: v / total for k, v in target_distribution.items()
            }

        self._population_size = population_size
        self._n_generations = n_generations
        self._seed = seed

    @property
    def algorithm_name(self) -> str:
        return "nsga2"

    def select(
        self,
        candidates: list[str],
        candidate_phonemes: list[list[str]],
        target_units: set[str],
        max_sentences: int | None = None,
        target_coverage: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> SelectionResult:
        start = time.perf_counter()

        # Edge case: empty target
        if not target_units:
            return SelectionResult(
                selected_indices=[],
                selected_sentences=[],
                coverage=1.0,
                covered_units=set(),
                missing_units=set(),
                unit=self.unit,
                algorithm=self.algorithm_name,
                elapsed_seconds=time.perf_counter() - start,
                iterations=0,
                metadata={"pareto_front": []},
            )

        # Edge case: no candidates
        if not candidates:
            return SelectionResult(
                selected_indices=[],
                selected_sentences=[],
                coverage=0.0,
                covered_units=set(),
                missing_units=set(target_units),
                unit=self.unit,
                algorithm=self.algorithm_name,
                elapsed_seconds=time.perf_counter() - start,
                iterations=0,
                metadata={"pareto_front": []},
            )

        # Precompute units per candidate
        candidate_units: list[set[str]] = []
        candidate_unit_lists: list[list[str]] | None = None
        if self._target_distribution is not None:
            candidate_unit_lists = []

        for phonemes in candidate_phonemes:
            unit_set = self._extract_units(phonemes) & target_units
            candidate_units.append(unit_set)
            if candidate_unit_lists is not None:
                candidate_unit_lists.append(self._extract_unit_list(phonemes))

        # Build pymoo problem
        problem = _CoverageSelectionProblem(
            candidate_units=candidate_units,
            candidate_unit_lists=candidate_unit_lists,
            target_units=target_units,
            target_distribution=self._target_distribution,
            max_sentences=max_sentences,
            weights=weights,
        )

        # Configure NSGA-II
        algorithm = NSGA2(
            pop_size=self._population_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(prob=0.9),
            mutation=BitflipMutation(prob=1.0 / len(candidates)),
            eliminate_duplicates=True,
        )

        termination = get_termination("n_gen", self._n_generations)

        res = pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=self._seed,
            verbose=False,
        )

        # Extract Pareto front
        pareto_front = []
        target_count = len(target_units)

        if res.X is not None:
            # Ensure 2D array (single solution edge case)
            X = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)

            for sol in X:
                indices = list(np.where(sol > 0.5)[0])
                covered = set()
                for idx in indices:
                    covered |= candidate_units[idx]
                cov = len(covered) / target_count if target_count > 0 else 1.0

                entry = {
                    "coverage": cov,
                    "n_sentences": len(indices),
                    "selected_indices": [int(i) for i in indices],
                }

                if self._target_distribution is not None and candidate_unit_lists is not None:
                    counts: Counter[str] = Counter()
                    for idx in indices:
                        counts.update(
                            u for u in candidate_unit_lists[idx]
                            if u in target_units
                        )
                    entry["kl_divergence"] = problem._kl_divergence(counts)

                pareto_front.append(entry)

        # Select best solution: highest coverage, then fewest sentences
        if pareto_front:
            best = max(
                pareto_front,
                key=lambda e: (e["coverage"], -e["n_sentences"]),
            )
            best_indices = best["selected_indices"]
        else:
            best_indices = []

        # Compute final coverage
        covered = set()
        for idx in best_indices:
            covered |= candidate_units[idx]
        coverage = len(covered) / target_count if target_count > 0 else 1.0

        elapsed = time.perf_counter() - start

        return SelectionResult(
            selected_indices=best_indices,
            selected_sentences=[candidates[i] for i in best_indices],
            coverage=coverage,
            covered_units=set(covered),
            missing_units=target_units - covered,
            unit=self.unit,
            algorithm=self.algorithm_name,
            elapsed_seconds=elapsed,
            iterations=self._n_generations,
            metadata={"pareto_front": pareto_front},
        )

