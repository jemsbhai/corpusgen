"""ILPSelector: exact Integer Linear Programming solver for optimal selection."""

from __future__ import annotations

import math
import time
import warnings
from typing import Any

from corpusgen.select.base import SelectorBase
from corpusgen.select.result import SelectionResult

try:
    import pulp
except ImportError:
    pulp = None  # type: ignore[assignment]


def _add_binary_variable(prob: Any, name: str) -> Any:
    """Create a binary variable through the newest available PuLP API."""
    add_variable = getattr(prob, "add_variable", None)
    if callable(add_variable):
        return add_variable(name, cat=pulp.LpBinary)
    return pulp.LpVariable(name, cat=pulp.LpBinary)


def _create_cbc_solver(time_limit: float | None) -> Any:
    """Return an available CBC solver across supported PuLP versions."""
    solver_kwargs = {"msg": False, "timeLimit": time_limit}

    # PuLP 3.3 deprecates its bundled CBC wrapper in favour of COIN_CMD.
    # COIN_CMD only works when CBC was installed separately (for example via
    # ``pulp[cbc]``), so check availability before choosing it.
    coin_solver_type = getattr(pulp, "COIN_CMD", None)
    if coin_solver_type is not None:
        coin_solver = coin_solver_type(**solver_kwargs)
        if coin_solver.available():
            return coin_solver

    # PuLP 2.x and installations without an external CBC still rely on the
    # bundled wrapper. Suppress only its PuLP 3.3 migration warning; passing
    # the solver explicitly prevents LpProblem.solve() from warning again.
    bundled_solver_type = getattr(pulp, "PULP_CBC_CMD", None)
    if bundled_solver_type is None:
        raise RuntimeError("No CBC solver is available. Install CBC with 'pip install pulp[cbc]'.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^PULP_CBC_CMD is deprecated and will be removed in PuLP 4\.0\.",
            category=DeprecationWarning,
        )
        return bundled_solver_type(**solver_kwargs)


class ILPSelector(SelectorBase):
    """Exact Set Cover solver via Integer Linear Programming.

    Formulates sentence selection as a binary ILP: minimize the number
    of selected sentences subject to every target unit being covered
    by at least one selected sentence. Provides provably optimal
    solutions for benchmarking approximation algorithms.

    Practical for small-to-medium problems (<10,000 candidates).
    Falls back gracefully with an informative error if PuLP is not
    installed.

    Requires: ``pip install corpusgen[optimization]`` or ``pip install pulp``.

    Args:
        unit: Coverage unit type.
        time_limit: Maximum solver time in seconds (default: 300).
            None means no time limit.
    """

    def __init__(
        self,
        unit: str = "phoneme",
        time_limit: float | None = 300.0,
    ) -> None:
        super().__init__(unit=unit)
        if pulp is None:
            raise ImportError(
                "ILPSelector requires PuLP. Install with: "
                "pip install corpusgen[optimization] or pip install pulp"
            )
        self._time_limit = time_limit

    @property
    def algorithm_name(self) -> str:
        return "ilp"

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
        self._validate_select_inputs(
            candidates, candidate_phonemes, max_sentences, target_coverage, weights
        )

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
                metadata={"solver_status": "Optimal"},
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
                metadata={"solver_status": "Infeasible"},
            )

        # Precompute units per candidate (only those in target)
        candidate_units: list[set[str]] = []
        for phonemes in candidate_phonemes:
            units = self._extract_units(phonemes)
            candidate_units.append(units & target_units)

        # Determine which units are actually coverable
        coverable = set()
        for cu in candidate_units:
            coverable |= cu
        uncoverable = target_units - coverable

        # Determine required coverage count
        target_count = len(target_units)
        required_covered = int(math.ceil(target_coverage * target_count))
        # Cap at what's actually coverable
        coverable_target = coverable & target_units
        required_covered = min(required_covered, len(coverable_target))

        # Build index: unit → list of candidate indices that cover it
        unit_list = sorted(coverable_target)
        unit_to_idx = {u: i for i, u in enumerate(unit_list)}
        n_candidates = len(candidates)

        # --- ILP formulation ---
        prob = pulp.LpProblem("SetCover", pulp.LpMinimize)

        # Binary decision variables: x[i] = 1 if candidate i is selected
        x = [
            _add_binary_variable(prob, f"x_{i}")
            for i in range(n_candidates)
        ]

        # Objective: minimize number of selected sentences
        prob += pulp.lpSum(x)

        # Budget constraint
        if max_sentences is not None:
            prob += pulp.lpSum(x) <= max_sentences

        # If target_coverage < 1.0, we use auxiliary variables for partial cover
        if required_covered < len(coverable_target):
            # Binary variable y[j] = 1 if unit j is covered
            y = [
                _add_binary_variable(prob, f"y_{j}")
                for j in range(len(unit_list))
            ]

            # y[j] can only be 1 if some selected candidate covers unit j
            for j, unit in enumerate(unit_list):
                covering = [
                    i for i in range(n_candidates)
                    if unit in candidate_units[i]
                ]
                if covering:
                    prob += y[j] <= pulp.lpSum(x[i] for i in covering)
                else:
                    prob += y[j] == 0

            # Must cover at least required_covered weighted units
            if weights is not None:
                weighted_sum = pulp.lpSum(
                    y[j] * weights.get(unit_list[j], 1.0)
                    for j in range(len(unit_list))
                )
                # Require weighted sum >= required fraction of total weight
                total_weight = sum(
                    weights.get(u, 1.0) for u in coverable_target
                )
                prob += weighted_sum >= target_coverage * total_weight
            else:
                prob += pulp.lpSum(y) >= required_covered
        else:
            # Full coverage of coverable units: every unit must be covered
            for unit in unit_list:
                covering = [
                    i for i in range(n_candidates)
                    if unit in candidate_units[i]
                ]
                if covering:
                    prob += pulp.lpSum(x[i] for i in covering) >= 1

        # Solve
        solver = _create_cbc_solver(self._time_limit)
        prob.solve(solver)

        solver_status = pulp.LpStatus[prob.status]

        # Extract solution — only if solver found a feasible solution
        selected_indices = []
        if prob.status == pulp.constants.LpStatusOptimal:
            for i in range(n_candidates):
                if x[i].varValue is not None and x[i].varValue > 0.5:
                    selected_indices.append(i)

        # Compute actual coverage
        covered: set[str] = set()
        for i in selected_indices:
            covered |= candidate_units[i]

        elapsed = time.perf_counter() - start
        coverage = len(covered) / target_count if target_count > 0 else 1.0

        return SelectionResult(
            selected_indices=selected_indices,
            selected_sentences=[candidates[i] for i in selected_indices],
            coverage=coverage,
            covered_units=set(covered),
            missing_units=target_units - covered,
            unit=self.unit,
            algorithm=self.algorithm_name,
            elapsed_seconds=elapsed,
            iterations=1,
            metadata={
                "solver_status": solver_status,
            },
        )
