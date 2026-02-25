"""Tests for selection algorithm foundational types: SelectionResult and SelectorBase."""

from __future__ import annotations

import pytest

from corpusgen.select.result import SelectionResult
from corpusgen.select.base import SelectorBase


# ── SelectionResult tests ──────────────────────────────────────────────


class TestSelectionResult:
    """Tests for the SelectionResult dataclass."""

    def _make_result(self, **overrides) -> SelectionResult:
        """Helper to build a SelectionResult with sensible defaults."""
        defaults = dict(
            selected_indices=[0, 3, 7],
            selected_sentences=["hello world", "foo bar", "baz qux"],
            coverage=0.85,
            covered_units={"a", "b", "c"},
            missing_units={"d"},
            unit="phoneme",
            algorithm="greedy",
            elapsed_seconds=0.123,
            iterations=42,
            metadata={},
        )
        defaults.update(overrides)
        return SelectionResult(**defaults)

    def test_creation(self):
        result = self._make_result()
        assert result.selected_indices == [0, 3, 7]
        assert result.selected_sentences == ["hello world", "foo bar", "baz qux"]
        assert result.coverage == 0.85
        assert result.covered_units == {"a", "b", "c"}
        assert result.missing_units == {"d"}
        assert result.unit == "phoneme"
        assert result.algorithm == "greedy"
        assert result.elapsed_seconds == 0.123
        assert result.iterations == 42
        assert result.metadata == {}

    def test_frozen(self):
        result = self._make_result()
        with pytest.raises(AttributeError):
            result.coverage = 0.99  # type: ignore[misc]

    def test_num_selected(self):
        result = self._make_result()
        assert result.num_selected == 3

    def test_num_selected_empty(self):
        result = self._make_result(selected_indices=[], selected_sentences=[])
        assert result.num_selected == 0

    def test_metadata_stored(self):
        meta = {"solver_status": "Optimal", "gap": 0.0}
        result = self._make_result(metadata=meta)
        assert result.metadata["solver_status"] == "Optimal"

    def test_all_unit_types(self):
        for unit in ("phoneme", "diphone", "triphone"):
            result = self._make_result(unit=unit)
            assert result.unit == unit

    def test_all_algorithm_names(self):
        for algo in ("greedy", "celf", "ilp", "stochastic", "distribution", "nsga2"):
            result = self._make_result(algorithm=algo)
            assert result.algorithm == algo


# ── SelectorBase tests ─────────────────────────────────────────────────


class DummySelector(SelectorBase):
    """Minimal concrete implementation for testing the ABC."""

    @property
    def algorithm_name(self) -> str:
        return "dummy"

    def select(
        self,
        candidates: list[str],
        candidate_phonemes: list[list[str]],
        target_units: set[str],
        max_sentences: int | None = None,
        target_coverage: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> SelectionResult:
        return SelectionResult(
            selected_indices=[],
            selected_sentences=[],
            coverage=0.0,
            covered_units=set(),
            missing_units=target_units,
            unit=self.unit,
            algorithm=self.algorithm_name,
            elapsed_seconds=0.0,
            iterations=0,
            metadata={},
        )


class TestSelectorBase:
    """Tests for the SelectorBase abstract class."""

    def test_instantiation(self):
        selector = DummySelector(unit="phoneme")
        assert selector.unit == "phoneme"
        assert selector.algorithm_name == "dummy"

    def test_default_unit(self):
        selector = DummySelector()
        assert selector.unit == "phoneme"

    def test_diphone_unit(self):
        selector = DummySelector(unit="diphone")
        assert selector.unit == "diphone"

    def test_triphone_unit(self):
        selector = DummySelector(unit="triphone")
        assert selector.unit == "triphone"

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid unit"):
            DummySelector(unit="quadphone")

    def test_select_returns_result(self):
        selector = DummySelector()
        result = selector.select(
            candidates=["hello"],
            candidate_phonemes=[["h", "ɛ", "l", "oʊ"]],
            target_units={"h", "ɛ", "l", "oʊ"},
        )
        assert isinstance(result, SelectionResult)
        assert result.algorithm == "dummy"
        assert result.unit == "phoneme"

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            SelectorBase()  # type: ignore[abstract]

    def test_extract_units_phoneme(self):
        selector = DummySelector(unit="phoneme")
        assert selector._extract_units(["a", "b", "c"]) == {"a", "b", "c"}

    def test_extract_units_diphone(self):
        selector = DummySelector(unit="diphone")
        assert selector._extract_units(["a", "b", "c"]) == {"a-b", "b-c"}

    def test_extract_units_triphone(self):
        selector = DummySelector(unit="triphone")
        assert selector._extract_units(["a", "b", "c"]) == {"a-b-c"}

    def test_extract_units_empty(self):
        selector = DummySelector(unit="phoneme")
        assert selector._extract_units([]) == set()

    def test_extract_units_single_phoneme_diphone(self):
        selector = DummySelector(unit="diphone")
        assert selector._extract_units(["a"]) == set()

    def test_extract_units_two_phonemes_triphone(self):
        selector = DummySelector(unit="triphone")
        assert selector._extract_units(["a", "b"]) == set()

    def test_extract_unit_list_phoneme(self):
        selector = DummySelector(unit="phoneme")
        assert selector._extract_unit_list(["a", "a", "b"]) == ["a", "a", "b"]

    def test_extract_unit_list_diphone(self):
        selector = DummySelector(unit="diphone")
        assert selector._extract_unit_list(["a", "b", "a"]) == ["a-b", "b-a"]

    def test_extract_unit_list_triphone(self):
        selector = DummySelector(unit="triphone")
        assert selector._extract_unit_list(["a", "b", "c", "a"]) == ["a-b-c", "b-c-a"]

    def test_extract_unit_list_empty(self):
        selector = DummySelector(unit="phoneme")
        assert selector._extract_unit_list([]) == []
