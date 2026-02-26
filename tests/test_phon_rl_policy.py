"""Tests for phon_rl.policy — PhonRLStrategy GuidanceStrategy wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from corpusgen.generate.guidance import GuidanceStrategy
from corpusgen.generate.phon_rl.policy import PhonRLStrategy


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture()
def mock_model() -> MagicMock:
    """Mock HuggingFace model."""
    model = MagicMock()
    model.device = "cpu"
    return model


@pytest.fixture()
def mock_tokenizer() -> MagicMock:
    """Mock HuggingFace tokenizer."""
    return MagicMock()


@pytest.fixture()
def strategy_no_adapter() -> PhonRLStrategy:
    """Strategy with no adapter path (base model pass-through)."""
    return PhonRLStrategy()


@pytest.fixture()
def strategy_with_adapter() -> PhonRLStrategy:
    """Strategy with an adapter path configured."""
    return PhonRLStrategy(adapter_path="/fake/adapter/path")


# -----------------------------------------------------------------------
# ABC compliance
# -----------------------------------------------------------------------


class TestABCCompliance:
    """PhonRLStrategy must satisfy the GuidanceStrategy ABC."""

    def test_is_subclass(self) -> None:
        assert issubclass(PhonRLStrategy, GuidanceStrategy)

    def test_has_name_property(self, strategy_no_adapter: PhonRLStrategy) -> None:
        assert hasattr(strategy_no_adapter, "name")
        assert isinstance(strategy_no_adapter.name, str)

    def test_has_prepare_method(self, strategy_no_adapter: PhonRLStrategy) -> None:
        assert callable(getattr(strategy_no_adapter, "prepare", None))

    def test_has_modify_logits_method(self, strategy_no_adapter: PhonRLStrategy) -> None:
        assert callable(getattr(strategy_no_adapter, "modify_logits", None))


# -----------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------


class TestConstruction:
    """Tests for PhonRLStrategy construction and properties."""

    def test_name_is_phon_rl(self) -> None:
        strategy = PhonRLStrategy()
        assert strategy.name == "phon_rl"

    def test_default_no_adapter(self) -> None:
        strategy = PhonRLStrategy()
        assert strategy.adapter_path is None

    def test_adapter_path_stored(self) -> None:
        strategy = PhonRLStrategy(adapter_path="/some/path")
        assert strategy.adapter_path == "/some/path"

    def test_not_loaded_initially(self) -> None:
        strategy = PhonRLStrategy(adapter_path="/some/path")
        assert strategy.is_adapter_loaded is False

    def test_no_adapter_shows_loaded(self) -> None:
        """With no adapter to load, is_adapter_loaded is trivially True."""
        strategy = PhonRLStrategy()
        assert strategy.is_adapter_loaded is True


# -----------------------------------------------------------------------
# modify_logits — identity pass-through
# -----------------------------------------------------------------------


class TestModifyLogits:
    """modify_logits should return logits unchanged (identity)."""

    def test_returns_same_logits_object(
        self, strategy_no_adapter: PhonRLStrategy
    ) -> None:
        """Logits tensor should be returned as-is (no copy, no mutation)."""
        fake_input_ids = MagicMock()
        fake_logits = MagicMock()
        result = strategy_no_adapter.modify_logits(fake_input_ids, fake_logits)
        assert result is fake_logits

    def test_identity_with_numpy_like(
        self, strategy_no_adapter: PhonRLStrategy
    ) -> None:
        """Works with any object — just passes it through."""
        logits = [1.0, 2.0, 3.0]
        result = strategy_no_adapter.modify_logits(None, logits)
        assert result is logits

    def test_identity_with_adapter_configured(
        self, strategy_with_adapter: PhonRLStrategy
    ) -> None:
        """Even with an adapter path, modify_logits is identity."""
        fake_logits = MagicMock()
        result = strategy_with_adapter.modify_logits(None, fake_logits)
        assert result is fake_logits


# -----------------------------------------------------------------------
# prepare — no adapter
# -----------------------------------------------------------------------


class TestPrepareNoAdapter:
    """prepare() with no adapter path should be a no-op."""

    def test_prepare_succeeds(
        self,
        strategy_no_adapter: PhonRLStrategy,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Should not raise."""
        strategy_no_adapter.prepare(
            target_units=["p", "t"],
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

    def test_prepare_does_not_modify_model(
        self,
        strategy_no_adapter: PhonRLStrategy,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """No adapter -> model should not be touched."""
        strategy_no_adapter.prepare(
            target_units=["p", "t"],
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        # Model should not have had load or merge methods called
        mock_model.load_adapter.assert_not_called() if hasattr(
            mock_model, "load_adapter"
        ) else None


# -----------------------------------------------------------------------
# prepare — with adapter (mocked PEFT)
# -----------------------------------------------------------------------


class TestPrepareWithAdapter:
    """prepare() with adapter_path should load PEFT adapter onto model."""

    @patch("corpusgen.generate.phon_rl.policy._load_peft_adapter")
    def test_loads_adapter_on_first_prepare(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        strategy = PhonRLStrategy(adapter_path="/fake/adapter")
        mock_load.return_value = mock_model  # returns the adapted model

        strategy.prepare(
            target_units=["p", "t"],
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        mock_load.assert_called_once_with(mock_model, "/fake/adapter")
        assert strategy.is_adapter_loaded is True

    @patch("corpusgen.generate.phon_rl.policy._load_peft_adapter")
    def test_adapter_loaded_only_once(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Second prepare() should not reload the adapter."""
        strategy = PhonRLStrategy(adapter_path="/fake/adapter")
        mock_load.return_value = mock_model

        strategy.prepare(["p"], mock_model, mock_tokenizer)
        strategy.prepare(["t"], mock_model, mock_tokenizer)

        assert mock_load.call_count == 1

    @patch("corpusgen.generate.phon_rl.policy._load_peft_adapter")
    def test_adapter_load_failure_raises(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """If adapter loading fails, the error should propagate."""
        strategy = PhonRLStrategy(adapter_path="/bad/path")
        mock_load.side_effect = FileNotFoundError("Adapter not found")

        with pytest.raises(FileNotFoundError, match="Adapter not found"):
            strategy.prepare(["p"], mock_model, mock_tokenizer)

        assert strategy.is_adapter_loaded is False


# -----------------------------------------------------------------------
# prepare — target_units stored
# -----------------------------------------------------------------------


class TestPrepareTargetUnits:
    """prepare() should store target_units for potential prompt use."""

    def test_target_units_stored(
        self,
        strategy_no_adapter: PhonRLStrategy,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        strategy_no_adapter.prepare(
            target_units=["p", "t", "k"],
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        assert strategy_no_adapter.current_target_units == ["p", "t", "k"]

    def test_target_units_updated_on_subsequent_prepare(
        self,
        strategy_no_adapter: PhonRLStrategy,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        strategy_no_adapter.prepare(["p", "t"], mock_model, mock_tokenizer)
        strategy_no_adapter.prepare(["b", "d"], mock_model, mock_tokenizer)
        assert strategy_no_adapter.current_target_units == ["b", "d"]

    def test_target_units_none_before_prepare(self) -> None:
        strategy = PhonRLStrategy()
        assert strategy.current_target_units is None


# -----------------------------------------------------------------------
# Full round-trip: prepare then modify_logits
# -----------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end: prepare then modify_logits returns identity."""

    def test_prepare_then_modify(
        self,
        strategy_no_adapter: PhonRLStrategy,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        strategy_no_adapter.prepare(
            target_units=["p", "t"],
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        fake_logits = MagicMock()
        result = strategy_no_adapter.modify_logits(None, fake_logits)
        assert result is fake_logits

    @patch("corpusgen.generate.phon_rl.policy._load_peft_adapter")
    def test_prepare_with_adapter_then_modify(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        strategy = PhonRLStrategy(adapter_path="/fake/adapter")
        mock_load.return_value = mock_model

        strategy.prepare(["p"], mock_model, mock_tokenizer)
        fake_logits = MagicMock()
        result = strategy.modify_logits(None, fake_logits)
        assert result is fake_logits
