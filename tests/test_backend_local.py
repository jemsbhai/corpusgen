"""Tests for LocalBackend — HuggingFace transformers local generation."""

from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from corpusgen.generate.phon_ctg.loop import GenerationBackend
from corpusgen.generate.guidance import GuidanceStrategy
from corpusgen.generate.backends.local import (
    LocalBackend,
    DEFAULT_PROMPT_TEMPLATE,
    _detect_device,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStrategy(GuidanceStrategy):
    """Concrete guidance strategy for testing."""

    def __init__(self) -> None:
        self.prepare_called = False
        self.prepare_args: tuple = ()
        self.modify_calls: int = 0

    @property
    def name(self) -> str:
        return "fake"

    def prepare(self, target_units: list[str], model: Any, tokenizer: Any) -> None:
        self.prepare_called = True
        self.prepare_args = (target_units, model, tokenizer)

    def modify_logits(self, input_ids: Any, logits: Any) -> Any:
        self.modify_calls += 1
        return logits


def _mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer with standard interface."""
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "</s>"
    tok.eos_token_id = 2
    # encode returns tensor-like mock
    encoded = MagicMock()
    encoded.input_ids = MagicMock()
    encoded.to.return_value = encoded
    tok.return_value = encoded
    # decode returns text
    tok.decode.return_value = "The cat sat on the mat."
    tok.batch_decode.return_value = ["The cat sat on the mat."]
    return tok


def _mock_model() -> MagicMock:
    """Create a mock model with generate() method."""
    model = MagicMock()
    # generate returns tensor of token ids
    model.generate.return_value = MagicMock()
    model.generate.return_value.__getitem__ = MagicMock(
        return_value=MagicMock()
    )
    model.device = MagicMock()
    model.device.type = "cpu"
    return model


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """LocalBackend creation and configuration."""

    def test_basic_creation(self):
        backend = LocalBackend(model_name="gpt2")
        assert isinstance(backend, GenerationBackend)
        assert backend.name == "local"

    def test_model_name_property(self):
        backend = LocalBackend(model_name="meta-llama/Llama-3.2-1B")
        assert backend.model_name == "meta-llama/Llama-3.2-1B"

    def test_default_language(self):
        backend = LocalBackend(model_name="gpt2")
        assert backend.language == "en-us"

    def test_custom_language(self):
        backend = LocalBackend(model_name="gpt2", language="fr-fr")
        assert backend.language == "fr-fr"

    def test_default_prompt_template(self):
        backend = LocalBackend(model_name="gpt2")
        assert backend.prompt_template == DEFAULT_PROMPT_TEMPLATE

    def test_custom_prompt_template(self):
        tpl = "Say something with: {target_units}"
        backend = LocalBackend(model_name="gpt2", prompt_template=tpl)
        assert backend.prompt_template == tpl

    def test_default_device_is_none(self):
        """Device is None until auto-detected at load time."""
        backend = LocalBackend(model_name="gpt2")
        assert backend.device is None

    def test_explicit_device(self):
        backend = LocalBackend(model_name="gpt2", device="cpu")
        assert backend.device == "cpu"

    def test_default_quantization_is_none(self):
        backend = LocalBackend(model_name="gpt2")
        assert backend.quantization is None

    def test_quantization_4bit(self):
        backend = LocalBackend(model_name="gpt2", quantization="4bit")
        assert backend.quantization == "4bit"

    def test_quantization_8bit(self):
        backend = LocalBackend(model_name="gpt2", quantization="8bit")
        assert backend.quantization == "8bit"

    def test_invalid_quantization_raises(self):
        with pytest.raises(ValueError, match="quantization"):
            LocalBackend(model_name="gpt2", quantization="3bit")

    def test_default_guidance_strategy_is_none(self):
        backend = LocalBackend(model_name="gpt2")
        assert backend.guidance_strategy is None

    def test_custom_guidance_strategy(self):
        strategy = _FakeStrategy()
        backend = LocalBackend(model_name="gpt2", guidance_strategy=strategy)
        assert backend.guidance_strategy is strategy

    def test_default_max_new_tokens(self):
        backend = LocalBackend(model_name="gpt2")
        assert backend.max_new_tokens == 256

    def test_custom_max_new_tokens(self):
        backend = LocalBackend(model_name="gpt2", max_new_tokens=128)
        assert backend.max_new_tokens == 128

    def test_default_temperature(self):
        backend = LocalBackend(model_name="gpt2")
        assert backend.temperature == 0.8

    def test_default_top_p(self):
        backend = LocalBackend(model_name="gpt2")
        assert backend.top_p == 0.95

    def test_default_do_sample(self):
        backend = LocalBackend(model_name="gpt2")
        assert backend.do_sample is True

    def test_model_kwargs(self):
        backend = LocalBackend(
            model_name="gpt2",
            model_kwargs={"trust_remote_code": True},
        )
        assert backend.model_kwargs == {"trust_remote_code": True}

    def test_default_model_kwargs_empty(self):
        backend = LocalBackend(model_name="gpt2")
        assert backend.model_kwargs == {}


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


class TestLazyLoading:
    """Model and tokenizer are loaded on first generate(), not in __init__."""

    def test_not_loaded_at_construction(self):
        backend = LocalBackend(model_name="gpt2")
        assert not backend.is_loaded

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_loaded_after_generate(self, mock_load_model, mock_load_tok):
        mock_load_tok.return_value = _mock_tokenizer()
        mock_load_model.return_value = _mock_model()

        backend = LocalBackend(model_name="gpt2", device="cpu")
        backend.generate(target_units=["p"], k=1)

        assert backend.is_loaded
        mock_load_model.assert_called_once()
        mock_load_tok.assert_called_once()

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_second_generate_does_not_reload(self, mock_load_model, mock_load_tok):
        mock_load_tok.return_value = _mock_tokenizer()
        mock_load_model.return_value = _mock_model()

        backend = LocalBackend(model_name="gpt2", device="cpu")
        backend.generate(target_units=["p"], k=1)
        backend.generate(target_units=["b"], k=1)

        assert mock_load_model.call_count == 1
        assert mock_load_tok.call_count == 1


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


class TestDeviceDetection:
    """Auto-detection of CUDA availability."""

    @patch("corpusgen.generate.backends.local._cuda_available", return_value=True)
    def test_auto_detect_cuda(self, _mock):
        assert _detect_device() == "cuda"

    @patch("corpusgen.generate.backends.local._cuda_available", return_value=False)
    def test_auto_detect_cpu_fallback(self, _mock):
        assert _detect_device() == "cpu"


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


class TestPromptFormatting:
    """Prompt template rendering."""

    def test_default_template_has_placeholder(self):
        assert "{target_units}" in DEFAULT_PROMPT_TEMPLATE

    def test_format_prompt_includes_targets(self):
        backend = LocalBackend(model_name="gpt2")
        prompt = backend.format_prompt(target_units=["ʃ", "θ"], k=3)
        assert "ʃ" in prompt
        assert "θ" in prompt

    def test_format_prompt_empty_targets(self):
        backend = LocalBackend(model_name="gpt2")
        prompt = backend.format_prompt(target_units=[], k=3)
        assert isinstance(prompt, str)

    def test_custom_template_formatting(self):
        tpl = "Sounds: {target_units} — make {k} sentences in {language}"
        backend = LocalBackend(model_name="gpt2", prompt_template=tpl, language="de")
        prompt = backend.format_prompt(target_units=["x"], k=2)
        assert "x" in prompt
        assert "2" in prompt
        assert "de" in prompt


# ---------------------------------------------------------------------------
# Generate: basic flow (fully mocked)
# ---------------------------------------------------------------------------


class TestGenerate:
    """generate() loads model, runs generation, phonemizes, returns candidates."""

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_returns_candidate_list(self, mock_load_model, mock_load_tok):
        tok = _mock_tokenizer()
        model = _mock_model()
        mock_load_tok.return_value = tok
        mock_load_model.return_value = model

        backend = LocalBackend(model_name="gpt2", device="cpu")
        candidates = backend.generate(target_units=["k", "æ"], k=3)

        assert isinstance(candidates, list)

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_candidates_have_text_and_phonemes(self, mock_load_model, mock_load_tok):
        tok = _mock_tokenizer()
        model = _mock_model()
        mock_load_tok.return_value = tok
        mock_load_model.return_value = model

        backend = LocalBackend(model_name="gpt2", device="cpu")
        candidates = backend.generate(target_units=["k"], k=3)

        for c in candidates:
            assert "text" in c
            assert "phonemes" in c
            assert isinstance(c["phonemes"], list)

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_k_zero_returns_empty(self, mock_load_model, mock_load_tok):
        backend = LocalBackend(model_name="gpt2", device="cpu")
        candidates = backend.generate(target_units=["p"], k=0)
        assert candidates == []
        # Model should not be loaded for k=0
        mock_load_model.assert_not_called()

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_calls_model_generate(self, mock_load_model, mock_load_tok):
        tok = _mock_tokenizer()
        model = _mock_model()
        mock_load_tok.return_value = tok
        mock_load_model.return_value = model

        backend = LocalBackend(model_name="gpt2", device="cpu")
        backend.generate(target_units=["p"], k=2)

        model.generate.assert_called()

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_multiple_candidates_from_num_return_sequences(
        self, mock_load_model, mock_load_tok
    ):
        tok = _mock_tokenizer()
        tok.batch_decode.return_value = [
            "First sentence here.",
            "Second sentence here.",
            "Third sentence here.",
        ]
        model = _mock_model()
        mock_load_tok.return_value = tok
        mock_load_model.return_value = model

        backend = LocalBackend(model_name="gpt2", device="cpu")
        candidates = backend.generate(target_units=["s"], k=3)

        assert len(candidates) <= 3


# ---------------------------------------------------------------------------
# Guidance strategy integration
# ---------------------------------------------------------------------------


class TestGuidanceIntegration:
    """GuidanceStrategy.prepare() is called before generation."""

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_prepare_called_with_targets(self, mock_load_model, mock_load_tok):
        tok = _mock_tokenizer()
        model = _mock_model()
        mock_load_tok.return_value = tok
        mock_load_model.return_value = model

        strategy = _FakeStrategy()
        backend = LocalBackend(
            model_name="gpt2",
            device="cpu",
            guidance_strategy=strategy,
        )
        backend.generate(target_units=["ʃ", "θ"], k=2)

        assert strategy.prepare_called
        assert strategy.prepare_args[0] == ["ʃ", "θ"]

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_prepare_receives_model_and_tokenizer(
        self, mock_load_model, mock_load_tok
    ):
        tok = _mock_tokenizer()
        model = _mock_model()
        mock_load_tok.return_value = tok
        mock_load_model.return_value = model

        strategy = _FakeStrategy()
        backend = LocalBackend(
            model_name="gpt2",
            device="cpu",
            guidance_strategy=strategy,
        )
        backend.generate(target_units=["p"], k=1)

        # prepare should receive the actual model and tokenizer
        assert strategy.prepare_args[1] is model
        assert strategy.prepare_args[2] is tok

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_no_strategy_no_logits_processor(self, mock_load_model, mock_load_tok):
        tok = _mock_tokenizer()
        model = _mock_model()
        mock_load_tok.return_value = tok
        mock_load_model.return_value = model

        backend = LocalBackend(model_name="gpt2", device="cpu")
        backend.generate(target_units=["p"], k=1)

        # Without strategy, generate should not pass logits_processor
        call_kwargs = model.generate.call_args[1]
        assert "logits_processor" not in call_kwargs or call_kwargs.get(
            "logits_processor"
        ) is None


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------


class TestQuantizationConfig:
    """Quantization parameter validation."""

    def test_none_is_valid(self):
        backend = LocalBackend(model_name="gpt2", quantization=None)
        assert backend.quantization is None

    def test_4bit_is_valid(self):
        backend = LocalBackend(model_name="gpt2", quantization="4bit")
        assert backend.quantization == "4bit"

    def test_8bit_is_valid(self):
        backend = LocalBackend(model_name="gpt2", quantization="8bit")
        assert backend.quantization == "8bit"

    def test_invalid_raises_valueerror(self):
        with pytest.raises(ValueError, match="quantization"):
            LocalBackend(model_name="gpt2", quantization="16bit")

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="quantization"):
            LocalBackend(model_name="gpt2", quantization="")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions."""

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_empty_target_units(self, mock_load_model, mock_load_tok):
        tok = _mock_tokenizer()
        model = _mock_model()
        mock_load_tok.return_value = tok
        mock_load_model.return_value = model

        backend = LocalBackend(model_name="gpt2", device="cpu")
        candidates = backend.generate(target_units=[], k=3)
        assert isinstance(candidates, list)

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch("corpusgen.generate.backends.local._load_model")
    def test_model_returns_empty_text(self, mock_load_model, mock_load_tok):
        tok = _mock_tokenizer()
        tok.batch_decode.return_value = [""]
        model = _mock_model()
        mock_load_tok.return_value = tok
        mock_load_model.return_value = model

        backend = LocalBackend(model_name="gpt2", device="cpu")
        candidates = backend.generate(target_units=["p"], k=1)
        assert candidates == []

    def test_negative_k_returns_empty(self):
        backend = LocalBackend(model_name="gpt2")
        candidates = backend.generate(target_units=["p"], k=-1)
        assert candidates == []


# ---------------------------------------------------------------------------
# Missing dependencies
# ---------------------------------------------------------------------------


class TestMissingDependencies:
    """Clear errors when optional packages are missing."""

    @patch(
        "corpusgen.generate.backends.local._load_model",
        side_effect=ImportError("No module named 'transformers'"),
    )
    def test_transformers_missing(self, _mock):
        backend = LocalBackend(model_name="gpt2", device="cpu")
        with pytest.raises(ImportError, match="transformers"):
            backend.generate(target_units=["p"], k=1)

    @patch("corpusgen.generate.backends.local._load_tokenizer")
    @patch(
        "corpusgen.generate.backends.local._load_model",
        side_effect=ImportError("No module named 'torch'"),
    )
    def test_torch_missing(self, _mock_model, _mock_tok):
        _mock_tok.return_value = _mock_tokenizer()
        backend = LocalBackend(model_name="gpt2", device="cpu")
        with pytest.raises(ImportError, match="torch"):
            backend.generate(target_units=["p"], k=1)
