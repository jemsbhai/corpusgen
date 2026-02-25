"""Tests for RepositoryBackend — selects from sentence pools."""

from unittest.mock import patch, MagicMock

import pytest

from corpusgen.generate.phon_ctg.loop import GenerationBackend
from corpusgen.generate.backends.repository import RepositoryBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_sentences():
    """Raw text sentences with known phoneme content."""
    return [
        "pat the bat",
        "kick the ball",
        "the dog sat",
        "fish and chips",
        "she sells seashells",
    ]


@pytest.fixture
def prephon_pool():
    """Pre-phonemized sentence pool."""
    return [
        {"text": "pat", "phonemes": ["p", "æ", "t"]},
        {"text": "bat", "phonemes": ["b", "æ", "t"]},
        {"text": "kick", "phonemes": ["k", "ɪ", "k"]},
        {"text": "dog", "phonemes": ["d", "ɒ", "ɡ"]},
        {"text": "fish", "phonemes": ["f", "ɪ", "ʃ"]},
    ]


# ---------------------------------------------------------------------------
# Construction: list-based with pre-phonemized data
# ---------------------------------------------------------------------------


class TestConstructionPrePhonemized:
    """RepositoryBackend created with pre-phonemized pool."""

    def test_basic_creation(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        assert isinstance(backend, GenerationBackend)
        assert backend.name == "repository"

    def test_pool_size(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        assert backend.pool_size == 5

    def test_requires_phonemes_key(self):
        with pytest.raises(ValueError, match="[Pp]honemes"):
            RepositoryBackend(pool=[
                {"text": "hello"},  # missing phonemes
            ])


# ---------------------------------------------------------------------------
# Construction: list-based with raw text
# ---------------------------------------------------------------------------


class TestConstructionRawText:
    """RepositoryBackend created with raw text + G2P."""

    def test_from_texts(self, simple_sentences):
        backend = RepositoryBackend.from_texts(
            texts=simple_sentences,
            language="en-us",
        )
        assert isinstance(backend, GenerationBackend)
        assert backend.pool_size == len(simple_sentences)

    def test_from_texts_default_language(self, simple_sentences):
        backend = RepositoryBackend.from_texts(texts=simple_sentences)
        assert backend.pool_size == len(simple_sentences)

    def test_from_texts_empty_raises(self):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            RepositoryBackend.from_texts(texts=[])

    def test_from_texts_stores_phonemes(self, simple_sentences):
        backend = RepositoryBackend.from_texts(
            texts=simple_sentences,
            language="en-us",
        )
        # Pool entries should have phonemes populated
        pool = backend.pool
        assert all("phonemes" in entry for entry in pool)
        assert all(len(entry["phonemes"]) > 0 for entry in pool)


# ---------------------------------------------------------------------------
# Construction: HuggingFace datasets
# ---------------------------------------------------------------------------


class TestConstructionHuggingFace:
    """RepositoryBackend created from HuggingFace datasets."""

    def _mock_dataset(self):
        """Create a mock HF dataset."""
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([
            {"text": "the cat sat"},
            {"text": "a big dog"},
            {"text": "red and blue"},
        ]))
        mock_ds.__len__ = MagicMock(return_value=3)
        return mock_ds

    @patch("corpusgen.generate.backends.repository._load_hf_dataset")
    def test_from_huggingface(self, mock_load):
        mock_load.return_value = self._mock_dataset()
        backend = RepositoryBackend.from_huggingface(
            dataset_name="test/dataset",
            text_column="text",
            language="en-us",
        )
        assert isinstance(backend, GenerationBackend)
        assert backend.pool_size == 3
        mock_load.assert_called_once()

    @patch("corpusgen.generate.backends.repository._load_hf_dataset")
    def test_from_huggingface_with_split(self, mock_load):
        mock_load.return_value = self._mock_dataset()
        RepositoryBackend.from_huggingface(
            dataset_name="test/dataset",
            text_column="text",
            split="train",
            language="en-us",
        )
        call_kwargs = mock_load.call_args
        assert call_kwargs is not None

    @patch("corpusgen.generate.backends.repository._load_hf_dataset")
    def test_from_huggingface_with_max_samples(self, mock_load):
        mock_load.return_value = self._mock_dataset()
        backend = RepositoryBackend.from_huggingface(
            dataset_name="test/dataset",
            text_column="text",
            language="en-us",
            max_samples=2,
        )
        assert backend.pool_size <= 2

    @patch("corpusgen.generate.backends.repository._load_hf_dataset")
    def test_from_huggingface_custom_text_column(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([
            {"sentence": "hello world"},
            {"sentence": "foo bar"},
        ]))
        mock_ds.__len__ = MagicMock(return_value=2)
        mock_load.return_value = mock_ds

        backend = RepositoryBackend.from_huggingface(
            dataset_name="test/dataset",
            text_column="sentence",
            language="en-us",
        )
        assert backend.pool_size == 2

    def test_from_huggingface_missing_dependency(self):
        """Raises ImportError if datasets not installed."""
        with patch(
            "corpusgen.generate.backends.repository._load_hf_dataset",
            side_effect=ImportError("No module named 'datasets'"),
        ):
            with pytest.raises(ImportError):
                RepositoryBackend.from_huggingface(
                    dataset_name="test/dataset",
                    text_column="text",
                    language="en-us",
                )


# ---------------------------------------------------------------------------
# Generate: basic behavior
# ---------------------------------------------------------------------------


class TestGenerate:
    """generate() returns candidates matching target units."""

    def test_returns_candidates_with_target_phonemes(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        # Target "p" — "pat" has it
        candidates = backend.generate(target_units=["p"], k=5)
        assert len(candidates) >= 1
        # At least one candidate should have "p" in its phonemes
        has_p = any("p" in c["phonemes"] for c in candidates)
        assert has_p

    def test_returns_at_most_k_candidates(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        candidates = backend.generate(target_units=["æ"], k=2)
        assert len(candidates) <= 2

    def test_candidates_have_text_and_phonemes(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        candidates = backend.generate(target_units=["p"], k=5)
        for c in candidates:
            assert "phonemes" in c
            assert "text" in c

    def test_returns_empty_for_impossible_target(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        candidates = backend.generate(target_units=["ʔ"], k=5)
        # ʔ is not in any pool entry — but backend may still return
        # candidates ranked by score; if none contain the target, could be empty
        # At minimum, the result is a list
        assert isinstance(candidates, list)

    def test_empty_target_units(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        candidates = backend.generate(target_units=[], k=5)
        assert isinstance(candidates, list)


# ---------------------------------------------------------------------------
# Generate: diphone/triphone targets
# ---------------------------------------------------------------------------


class TestGenerateNgrams:
    """generate() works with diphone and triphone target units."""

    def test_diphone_target(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        # "pat" has phonemes [p, æ, t] -> diphones: p-æ, æ-t
        candidates = backend.generate(target_units=["p-æ"], k=5)
        assert len(candidates) >= 1

    def test_triphone_target(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        # "pat" has phonemes [p, æ, t] -> triphone: p-æ-t
        candidates = backend.generate(target_units=["p-æ-t"], k=5)
        assert len(candidates) >= 1


# ---------------------------------------------------------------------------
# Deduplication: used sentences removed
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Used sentences are removed from the pool."""

    def test_mark_used_reduces_pool(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        assert backend.pool_size == 5
        backend.mark_used(0)
        assert backend.pool_size == 4

    def test_mark_used_removes_correct_sentence(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        first_text = backend.pool[0]["text"]
        backend.mark_used(0)
        remaining_texts = [e["text"] for e in backend.pool]
        assert first_text not in remaining_texts

    def test_generate_excludes_used(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        # Get first round of candidates
        candidates1 = backend.generate(target_units=["p"], k=5)
        # Mark the first candidate as used by its pool index
        if candidates1:
            backend.mark_used(candidates1[0]["_pool_index"])
            pool_size_after = backend.pool_size
            assert pool_size_after == 4

    def test_exhaustion_after_all_used(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        # Mark all as used
        for i in range(len(prephon_pool)):
            backend.mark_used(0)  # always remove first remaining
        assert backend.pool_size == 0
        candidates = backend.generate(target_units=["p"], k=5)
        assert candidates == []


# ---------------------------------------------------------------------------
# Pool access
# ---------------------------------------------------------------------------


class TestPoolAccess:
    """Read access to the pool state."""

    def test_pool_property(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        pool = backend.pool
        assert isinstance(pool, list)
        assert len(pool) == 5

    def test_pool_is_copy(self, prephon_pool):
        """Modifying returned pool doesn't affect internal state."""
        backend = RepositoryBackend(pool=prephon_pool)
        pool = backend.pool
        pool.clear()
        assert backend.pool_size == 5  # unchanged


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions."""

    def test_single_sentence_pool(self):
        pool = [{"text": "hi", "phonemes": ["h", "aɪ"]}]
        backend = RepositoryBackend(pool=pool)
        assert backend.pool_size == 1
        candidates = backend.generate(target_units=["h"], k=5)
        assert len(candidates) == 1

    def test_k_zero(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        candidates = backend.generate(target_units=["p"], k=0)
        assert candidates == []

    def test_mark_used_invalid_index_raises(self, prephon_pool):
        backend = RepositoryBackend(pool=prephon_pool)
        with pytest.raises(IndexError):
            backend.mark_used(99)

    def test_duplicate_phonemes_in_pool(self):
        """Pool entries can have repeated phonemes."""
        pool = [
            {"text": "papa", "phonemes": ["p", "ɑ", "p", "ɑ"]},
        ]
        backend = RepositoryBackend(pool=pool)
        candidates = backend.generate(target_units=["p"], k=5)
        assert len(candidates) == 1
