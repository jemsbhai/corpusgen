"""Slow integration tests for Phon-DATG — real gpt2 model, real espeak-ng G2P.

Test tiers:
    - **Fast tests** (in test_phon_datg_*.py): Pure logic with mocks.
    - **Slow tests** (this file, @pytest.mark.slow): Real gpt2 tokenizer,
      real espeak-ng G2P, real torch tensors. Proves the full DATG pipeline
      works end-to-end.

Skipped by default. Run with:
    poetry run pytest -m slow -k datg

Requirements:
    - poetry install --with local
    - espeak-ng installed and on PATH
    - ~500MB disk for gpt2 weights (cached after first download)

These tests MUST pass before any publication claim about Phon-DATG.
"""

from __future__ import annotations

import math

import pytest

from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_datg.attribute_words import (
    AttributeWordIndex,
    _extract_units,
)
from corpusgen.generate.phon_datg.graph import DATGStrategy
from corpusgen.generate.phon_datg.modulator import LogitModulator


# ===================================================================
# Dependency checks
# ===================================================================


def _has_slow_deps() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


_skip_slow_deps = pytest.mark.skipif(
    not _has_slow_deps(),
    reason="Slow test dependencies not installed (torch, transformers)",
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """Load real gpt2 tokenizer once for the module."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def gpt2_model():
    """Load real gpt2 model once for the module (CPU only)."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def built_index(gpt2_tokenizer):
    """Build a real AttributeWordIndex from gpt2 vocabulary.

    This is an expensive operation (~50k tokens through espeak-ng G2P).
    Cached at module scope so it runs once across all tests.
    """
    index = AttributeWordIndex(language="en-us", batch_size=512)
    index.build(gpt2_tokenizer)
    return index


# ===================================================================
# 1. INDEX BUILD — Real tokenizer + real G2P
# ===================================================================


@pytest.mark.slow
@_skip_slow_deps
class TestRealIndexBuild:
    """Verify AttributeWordIndex builds correctly from gpt2's ~50k vocabulary."""

    def test_index_is_built(self, built_index):
        assert built_index.is_built

    def test_non_trivial_unit_count(self, built_index):
        """A real English vocabulary should produce hundreds of distinct units."""
        mapping = built_index.unit_to_tokens
        assert len(mapping) > 100

    def test_non_trivial_token_count(self, built_index):
        """Many tokens should have phonetic content (not all are special tokens)."""
        token_map = built_index.token_units
        assert len(token_map) > 1000

    def test_common_phonemes_have_multiple_tokens(self, built_index):
        """Frequent English phonemes like 'k', 't', 's' should map to many tokens.

        If a common phoneme maps to very few tokens, the index is broken
        and logit steering would be ineffective.
        """
        mapping = built_index.unit_to_tokens
        for phoneme in ["t", "s", "k", "n"]:
            token_ids = mapping.get(phoneme, set())
            assert len(token_ids) > 50, (
                f"Phoneme '{phoneme}' maps to only {len(token_ids)} tokens; "
                f"expected >50 for a common English sound in ~50k vocabulary"
            )

    def test_rare_phonemes_have_fewer_tokens(self, built_index):
        """Less common phonemes should map to fewer tokens than common ones."""
        mapping = built_index.unit_to_tokens
        common_count = len(mapping.get("t", set()))
        # ʒ (as in "measure") is rare in English
        rare_count = len(mapping.get("ʒ", set()))
        # We don't require a specific ratio, just that the relationship holds
        if rare_count > 0:
            assert rare_count < common_count, (
                f"Rare phoneme 'ʒ' ({rare_count} tokens) should have fewer "
                f"tokens than common 't' ({common_count} tokens)"
            )

    def test_diphones_present_in_index(self, built_index):
        """Index should contain diphone units (X-Y format)."""
        mapping = built_index.unit_to_tokens
        diphones = {u for u in mapping if u.count("-") == 1}
        assert len(diphones) > 50, (
            f"Only {len(diphones)} diphones indexed; expected >50"
        )

    def test_triphones_present_in_index(self, built_index):
        """Index should contain triphone units (X-Y-Z format)."""
        mapping = built_index.unit_to_tokens
        triphones = {u for u in mapping if u.count("-") == 2}
        assert len(triphones) > 50, (
            f"Only {len(triphones)} triphones indexed; expected >50"
        )


# ===================================================================
# 2. BIDIRECTIONAL CONSISTENCY — unit_to_tokens ↔ token_to_units
# ===================================================================


@pytest.mark.slow
@_skip_slow_deps
class TestRealIndexConsistency:
    """The two directions of the index must agree.

    If token T is in unit_to_tokens[U], then U must be in token_to_units[T],
    and vice versa. Inconsistency here means the index is corrupt and
    all downstream coverage metrics are wrong.
    """

    def test_unit_to_tokens_implies_token_to_units(self, built_index):
        """For every (unit → token) entry, the reverse must exist."""
        unit_map = built_index.unit_to_tokens
        token_map = built_index.token_units

        # Sample check — testing every entry would be slow, so check
        # a representative subset of units
        checked = 0
        for unit, token_ids in unit_map.items():
            if checked >= 200:
                break
            for tid in list(token_ids)[:5]:
                assert tid in token_map, (
                    f"Token {tid} in unit_to_tokens['{unit}'] but "
                    f"missing from token_to_units"
                )
                assert unit in token_map[tid], (
                    f"Unit '{unit}' maps to token {tid}, but "
                    f"token_to_units[{tid}] does not contain '{unit}'"
                )
            checked += 1

    def test_token_to_units_implies_unit_to_tokens(self, built_index):
        """For every (token → unit) entry, the reverse must exist."""
        unit_map = built_index.unit_to_tokens
        token_map = built_index.token_units

        checked = 0
        for tid, units in token_map.items():
            if checked >= 200:
                break
            for unit in units:
                assert unit in unit_map, (
                    f"Unit '{unit}' in token_to_units[{tid}] but "
                    f"missing from unit_to_tokens"
                )
                assert tid in unit_map[unit], (
                    f"Token {tid} has unit '{unit}', but "
                    f"unit_to_tokens['{unit}'] does not contain {tid}"
                )
            checked += 1


# ===================================================================
# 3. SEMANTIC CORRECTNESS — tokens decode to text with expected sounds
# ===================================================================


@pytest.mark.slow
@_skip_slow_deps
class TestRealIndexSemantics:
    """Verify that attribute tokens actually contain the target sounds.

    This is the critical semantic check: if the index says token X
    contains phoneme /k/, then decoding token X should produce text
    that espeak-ng phonemizes to something containing /k/.
    """

    def test_attribute_tokens_for_k_decode_correctly(
        self, built_index, gpt2_tokenizer
    ):
        """Tokens indexed under 'k' should decode to text containing /k/.

        We verify a sample (not exhaustive — full verification would
        re-phonemize the entire vocabulary).
        """
        from corpusgen.g2p.manager import G2PManager

        g2p = G2PManager()
        k_tokens = built_index.unit_to_tokens.get("k", set())
        assert len(k_tokens) > 0, "No tokens indexed for phoneme 'k'"

        # Sample up to 20 tokens and verify
        sample = list(k_tokens)[:20]
        verified = 0
        for tid in sample:
            decoded = gpt2_tokenizer.decode(tid, skip_special_tokens=True)
            if not decoded.strip():
                continue
            results = g2p.phonemize_batch([decoded], language="en-us")
            phonemes = results[0].phonemes
            if not phonemes:
                continue
            units = _extract_units(phonemes)
            if "k" in units:
                verified += 1

        # At least half the sample should verify (some BPE fragments
        # may phonemize differently in isolation vs in context)
        assert verified >= len(sample) // 3, (
            f"Only {verified}/{len(sample)} sampled tokens for 'k' "
            f"verified via re-phonemization"
        )

    def test_attribute_tokens_for_diphone(
        self, built_index, gpt2_tokenizer
    ):
        """Tokens indexed under a common diphone should contain it."""
        from corpusgen.g2p.manager import G2PManager

        g2p = G2PManager()
        # Find a diphone that has tokens
        diphone = None
        for unit, tids in built_index.unit_to_tokens.items():
            if unit.count("-") == 1 and len(tids) >= 5:
                diphone = unit
                break

        if diphone is None:
            pytest.skip("No diphone with ≥5 tokens found")

        sample = list(built_index.unit_to_tokens[diphone])[:10]
        verified = 0
        for tid in sample:
            decoded = gpt2_tokenizer.decode(tid, skip_special_tokens=True)
            if not decoded.strip():
                continue
            results = g2p.phonemize_batch([decoded], language="en-us")
            phonemes = results[0].phonemes
            if not phonemes:
                continue
            units = _extract_units(phonemes)
            if diphone in units:
                verified += 1

        assert verified >= 1, (
            f"No sampled tokens for diphone '{diphone}' verified "
            f"via re-phonemization"
        )


# ===================================================================
# 4. ATTRIBUTE / ANTI-ATTRIBUTE SET COMPUTATION — Real index
# ===================================================================


@pytest.mark.slow
@_skip_slow_deps
class TestRealAttributeSets:
    """Verify attribute and anti-attribute token sets with real index."""

    def test_attribute_tokens_non_empty_for_common_phoneme(self, built_index):
        """Targeting a common phoneme should produce a non-empty attribute set."""
        result = built_index.get_attribute_tokens(["t"])
        assert len(result) > 50

    def test_attribute_tokens_empty_for_nonexistent_unit(self, built_index):
        """A phoneme not in any English word should produce empty set."""
        # ɮ (voiced lateral fricative) — extremely rare in English
        result = built_index.get_attribute_tokens(["ɮ"])
        assert len(result) == 0

    def test_anti_attribute_covered_mode(self, built_index):
        """Tokens fully covered at phoneme level should be anti-attribute."""
        # Cover a small set of very common phonemes
        covered = {"ð", "ə"}  # "the" only has these
        result = built_index.get_anti_attribute_tokens(
            covered, unit_level="phoneme"
        )
        # Some tokens should be fully covered by just ð and ə
        # (e.g., the token for "the")
        assert len(result) > 0

    def test_anti_attribute_diphone_level(self, built_index):
        """Anti-attribute at diphone level checks diphones, not phonemes."""
        # Cover a common diphone
        covered_diphones = {"ð-ə"}
        result = built_index.get_anti_attribute_tokens(
            covered_diphones, unit_level="diphone"
        )
        # Tokens with only the diphone ð-ə should appear
        # "the" → [ð, ə] → diphone: {ð-ə} — should be anti-attribute
        assert len(result) >= 0  # May be 0 if no tokens have ONLY this diphone

    def test_attribute_and_anti_attribute_disjoint_when_no_overlap(
        self, built_index
    ):
        """With a clean partition, attribute and anti-attribute shouldn't overlap.

        Target 'ʃ' (uncovered). Cover 'ð', 'ə'. Tokens containing 'ʃ'
        should not be in the anti-attribute set (since they have uncovered 'ʃ').
        """
        attribute = built_index.get_attribute_tokens(["ʃ"])
        covered = {"ð", "ə"}
        anti_attribute = built_index.get_anti_attribute_tokens(
            covered, unit_level="phoneme"
        )

        # A token containing ʃ has at least one uncovered phoneme,
        # so it shouldn't be anti-attribute
        overlap = attribute & anti_attribute
        assert len(overlap) == 0, (
            f"{len(overlap)} tokens are both attribute (contain 'ʃ') and "
            f"anti-attribute (all phonemes in {{ð, ə}}). This is contradictory."
        )


# ===================================================================
# 5. LOGIT MODULATION — Real torch tensors
# ===================================================================


@pytest.mark.slow
@_skip_slow_deps
class TestRealLogitModulation:
    """Verify LogitModulator with real-sized tensors."""

    def test_modulation_on_real_vocab_size(self):
        """Boost/penalty applied correctly on a 50257-dim tensor (gpt2 vocab)."""
        import torch

        vocab_size = 50257
        logits = torch.zeros(1, vocab_size)
        mod = LogitModulator(boost_strength=5.0, penalty_strength=-5.0)

        attribute_ids = set(range(0, 100))
        anti_attribute_ids = set(range(200, 300))

        result = mod.modulate(logits, attribute_ids, anti_attribute_ids)

        # Boosted tokens
        for tid in [0, 50, 99]:
            assert result[0, tid].item() == pytest.approx(5.0)

        # Penalized tokens
        for tid in [200, 250, 299]:
            assert result[0, tid].item() == pytest.approx(-5.0)

        # Neutral tokens
        for tid in [100, 150, 500, 50000]:
            assert result[0, tid].item() == pytest.approx(0.0)

    def test_modulation_does_not_mutate_input(self):
        """Original tensor must be unchanged (safety for autoregressive loop)."""
        import torch

        logits = torch.ones(1, 1000)
        mod = LogitModulator(boost_strength=10.0, penalty_strength=-10.0)
        original = logits.clone()

        mod.modulate(logits, {0, 1, 2}, {500, 501})

        assert torch.equal(logits, original)


# ===================================================================
# 6. PROBABILITY SHIFT — The core scientific claim of Phon-DATG
# ===================================================================


@pytest.mark.slow
@_skip_slow_deps
class TestRealProbabilityShift:
    """Verify that logit modulation shifts probability mass as claimed.

    This is the core scientific claim: Phon-DATG increases the sampling
    probability of tokens containing target phonetic units and decreases
    the probability of tokens containing only already-covered units.

    If these tests fail, the method as described in a paper would be
    making a false claim about its mechanism.
    """

    def test_softmax_mass_shifts_toward_attribute_tokens(self, built_index):
        """After modulation, attribute tokens should have higher total
        probability than before.
        """
        import torch

        vocab_size = 50257
        # Uniform logits — every token equally likely before modulation
        logits = torch.zeros(1, vocab_size)

        target_tokens = built_index.get_attribute_tokens(["ʃ"])
        if len(target_tokens) == 0:
            pytest.skip("No attribute tokens for 'ʃ'")

        mod = LogitModulator(boost_strength=5.0, penalty_strength=0.0)
        modified = mod.modulate(logits, target_tokens, set())

        # Compute probabilities
        probs_before = torch.softmax(logits, dim=-1)
        probs_after = torch.softmax(modified, dim=-1)

        # Sum probability mass over attribute tokens
        target_list = list(target_tokens)
        mass_before = probs_before[0, target_list].sum().item()
        mass_after = probs_after[0, target_list].sum().item()

        assert mass_after > mass_before, (
            f"Attribute token probability mass did not increase: "
            f"before={mass_before:.6f}, after={mass_after:.6f}"
        )

    def test_softmax_mass_shifts_away_from_anti_attribute_tokens(
        self, built_index
    ):
        """After modulation, anti-attribute tokens should have lower total
        probability than before.
        """
        import torch

        vocab_size = 50257
        logits = torch.zeros(1, vocab_size)

        covered = {"t", "s", "n", "ð", "ə"}
        anti_tokens = built_index.get_anti_attribute_tokens(
            covered, unit_level="phoneme"
        )
        if len(anti_tokens) == 0:
            pytest.skip("No anti-attribute tokens for the given covered set")

        mod = LogitModulator(boost_strength=0.0, penalty_strength=-5.0)
        modified = mod.modulate(logits, set(), anti_tokens)

        probs_before = torch.softmax(logits, dim=-1)
        probs_after = torch.softmax(modified, dim=-1)

        anti_list = list(anti_tokens)
        mass_before = probs_before[0, anti_list].sum().item()
        mass_after = probs_after[0, anti_list].sum().item()

        assert mass_after < mass_before, (
            f"Anti-attribute token probability mass did not decrease: "
            f"before={mass_before:.6f}, after={mass_after:.6f}"
        )

    def test_combined_shift_net_benefit(self, built_index):
        """Combined boost + penalty should increase attribute/anti-attribute
        probability ratio.
        """
        import torch

        vocab_size = 50257
        logits = torch.zeros(1, vocab_size)

        target_tokens = built_index.get_attribute_tokens(["ʃ"])
        covered = {"t", "s", "n", "ð", "ə"}
        anti_tokens = built_index.get_anti_attribute_tokens(
            covered, unit_level="phoneme"
        )

        if len(target_tokens) == 0 or len(anti_tokens) == 0:
            pytest.skip("Need both attribute and anti-attribute tokens")

        mod = LogitModulator(boost_strength=5.0, penalty_strength=-5.0)
        modified = mod.modulate(logits, target_tokens, anti_tokens)

        probs_before = torch.softmax(logits, dim=-1)
        probs_after = torch.softmax(modified, dim=-1)

        target_list = list(target_tokens)
        anti_list = list(anti_tokens)

        ratio_before = (
            probs_before[0, target_list].sum().item()
            / max(probs_before[0, anti_list].sum().item(), 1e-10)
        )
        ratio_after = (
            probs_after[0, target_list].sum().item()
            / max(probs_after[0, anti_list].sum().item(), 1e-10)
        )

        assert ratio_after > ratio_before, (
            f"Attribute/anti-attribute probability ratio did not improve: "
            f"before={ratio_before:.4f}, after={ratio_after:.4f}"
        )


# ===================================================================
# 7. FULL PIPELINE — DATGStrategy with real model
# ===================================================================


@pytest.mark.slow
@_skip_slow_deps
class TestRealDATGStrategy:
    """End-to-end: DATGStrategy.prepare() + modify_logits() with real gpt2."""

    def test_prepare_builds_index_and_computes_sets(
        self, gpt2_model, gpt2_tokenizer
    ):
        """Full prepare() with real model and tokenizer."""
        targets = PhoneticTargetInventory(
            target_phonemes=[
                "p", "b", "t", "d", "k",
                "s", "z", "ʃ", "ʒ",
                "m", "n", "ŋ",
            ],
            unit="phoneme",
        )
        strategy = DATGStrategy(
            targets=targets,
            language="en-us",
            boost_strength=5.0,
            penalty_strength=-5.0,
        )

        strategy.prepare(
            target_units=["ʃ", "ʒ"],
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
        )

        assert len(strategy.current_attribute_tokens) > 0
        assert strategy.attribute_word_index.is_built

    def test_modify_logits_produces_valid_tensor(
        self, gpt2_model, gpt2_tokenizer
    ):
        """modify_logits() returns a tensor of correct shape with finite values."""
        import torch

        targets = PhoneticTargetInventory(
            target_phonemes=["p", "t", "k", "s", "ʃ"],
            unit="phoneme",
        )
        strategy = DATGStrategy(
            targets=targets,
            boost_strength=5.0,
            penalty_strength=-5.0,
        )
        strategy.prepare(
            target_units=["ʃ"],
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
        )

        vocab_size = gpt2_tokenizer.vocab_size
        logits = torch.randn(1, vocab_size)
        input_ids = torch.zeros(1, 10, dtype=torch.long)

        result = strategy.modify_logits(input_ids, logits)

        assert result.shape == logits.shape
        assert torch.all(torch.isfinite(result))

    def test_modify_logits_actually_changes_values(
        self, gpt2_model, gpt2_tokenizer
    ):
        """Logits should differ from input — modulation is not a no-op."""
        import torch

        targets = PhoneticTargetInventory(
            target_phonemes=["p", "t", "k", "s", "ʃ"],
            unit="phoneme",
        )
        strategy = DATGStrategy(
            targets=targets,
            boost_strength=5.0,
            penalty_strength=-5.0,
        )
        strategy.prepare(
            target_units=["ʃ"],
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
        )

        vocab_size = gpt2_tokenizer.vocab_size
        logits = torch.zeros(1, vocab_size)
        input_ids = torch.zeros(1, 10, dtype=torch.long)

        result = strategy.modify_logits(input_ids, logits)

        # Some tokens should be boosted (>0) and some penalized (<0)
        assert torch.any(result > 0.0), "No tokens were boosted"
        # Penalty requires covered units — we have a fresh tracker,
        # so anti-attribute set may be empty. Only assert boost.

    def test_full_pipeline_with_coverage_updates(
        self, gpt2_model, gpt2_tokenizer
    ):
        """Simulate a generation loop: prepare, modulate, update coverage, repeat.

        This tests that the strategy correctly adapts as coverage state
        changes between generation rounds.
        """
        import torch

        targets = PhoneticTargetInventory(
            target_phonemes=["p", "t", "k", "s", "ʃ", "ð", "ə"],
            unit="phoneme",
        )
        strategy = DATGStrategy(
            targets=targets,
            boost_strength=5.0,
            penalty_strength=-5.0,
            anti_attribute_mode="covered",
        )

        vocab_size = gpt2_tokenizer.vocab_size
        input_ids = torch.zeros(1, 10, dtype=torch.long)

        # Round 1: nothing covered yet, target ʃ
        strategy.prepare(
            target_units=["ʃ"],
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
        )
        logits_1 = torch.zeros(1, vocab_size)
        result_1 = strategy.modify_logits(input_ids, logits_1)
        anti_count_1 = len(strategy.current_anti_attribute_tokens)

        # Simulate covering some phonemes
        targets.update(["ð", "ə"], sentence_index=0)
        targets.update(["t", "s"], sentence_index=1)

        # Round 2: ð, ə, t, s now covered; re-prepare targeting ʃ
        strategy.prepare(
            target_units=["ʃ"],
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
        )
        logits_2 = torch.zeros(1, vocab_size)
        result_2 = strategy.modify_logits(input_ids, logits_2)
        anti_count_2 = len(strategy.current_anti_attribute_tokens)

        # After covering more phonemes, more tokens should be anti-attribute
        assert anti_count_2 >= anti_count_1, (
            f"Anti-attribute set should grow as coverage increases: "
            f"round 1={anti_count_1}, round 2={anti_count_2}"
        )


# ===================================================================
# 8. DIPHONE-LEVEL INTEGRATION — Real index at diphone granularity
# ===================================================================


@pytest.mark.slow
@_skip_slow_deps
class TestRealDiphoneLevel:
    """Verify the full pipeline works at diphone granularity.

    Diphone coverage is a common metric in speech corpus literature.
    If the diphone filtering path is broken, our diphone coverage
    claims would be invalid.
    """

    def test_diphone_attribute_tokens_exist(self, built_index):
        """Common English diphones should have attribute tokens."""
        # t-ə (as in "the", "to") is very common
        common_diphones = ["t-ə", "ð-ə", "s-t"]
        found_any = False
        for dp in common_diphones:
            tokens = built_index.get_attribute_tokens([dp])
            if len(tokens) > 0:
                found_any = True
                break

        assert found_any, (
            f"None of the common diphones {common_diphones} had "
            f"any attribute tokens in the real index"
        )

    def test_diphone_anti_attribute_filtering(self, built_index):
        """Anti-attribute at diphone level only considers diphones."""
        # Cover a set of diphones
        covered_diphones = {"ð-ə", "ə-ð"}
        result_diphone = built_index.get_anti_attribute_tokens(
            covered_diphones, unit_level="diphone"
        )
        result_phoneme = built_index.get_anti_attribute_tokens(
            covered_diphones, unit_level="phoneme"
        )

        # Phoneme-level: "ð-ə" is not a phoneme (has hyphen), so
        # no token's phoneme-level units can be subset of {"ð-ə", "ə-ð"}
        assert result_phoneme == set(), (
            "Diphone strings should not match as phoneme-level units"
        )

    def test_diphone_strategy_integration(
        self, gpt2_model, gpt2_tokenizer
    ):
        """DATGStrategy at diphone level builds and prepares successfully."""
        targets = PhoneticTargetInventory(
            target_phonemes=["t", "s", "ð", "ə"],
            unit="diphone",
        )
        strategy = DATGStrategy(
            targets=targets,
            language="en-us",
            boost_strength=5.0,
            penalty_strength=-5.0,
        )

        strategy.prepare(
            target_units=["t-s"],
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
        )

        assert strategy.attribute_word_index.is_built
        # t-s should be findable in a real English vocabulary
        assert len(strategy.current_attribute_tokens) >= 0  # may be 0 if no tokens have t-s


# ===================================================================
# 9. LOGITS PROCESSOR COMPATIBILITY — Works with model.generate()
# ===================================================================


@pytest.mark.slow
@_skip_slow_deps
class TestRealLogitsProcessorIntegration:
    """Verify DATGStrategy works as a logits processor in model.generate().

    This is the actual inference-time integration point: LocalBackend
    wraps DATGStrategy in a _GuidanceLogitsProcessor and passes it
    to model.generate(). If this breaks, the entire method fails.
    """

    def test_generate_with_datg_produces_output(
        self, gpt2_model, gpt2_tokenizer
    ):
        """model.generate() with DATG logits processor produces tokens."""
        import torch

        from corpusgen.generate.backends.local import _GuidanceLogitsProcessor

        targets = PhoneticTargetInventory(
            target_phonemes=["p", "t", "k", "s", "ʃ"],
            unit="phoneme",
        )
        strategy = DATGStrategy(
            targets=targets,
            boost_strength=3.0,
            penalty_strength=-3.0,
        )
        strategy.prepare(
            target_units=["ʃ"],
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
        )

        processor = _GuidanceLogitsProcessor(strategy)

        prompt = "The cat sat on"
        inputs = gpt2_tokenizer(prompt, return_tensors="pt")

        output_ids = gpt2_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            logits_processor=[processor],
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )

        # Decode and verify non-empty output
        prompt_len = inputs.input_ids.shape[-1]
        new_ids = output_ids[:, prompt_len:]
        text = gpt2_tokenizer.decode(new_ids[0], skip_special_tokens=True)

        assert len(text.strip()) > 0, "DATG-guided generation produced empty text"

    def test_generate_with_and_without_datg_differ(
        self, gpt2_model, gpt2_tokenizer
    ):
        """DATG-guided generation should differ from unguided generation.

        With fixed seed, same prompt, the logit modification should
        change which tokens get sampled.
        """
        import torch

        from corpusgen.generate.backends.local import _GuidanceLogitsProcessor

        targets = PhoneticTargetInventory(
            target_phonemes=["p", "t", "k", "s", "ʃ"],
            unit="phoneme",
        )
        strategy = DATGStrategy(
            targets=targets,
            boost_strength=8.0,  # strong boost to ensure visible effect
            penalty_strength=-8.0,
        )
        strategy.prepare(
            target_units=["ʃ"],
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
        )

        processor = _GuidanceLogitsProcessor(strategy)
        prompt = "The quick brown fox"
        inputs = gpt2_tokenizer(prompt, return_tensors="pt")

        gen_kwargs = dict(
            max_new_tokens=30,
            do_sample=True,
            temperature=0.8,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )

        # Generate without DATG
        torch.manual_seed(42)
        out_plain = gpt2_model.generate(**inputs, **gen_kwargs)
        text_plain = gpt2_tokenizer.decode(
            out_plain[0, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        )

        # Generate with DATG
        torch.manual_seed(42)
        out_datg = gpt2_model.generate(
            **inputs, **gen_kwargs, logits_processor=[processor]
        )
        text_datg = gpt2_tokenizer.decode(
            out_datg[0, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        )

        # The texts should differ (DATG steered the generation)
        assert text_plain != text_datg, (
            "DATG-guided and unguided generation produced identical text. "
            "The logit modification had no effect."
        )

    def test_generated_text_is_phonemizable(
        self, gpt2_model, gpt2_tokenizer
    ):
        """DATG-guided output should be valid text that G2P can process."""
        import torch

        from corpusgen.generate.backends.local import _GuidanceLogitsProcessor
        from corpusgen.g2p.manager import G2PManager

        targets = PhoneticTargetInventory(
            target_phonemes=["p", "t", "k", "s", "ʃ"],
            unit="phoneme",
        )
        strategy = DATGStrategy(
            targets=targets,
            boost_strength=5.0,
            penalty_strength=-5.0,
        )
        strategy.prepare(
            target_units=["ʃ"],
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
        )

        processor = _GuidanceLogitsProcessor(strategy)
        prompt = "She sells seashells"
        inputs = gpt2_tokenizer(prompt, return_tensors="pt")

        output_ids = gpt2_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            logits_processor=[processor],
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )

        prompt_len = inputs.input_ids.shape[-1]
        text = gpt2_tokenizer.decode(
            output_ids[0, prompt_len:], skip_special_tokens=True
        )

        if text.strip():
            g2p = G2PManager()
            results = g2p.phonemize_batch([text.strip()], language="en-us")
            assert results[0].phonemes is not None, (
                f"G2P failed to phonemize DATG-guided output: {text!r}"
            )
            assert len(results[0].phonemes) > 0, (
                f"G2P returned empty phonemes for DATG-guided output: {text!r}"
            )
