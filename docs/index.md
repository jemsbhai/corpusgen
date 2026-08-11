# corpusgen

**Language-agnostic framework for generating and evaluating speech corpora with maximal phoneme coverage.**

`corpusgen` helps you build phonetically-balanced text corpora for speech
synthesis (TTS), speech recognition (ASR), and clinical speech assessment
across many languages.

## Key Capabilities

- **Evaluate** any text corpus for phoneme, diphone, or triphone coverage
- **Select** optimal sentence subsets using 6 algorithms (greedy, CELF, ILP, stochastic, distribution-aware, NSGA-II)
- **Generate** targeted sentences via LLM APIs, local models, or sentence pools
- **PHOIBLE integration** — phoneme inventories for 2,186 languages
- **Grapheme-to-phoneme** via espeak-ng for 100+ languages

## Installation

```bash
python -m pip install --upgrade corpusgen
corpusgen --version
```

G2P workflows require espeak-ng to be installed separately. See the
[README](https://github.com/jemsbhai/corpusgen#prerequisites) for
platform-specific instructions.

Inventory lookup and examples that use `target_phonemes="phoible"` also need
the pinned PHOIBLE dataset. Download and checksum-verify it once:

```bash
python -c "from corpusgen.inventory import PhoibleDataset; PhoibleDataset().download()"
```

Optional integrations are available as pip extras:

```bash
python -m pip install "corpusgen[llm]"          # LLM APIs
python -m pip install "corpusgen[local]"        # local models and Phon-RL
python -m pip install "corpusgen[repository]"   # HuggingFace datasets
python -m pip install "corpusgen[optimization]" # ILP and NSGA-II
```

## Quick Start

```python
import corpusgen

# Evaluate a corpus
report = corpusgen.evaluate(
    ["The quick brown fox jumps over the lazy dog."],
    language="en-us",
    target_phonemes="phoible",
)
print(f"Coverage: {report.coverage:.1%}")

# Select optimal sentences
result = corpusgen.select_sentences(
    candidates=["The cat sat on the mat.", "Big dogs dig deep holes."],
    language="en-us",
    algorithm="greedy",
)
print(f"Selected {result.num_selected} sentences")

# Explore phoneme inventories
inv = corpusgen.get_inventory("en-us")
print(f"{inv.language_name}: {inv.size} segments")
```

## Learn More

- [Examples](examples.md) — runnable scripts demonstrating core workflows
- [API Reference](api.md) — core and advanced public API documentation
- [Changelog](https://github.com/jemsbhai/corpusgen/blob/main/CHANGELOG.md) — release history
- [GitHub Repository](https://github.com/jemsbhai/corpusgen) — source code, issues, contributing guidelines
