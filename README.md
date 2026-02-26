# corpusgen

**Language-agnostic framework for generating and evaluating speech corpora with maximal phoneme coverage.**

`corpusgen` helps you build phonetically-balanced text corpora for speech synthesis (TTS), speech recognition (ASR), and clinical speech assessment — in any language.

## What's Working Now (v0.1.0-alpha)

- **Evaluate** any text corpus for phoneme, diphone, or triphone coverage
- **PHOIBLE integration** — phoneme inventories for 2,186 languages (3,020 inventories)
- **Grapheme-to-phoneme** via espeak-ng for 100+ languages
- **Espeak ↔ PHOIBLE mapping** — seamless bridge between G2P and phonological databases
- **Structured reports** — three verbosity levels, JSON export, JSON-LD-EX compatibility
- **40-language test suite** — validated across 12 language families

- **6 selection algorithms** for corpus optimization:
  - **Greedy Set Cover** — ln(n)+1 approximation, the standard workhorse
  - **CELF** — lazy evaluation speedup, identical results up to 700× faster
  - **Stochastic Greedy** — (1-1/e-ε) approximation, scales to massive corpora
  - **ILP** — exact optimal solutions via Integer Linear Programming (ground truth)
  - **Distribution-Aware** — KL-divergence minimization for frequency matching
  - **NSGA-II** — multi-objective Pareto optimization (coverage × cost × distribution)
- **Phoneme weighting** — uniform, frequency-inverse, and linguistic class strategies

### Coming Soon

- 📚 **Repository-based generation** — curated sentence banks on HuggingFace Hub
- 🤖 **LLM API generation** — targeted sentences via OpenAI/Anthropic/Ollama
- 🧠 **Phon-CTG** — fine-tuned local model for phoneme-targeted generation (the core research contribution)
- **CLI** — command-line interface for all operations

## Prerequisites

### espeak-ng (required)

`corpusgen` uses [espeak-ng](https://github.com/espeak-ng/espeak-ng) for grapheme-to-phoneme conversion. Install it before using corpusgen.

<details>
<summary><strong>Windows</strong></summary>

1. Download the latest `.msi` installer from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)
2. Run the installer (default path: `C:\Program Files\eSpeak NG\`)
3. Set the environment variable so Python can find the shared library:

```powershell
[Environment]::SetEnvironmentVariable("PHONEMIZER_ESPEAK_LIBRARY", "C:\Program Files\eSpeak NG\libespeak-ng.dll", "User")
```

4. Restart your terminal and verify:

```powershell
espeak-ng --version
```

</details>

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install espeak-ng
```

</details>

<details>
<summary><strong>Linux (Ubuntu/Debian)</strong></summary>

```bash
sudo apt-get update && sudo apt-get install -y espeak-ng
```

</details>

<details>
<summary><strong>Docker / CI</strong></summary>

```dockerfile
RUN apt-get update && apt-get install -y espeak-ng && rm -rf /var/lib/apt/lists/*
```

</details>

### PHOIBLE data (recommended)

To use PHOIBLE phoneme inventories (2,186 languages), download the data on first use:

```python
from corpusgen.inventory import PhoibleDataset
PhoibleDataset().download()  # cached at ~/.corpusgen/phoible.csv (~24 MB)
```

This only needs to be done once.

## Installation

### Development setup (current)

```bash
git clone https://github.com/jemsbhai/corpusgen.git
cd corpusgen
poetry install
poetry run pytest
```

### With local model support (GPU recommended)

For Phon-RL training and Phon-DATG logit steering with local models:

```bash
# 1. Install corpusgen with local model dependencies
poetry install --with local

# 2. IMPORTANT: Replace CPU torch with CUDA torch for GPU acceleration.
#    The default Poetry install pulls CPU-only torch from PyPI.
#    For NVIDIA GPUs (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# Verify GPU is available:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

> **Note:** Check [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct CUDA version matching your driver. Common options: `cu118`, `cu121`, `cu124`.

### PyPI (coming soon)

```bash
pip install corpusgen
```

## Quick Start

### Evaluate a corpus for phoneme coverage

```python
import corpusgen

# Evaluate against the PHOIBLE inventory for English
report = corpusgen.evaluate(
    ["The quick brown fox jumps over the lazy dog.",
     "She sells seashells by the seashore.",
     "Pack my box with five dozen liquor jugs."],
    language="en-us",
    target_phonemes="phoible",
)

print(report.render())
# Coverage: 65.0% (26/40 phonemes)
# Missing: ʒ, ð, θ, ...

print(report.coverage)           # 0.65
print(report.missing_phonemes)   # {'ʒ', 'ð', 'θ', ...}
print(report.covered_phonemes)   # {'p', 'b', 't', 'd', ...}
```

### Discover what phonemes are in your corpus

```python
import corpusgen

# No target = derive inventory from the text itself
report = corpusgen.evaluate(
    ["Hello world."],
    language="en-us",
)
print(report.target_phonemes)    # all phonemes found in the text
print(report.coverage)           # 1.0 (100% by definition)
```

### Use PHOIBLE inventories directly

```python
from corpusgen import get_inventory

# Get the PHOIBLE inventory via espeak code
inv = get_inventory("en-us")
print(inv.language_name)          # 'English'
print(inv.consonants)             # ['p', 'b', 't', 'd', 'k', ...]
print(inv.vowels)                 # ['iː', 'ɪ', 'ɛ', 'æ', ...]
print(inv.has_tones)              # False
print(inv.consonant_count)        # 27
print(inv.vowel_count)            # 13

# Query by distinctive features
nasals = inv.segments_with_feature("nasal", "+")
print([s.phoneme for s in nasals])  # ['m', 'n', 'ŋ']

# Get all inventories for a language (different sources/dialects)
from corpusgen.inventory import PhoibleDataset
ds = PhoibleDataset()
all_eng = ds.get_all_inventories("eng")
print(f"English has {len(all_eng)} inventories in PHOIBLE")

# Union inventory (maximally inclusive)
union = ds.get_union_inventory("eng")
print(f"Union: {union.size} segments from all sources")
```

### Evaluate with diphone or triphone coverage

```python
import corpusgen

report = corpusgen.evaluate(
    ["The quick brown fox jumps."],
    language="en-us",
    target_phonemes="phoible",
    unit="diphone",
)
print(f"Diphone coverage: {report.coverage:.1%}")
```

### Work with multiple languages

```python
import corpusgen

for lang in ["en-us", "fr-fr", "de", "ar", "hi", "ja", "cmn"]:
    report = corpusgen.evaluate(
        ["Hello world."],  # espeak handles transliteration
        language=lang,
    )
    print(f"{lang}: {len(report.target_phonemes)} unique phonemes")
```

### Select optimal sentences from a candidate pool

```python
import corpusgen

candidates = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck?",
    "To be or not to be, that is the question.",
    # ... hundreds or thousands of candidates
]

# Greedy selection for maximal phoneme coverage
result = corpusgen.select_sentences(
    candidates,
    language="en-us",
    algorithm="greedy",  # or "celf", "stochastic", "ilp", "distribution", "nsga2"
)

print(f"Selected {result.num_selected} of {len(candidates)} sentences")
print(f"Coverage: {result.coverage:.1%}")
print(f"Missing: {result.missing_units}")

# Use budget constraints
result = corpusgen.select_sentences(
    candidates,
    language="en-us",
    algorithm="greedy",
    max_sentences=5,          # select at most 5
    target_coverage=0.9,      # or stop at 90% coverage
)
```

### Compare algorithms

```python
from corpusgen.select import GreedySelector, CELFSelector, ILPSelector

# Pre-phonemized for fair comparison (same G2P for all)
result_greedy = corpusgen.select_sentences(
    candidates, language="en-us", algorithm="greedy"
)
result_celf = corpusgen.select_sentences(
    candidates, language="en-us", algorithm="celf"
)
result_ilp = corpusgen.select_sentences(
    candidates, language="en-us", algorithm="ilp"
)

for r in [result_greedy, result_celf, result_ilp]:
    print(f"{r.algorithm:12s} | {r.num_selected} sentences | "
          f"{r.coverage:.1%} | {r.elapsed_seconds:.3f}s")
```

### Weighted selection (prioritize rare phonemes)

```python
import corpusgen
from corpusgen.weights import frequency_inverse_weights
from corpusgen.g2p.manager import G2PManager

# Phonemize candidates
g2p = G2PManager()
results = g2p.phonemize_batch(candidates, language="en-us")
candidate_phonemes = [r.phonemes for r in results]

# Build inverse-frequency weights (rare phonemes get higher weight)
all_phonemes = set()
for p in candidate_phonemes:
    all_phonemes.update(p)
weights = frequency_inverse_weights(all_phonemes, candidate_phonemes)

result = corpusgen.select_sentences(
    candidates,
    target_phonemes=sorted(all_phonemes),
    candidate_phonemes=candidate_phonemes,
    algorithm="greedy",
    weights=weights,
)
```

### Export reports

```python
import corpusgen

report = corpusgen.evaluate(
    ["The quick brown fox."],
    language="en-us",
    target_phonemes="phoible",
)

# JSON
print(report.to_json(indent=2))

# JSON-LD (linked data)
doc = report.to_jsonld_ex()

# Python dict
d = report.to_dict()

# Human-readable at different verbosity levels
from corpusgen.evaluate.report import Verbosity
print(report.render(verbosity=Verbosity.MINIMAL))   # coverage + missing
print(report.render(verbosity=Verbosity.NORMAL))     # + per-phoneme counts
print(report.render(verbosity=Verbosity.VERBOSE))    # + per-sentence breakdown
```

## Architecture

```
corpusgen/
├── g2p/                  # Grapheme-to-phoneme conversion
│   ├── manager.py        # G2PManager — multi-backend G2P (espeak-ng)
│   └── result.py         # G2PResult — phonemes, diphones, triphones
├── coverage/
│   └── tracker.py        # CoverageTracker — phoneme/diphone/triphone tracking
├── evaluate/
│   ├── evaluate.py       # evaluate() — top-level user-facing API
│   └── report.py         # EvaluationReport, SentenceDetail, Verbosity
├── inventory/
│   ├── models.py         # Segment (38 features), Inventory
│   ├── phoible.py        # PhoibleDataset — PHOIBLE CSV loader/cache/query
│   └── mapping.py        # EspeakMapping — espeak ↔ ISO 639-3
├── data/
│   └── espeak_iso_mapping.json  # Bundled voice code mapping
├── select/
│   ├── base.py           # SelectorBase ABC
│   ├── result.py         # SelectionResult
│   ├── greedy.py         # GreedySelector
│   ├── celf.py           # CELFSelector
│   ├── stochastic.py     # StochasticGreedySelector
│   ├── ilp.py            # ILPSelector (optional: pulp)
│   ├── distribution.py   # DistributionAwareSelector
│   └── nsga2.py          # NSGA2Selector (optional: pymoo)
├── weights/              # Phoneme weighting strategies
├── generate/             # (Phase 4-5: repository, LLM, local)
```

## Language Support

`corpusgen` supports any language available in both espeak-ng and PHOIBLE:

- **G2P (espeak-ng):** 100+ languages
- **Inventories (PHOIBLE):** 2,186 languages, 3,020 inventories, 8 sources
- **Tested across:** 40 languages, 12 language families, 10+ scripts

The espeak-to-PHOIBLE mapping covers 85+ languages with automatic macrolanguage resolution (e.g., `ms` → Standard Malay, `sw` → Swahili).

## Reproducibility

For reproducible results across machines:

1. **Pin corpusgen version** in your dependency file
2. **Pin espeak-ng version**: Record `espeak-ng --version` in experiment logs
3. **Use `poetry.lock`**: Pins all transitive dependencies
4. **Record PHOIBLE version**: Note the download date of `~/.corpusgen/phoible.csv`

## Citation

If you use `corpusgen` in your research, please cite:

```bibtex
@software{corpusgen2025,
  title={corpusgen: Language-Agnostic Speech Corpus Generation with Maximal Phoneme Coverage},
  author={Syed, Muntaser},
  year={2025},
  url={https://github.com/jemsbhai/corpusgen}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
