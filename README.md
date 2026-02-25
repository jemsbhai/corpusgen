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

### Coming Soon

- 📚 **Repository-based generation** — curated sentence banks on HuggingFace Hub
- 🤖 **LLM API generation** — targeted sentences via OpenAI/Anthropic/Ollama
- 🧠 **Phon-CTG** — fine-tuned local model for phoneme-targeted generation (the core research contribution)
- **Selection algorithms** — greedy/CELF sentence selection with coverage optimization
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
├── generate/             # (Phase 3-5: repository, LLM, local)
├── select/               # (Phase 3: greedy/CELF selection)
└── weights/              # (Phase 3: phoneme weighting)
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
