# corpusgen

**Language-agnostic framework for generating and evaluating speech corpora with maximal phoneme coverage.**

`corpusgen` helps you build phonetically-balanced text corpora for speech synthesis (TTS), speech recognition (ASR), and clinical speech assessment — in any language.

## Features

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

- **Phon-CTG generation framework** — orchestrated corpus generation with pluggable backends:
  - **Repository backend** — select from sentence pools (pre-phonemized, raw text, or HuggingFace datasets)
  - **LLM API backend** — generate targeted sentences via OpenAI/Anthropic/Ollama (BYO API key)
  - **Local model backend** — HuggingFace transformers with CUDA auto-detect and 4-bit/8-bit quantization
- **Phon-DATG** — inference-time logit steering for phonetically-targeted local generation
- **Phon-RL** — PPO-based policy fine-tuning with composite phonetic reward (custom implementation, no trl dependency)
- **Built-in scorers** — n-gram phonotactic naturalness + LM perplexity fluency scoring
- **CLI** — `corpusgen evaluate`, `corpusgen select`, `corpusgen inventory`, `corpusgen generate` from the command line

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

### From PyPI

```bash
pip install corpusgen
```

### Development setup

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

## Quick Start

### Evaluate a corpus for phoneme coverage

```python
import corpusgen

report = corpusgen.evaluate(
    ["The quick brown fox jumps over the lazy dog.",
     "She sells seashells by the seashore.",
     "Pack my box with five dozen liquor jugs."],
    language="en-us",
    target_phonemes="phoible",
)

print(report.render())
print(report.coverage)           # 0.65
print(report.missing_phonemes)   # {'ʒ', 'ð', 'θ', ...}
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
]

result = corpusgen.select_sentences(
    candidates,
    language="en-us",
    algorithm="greedy",  # or "celf", "stochastic", "ilp", "distribution", "nsga2"
)

print(f"Selected {result.num_selected} of {len(candidates)} sentences")
print(f"Coverage: {result.coverage:.1%}")
```

### Generate a corpus from a sentence pool

The fastest way is the CLI:

```bash
# Select best sentences from a pool for maximal phoneme coverage
corpusgen generate -b repository -l en-us --file pool.txt --max-sentences 50

# With multi-objective scoring (coverage + phonotactic naturalness)
corpusgen generate -b repository -l en-us --file pool.txt \
  --coverage-weight 0.7 --phonotactic-weight 0.3 --phonotactic-scorer ngram
```

Or use the Python API for full control:

```python
from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_ctg.scorer import PhoneticScorer
from corpusgen.generate.phon_ctg.loop import GenerationLoop, StoppingCriteria
from corpusgen.generate.backends.repository import RepositoryBackend
from corpusgen.g2p.manager import G2PManager

# 1. Phonemize a sentence pool
g2p = G2PManager()
sentences = ["The cat sat on the mat.", "Big dogs bark loudly.", ...]
results = g2p.phonemize_batch(sentences, language="en-us")
pool = [
    {"text": s, "phonemes": r.phonemes}
    for s, r in zip(sentences, results) if r.phonemes
]

# 2. Set up targets, scorer, and backend
targets = PhoneticTargetInventory(
    target_phonemes=["p", "b", "t", "d", "k", "g"],
    unit="phoneme",
)
scorer = PhoneticScorer(targets=targets, coverage_weight=1.0)
backend = RepositoryBackend(pool=pool)

# 3. Run the generation loop
loop = GenerationLoop(
    backend=backend,
    targets=targets,
    scorer=scorer,
    stopping_criteria=StoppingCriteria(
        target_coverage=0.9,
        max_sentences=20,
    ),
)
result = loop.run()

print(f"Generated {result.num_generated} sentences, coverage: {result.coverage:.1%}")
```

### Generate with an LLM API

Via CLI:

```bash
# Requires: poetry install --with llm
corpusgen generate -b llm_api -l en-us --model openai/gpt-4o-mini --max-sentences 20
```

Or Python API:

```python
from corpusgen.generate.backends.llm_api import LLMBackend

# Requires: poetry install --with llm
# Set your API key: export OPENAI_API_KEY=...
backend = LLMBackend(
    model="gpt-4o-mini",
    language="en-us",
)

# Use with the same GenerationLoop as above
loop = GenerationLoop(
    backend=backend,
    targets=targets,
    scorer=scorer,
    stopping_criteria=StoppingCriteria(target_coverage=0.9),
)
result = loop.run()
```

### Fine-tune a model with phonetic reward (Phon-RL)

```python
from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_rl.reward import PhoneticReward
from corpusgen.generate.phon_rl.trainer import PhonRLTrainer, TrainingConfig

# Requires: poetry install --with local

# 1. Define targets and reward
targets = PhoneticTargetInventory(
    target_phonemes=["p", "b", "t", "d", "k"],
    unit="phoneme",
)
reward = PhoneticReward(targets=targets, coverage_weight=1.0)

# 2. Configure PPO training
config = TrainingConfig(
    model_name="gpt2",
    num_steps=100,
    learning_rate=1e-5,
    kl_coeff=0.1,
    use_peft=True,     # LoRA for parameter-efficient training
    peft_r=8,
    peft_alpha=16,
    device=None,        # auto-detect GPU
)

# 3. Train with dynamic prompts that adapt to coverage gaps
def make_prompt(targets):
    missing = targets.next_targets(5)
    return f"Write a sentence using these sounds: {', '.join(missing)}"

trainer = PhonRLTrainer(reward=reward, config=config)
result = trainer.train(prompt_fn=make_prompt)

print(f"Final coverage: {result.final_coverage:.1%}")
trainer.save_checkpoint("./phon_rl_checkpoint")
```

### Use PHOIBLE inventories directly

```python
from corpusgen import get_inventory

inv = get_inventory("en-us")
print(inv.language_name)          # 'English'
print(inv.consonants)             # ['p', 'b', 't', 'd', 'k', ...]
print(inv.vowels)                 # ['iː', 'ɪ', 'ɛ', 'æ', ...]

# Query by distinctive features
nasals = inv.segments_with_feature("nasal", "+")
print([s.phoneme for s in nasals])  # ['m', 'n', 'ŋ']
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

# Human-readable at different verbosity levels
from corpusgen.evaluate.report import Verbosity
print(report.render(verbosity=Verbosity.MINIMAL))
print(report.render(verbosity=Verbosity.NORMAL))
print(report.render(verbosity=Verbosity.VERBOSE))
```

## CLI Usage

```bash
# Show PHOIBLE phoneme inventory for a language
corpusgen inventory --language en-us
corpusgen inventory --language fr-fr --format json
corpusgen inventory --language en-us --source upsid

# Evaluate a corpus for phoneme coverage
corpusgen evaluate "The cat sat on the mat." --language en-us
corpusgen evaluate --file corpus.txt --language en-us --target phoible
corpusgen evaluate --file corpus.txt -l en-us --unit diphone --format json
corpusgen evaluate --file corpus.txt -l en-us --verbosity verbose

# Select optimal sentences from a candidate pool
corpusgen select --file candidates.txt --language en-us
corpusgen select -f pool.txt -l en-us --algorithm celf --max-sentences 50
corpusgen select -f pool.txt -l en-us --target phoible --target-coverage 0.95
corpusgen select -f pool.txt -l en-us --output selected.txt --format json

# Generate sentences targeting phoneme coverage
# --- Repository backend (sentence pool) ---
corpusgen generate -b repository -l en-us --file pool.txt --max-sentences 50
corpusgen generate -b repository -l en-us --file pool.txt --unit diphone --format json
corpusgen generate -b repository -l en-us --file pool.txt --phonemes "ʃ,ʒ,θ" --weights "ʃ:2.0,θ:1.5"
corpusgen generate -b repository -l en-us --file pool.txt --output generated.txt

# --- Repository backend with HuggingFace dataset ---
corpusgen generate -b repository -l en-us --dataset wikitext --split train --max-samples 1000

# --- LLM API backend (requires API key) ---
corpusgen generate -b llm_api -l en-us --model openai/gpt-4o-mini --max-sentences 20
corpusgen generate -b llm_api -l en-us --model openai/gpt-4o-mini --api-key sk-... --llm-temperature 0.9

# --- Local model backend (requires torch) ---
corpusgen generate -b local -l en-us --model gpt2 --device cuda --max-sentences 30
corpusgen generate -b local -l en-us --model gpt2 --quantization 4bit --local-temperature 0.7

# --- With built-in scorers (multi-objective candidate ranking) ---
corpusgen generate -b repository -l en-us --file pool.txt \
  --coverage-weight 0.6 \
  --phonotactic-weight 0.3 --phonotactic-scorer ngram \
  --fluency-weight 0.1 --fluency-scorer perplexity --fluency-model gpt2

# --- With corpus-trained phonotactic model ---
corpusgen generate -b repository -l en-us --file pool.txt \
  --phonotactic-weight 0.3 --phonotactic-scorer ngram \
  --phonotactic-corpus reference.txt --phonotactic-n 3

# --- With guidance strategies (local backend only) ---
corpusgen generate -b local -l en-us --model gpt2 --guidance datg --datg-boost 5.0
corpusgen generate -b local -l en-us --model gpt2 --guidance rl --rl-adapter-path ./checkpoint
corpusgen generate -b local -l en-us --model gpt2 --guidance datg --guidance-config datg.json

# --- Custom prompt templates ---
corpusgen generate -b llm_api -l en-us --model openai/gpt-4o-mini \
  --prompt-template "Write {k} English sentences containing: {target_units}"
corpusgen generate -b llm_api -l en-us --model openai/gpt-4o-mini \
  --prompt-template prompt.txt
```

## Architecture

```
corpusgen/
├── cli/                  # Command-line interface
│   ├── evaluate.py       # corpusgen evaluate
│   ├── generate.py       # corpusgen generate
│   ├── inventory.py      # corpusgen inventory
│   └── select.py         # corpusgen select
├── g2p/                  # Grapheme-to-phoneme conversion
│   ├── manager.py        # G2PManager — multi-backend G2P (espeak-ng)
│   └── result.py         # G2PResult — phonemes, diphones, triphones
├── coverage/
│   └── tracker.py        # CoverageTracker — phoneme/diphone/triphone tracking
├── evaluate/
│   ├── evaluate.py       # evaluate() — top-level API
│   └── report.py         # EvaluationReport, Verbosity
├── inventory/
│   ├── models.py         # Segment (38 features), Inventory
│   ├── phoible.py        # PhoibleDataset — PHOIBLE loader/cache/query
│   └── mapping.py        # EspeakMapping — espeak ↔ ISO 639-3
├── select/
│   ├── greedy.py         # GreedySelector
│   ├── celf.py           # CELFSelector (lazy evaluation)
│   ├── stochastic.py     # StochasticGreedySelector
│   ├── ilp.py            # ILPSelector (exact, optional: pulp)
│   ├── distribution.py   # DistributionAwareSelector (KL-divergence)
│   └── nsga2.py          # NSGA2Selector (Pareto, optional: pymoo)
├── weights/              # Phoneme weighting strategies
├── generate/
│   ├── phon_ctg/         # Orchestration framework
│   │   ├── targets.py    # PhoneticTargetInventory
│   │   ├── scorer.py     # PhoneticScorer (coverage + phonotactic + fluency)
│   │   ├── constraints.py # PhonotacticConstraint ABC + N-gram model
│   │   └── loop.py       # GenerationLoop + StoppingCriteria
│   ├── scorers/          # Built-in scoring functions
│   │   ├── phonotactic.py # NgramPhonotacticScorer (save/load, corpus-trained)
│   │   └── fluency.py    # PerplexityFluencyScorer (lazy LM, model sharing)
│   ├── phon_rl/          # RL-based guidance (PPO)
│   │   ├── reward.py     # PhoneticReward (composite, hierarchical)
│   │   ├── trainer.py    # PhonRLTrainer (custom PPO, no trl)
│   │   ├── policy.py     # PhonRLStrategy (GuidanceStrategy wrapper)
│   │   └── value_head.py # ValueHead (nn.Module for GAE)
│   ├── phon_datg/        # Inference-time logit steering
│   │   ├── attribute_words.py  # Vocabulary phonemization + index
│   │   ├── modulator.py  # Additive logit modulation
│   │   └── graph.py      # DATGStrategy (GuidanceStrategy)
│   ├── guidance.py       # GuidanceStrategy ABC
│   └── backends/         # Pluggable generation engines
│       ├── repository.py # Sentence pool selection + HuggingFace datasets
│       ├── llm_api.py    # Multi-provider LLM API (litellm)
│       └── local.py      # HuggingFace transformers + quantization
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
