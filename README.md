# corpusgen

[![DOI](https://zenodo.org/badge/1021961235.svg)](https://doi.org/10.5281/zenodo.18881479)
[![PyPI](https://img.shields.io/pypi/v/corpusgen.svg)](https://pypi.org/project/corpusgen/)
[![Python](https://img.shields.io/pypi/pyversions/corpusgen.svg)](https://pypi.org/project/corpusgen/)
[![CI](https://github.com/jemsbhai/corpusgen/actions/workflows/ci.yml/badge.svg)](https://github.com/jemsbhai/corpusgen/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://jemsbhai.github.io/corpusgen/)

**Language-agnostic framework for generating and evaluating speech corpora with maximal phoneme coverage.**

`corpusgen` helps you build phonetically-balanced text corpora for speech
synthesis (TTS), speech recognition (ASR), and clinical speech assessment
across many languages.

## Features

- **Evaluate** any text corpus for phoneme, diphone, or triphone coverage
- **PHOIBLE integration** — phoneme inventories for 2,186 languages (3,020 inventories)
- **Grapheme-to-phoneme** via espeak-ng for 100+ languages
- **Espeak ↔ PHOIBLE mapping** — seamless bridge between G2P and phonological databases
- **Distribution quality metrics** — Shannon entropy, normalized entropy, JSD (vs uniform or reference), Pearson correlation, coefficient of variation, PCD composite score
- **Coverage trajectory tracking** — step-by-step coverage saturation curves for any selection or generation result
- **Text quality metrics** — sentence length stats, vocabulary diversity (TTR, hapax ratio), Flesch readability scores
- **Error rate metrics** — WER, CER, PER, SER with per-sentence breakdowns and corpus-level micro-averaging
- **Corpus-level perplexity** — batched LM perplexity via GPT-2 (or any causal LM), both token-weighted corpus perplexity and sentence-weighted mean, with model sharing support
- **Structured reports** — three verbosity levels, JSON export, JSON-LD-EX compatibility
- **40-language test suite** — validated across 12 language families

- **6 selection algorithms** for corpus optimization:
  - **Greedy Set Cover** — ln(n)+1 approximation, the standard workhorse
  - **CELF** — submodularity-based lazy evaluation with the same greedy result
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
- **Built-in scorers** — n-gram phonotactics, LM perplexity, and API-level readability scoring
- **CLI** — `corpusgen evaluate`, `corpusgen select`, `corpusgen inventory`, `corpusgen generate` from the command line

## Prerequisites

### espeak-ng (required for G2P workflows)

`corpusgen` uses [espeak-ng](https://github.com/espeak-ng/espeak-ng) for
grapheme-to-phoneme conversion. Install it before running workflows that
phonemize raw text.

<details>
<summary><strong>Windows</strong></summary>

1. Download the `.msi` installer matching your Python architecture from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)
2. Run the installer (normally `C:\Program Files\eSpeak NG\` for 64-bit or
   `C:\Program Files (x86)\eSpeak NG\` for 32-bit)
3. The default DLL path is detected automatically. If you install elsewhere,
   set the shared-library override:

```powershell
[Environment]::SetEnvironmentVariable("PHONEMIZER_ESPEAK_LIBRARY", "C:\Program Files\eSpeak NG\libespeak-ng.dll", "User")
```

4. Restart your terminal and verify both the executable and Python backend:

```powershell
espeak-ng --version
python -c "from corpusgen.g2p import G2PManager; result = G2PManager().phonemize('hello'); print(len(result.phonemes), 'phonemes')"
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

### PHOIBLE data (required for inventory-based features)

`get_inventory()`, the `inventory` command, evaluation or selection with a
`phoible` target, and every current generation workflow require the pinned
PHOIBLE inventory dataset. Download and checksum-verify it once:

```python
from corpusgen.inventory import PhoibleDataset
PhoibleDataset().download()  # cached at ~/.corpusgen/phoible.csv (~24 MB)
```

This only needs to be done once. Evaluation and selection can instead derive
targets from observed or candidate phonemes and do not require PHOIBLE data.
Every current `generate` command resolves a PHOIBLE baseline; `--phonemes`
adds symbols to that baseline rather than replacing it.

## Installation

### From PyPI

```bash
python -m pip install --upgrade corpusgen
corpusgen --version
```

### Optional features

Install only the integrations you need:

| Extra | Enables |
|---|---|
| `llm` | LLM API generation with LiteLLM, OpenAI, and Anthropic |
| `local` | Local HuggingFace models, Phon-DATG, and Phon-RL |
| `repository` | HuggingFace dataset-backed repositories |
| `optimization` | ILP and NSGA-II selection |
| `eval` | Matplotlib support for plotting analysis results |
| `full` | All optional integrations |

```bash
python -m pip install "corpusgen[llm]"
# Or install every optional integration:
python -m pip install "corpusgen[full]"
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

For NVIDIA acceleration, use the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) to
choose the build matching your OS and driver. Run its generated command inside
the Poetry environment by replacing the leading `pip install` (or `pip3
install`) with `poetry run python -m pip install`. Supported CUDA build indexes
change over time.

```bash
# PyPI users:
python -m pip install "corpusgen[local]"

# Contributors working from a clone:
poetry install --with local

# Install the GPU-enabled PyTorch build using the selector-generated
# command as described above, then verify GPU access (PyPI install):
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Contributor equivalent:
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

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
print(f"Coverage: {report.coverage:.1%}")
print(sorted(report.missing_phonemes))
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
    algorithm="greedy",  # also: "celf" or "stochastic"
)

print(f"Selected {result.num_selected} of {len(candidates)} sentences")
print(f"Coverage: {result.coverage:.1%}")
```

ILP and NSGA-II require the `optimization` extra. Distribution-aware selection
also requires a `target_distribution`; see the [API documentation](https://jemsbhai.github.io/corpusgen/).

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
sentences = [
    "The cat sat on the mat.",
    "Big dogs bark loudly.",
    "Six green frogs jumped into the pond.",
]
results = g2p.phonemize_batch(sentences, language="en-us")
pool = [
    {"text": s, "phonemes": r.phonemes}
    for s, r in zip(sentences, results) if r.phonemes
]

# 2. Set up targets, scorer, and backend
targets = PhoneticTargetInventory(
    target_phonemes=["p", "b", "t", "d", "k", "ɡ"],
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
# Requires: python -m pip install "corpusgen[llm]"
# Contributor equivalent: poetry install --with llm
corpusgen generate -b llm_api -l en-us --model openai/gpt-4o-mini --max-sentences 20
```

Or Python API:

```python
from corpusgen.generate.backends.llm_api import LLMBackend
from corpusgen.generate.phon_ctg.loop import GenerationLoop, StoppingCriteria
from corpusgen.generate.phon_ctg.scorer import PhoneticScorer
from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory

# Requires: python -m pip install "corpusgen[llm]"
# Set your API key: export OPENAI_API_KEY=...
backend = LLMBackend(
    model="gpt-4o-mini",
    language="en-us",
)

# This is an alternative setup, so start with a fresh target state.
llm_targets = PhoneticTargetInventory(
    target_phonemes=["p", "b", "t", "d", "k", "ɡ"],
    unit="phoneme",
)
llm_scorer = PhoneticScorer(targets=llm_targets, coverage_weight=1.0)
loop = GenerationLoop(
    backend=backend,
    targets=llm_targets,
    scorer=llm_scorer,
    stopping_criteria=StoppingCriteria(target_coverage=0.9, max_sentences=20),
)
result = loop.run()
```

### Fine-tune a model with phonetic reward (Phon-RL)

```python
from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_rl.reward import PhoneticReward
from corpusgen.generate.phon_rl.trainer import PhonRLTrainer, TrainingConfig

# Requires: python -m pip install "corpusgen[local]"

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
print(inv.language_name)
print(inv.consonants)
print(inv.vowels)

# Query by distinctive features
nasals = inv.segments_with_feature("nasal", "+")
print([s.phoneme for s in nasals])
```

An eSpeak voice such as `en-us` is mapped to an ISO 639-3 code, after which
`get_inventory()` returns the largest matching PHOIBLE inventory by default.

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

### Analyze distribution quality

```python
import corpusgen

report = corpusgen.evaluate(
    ["The cat sat on the mat.", "Big dogs dig deep holes."],
    language="en-us",
    target_phonemes="phoible",
)

# Distribution metrics are auto-computed
dm = report.distribution
print(f"Normalized entropy: {dm.normalized_entropy:.4f}")  # 1.0 = perfectly uniform
print(f"JSD vs uniform: {dm.jsd_uniform:.6f}")             # 0.0 = perfectly uniform
print(f"PCD (uniform): {dm.pcd_uniform:.4f}")              # coverage × (1 - JSD)

# Compare against a natural language reference distribution
from corpusgen.evaluate.distribution import compute_distribution_metrics

reference = {"p": 0.04, "t": 0.07, "k": 0.03, "ə": 0.12}  # example frequencies
dm_ref = compute_distribution_metrics(
    report.phoneme_counts, report.target_phonemes, reference_distribution=reference
)
print(f"JSD vs reference: {dm_ref.jsd_reference:.6f}")
print(f"Pearson correlation: {dm_ref.pearson_correlation}")
```

### Plot coverage saturation curves

```python
from corpusgen.evaluate.trajectory import compute_coverage_trajectory
from corpusgen.g2p import G2PManager

sentences = ["The cat sat on the mat.", "Big dogs dig deep holes."]
g2p = G2PManager()
sequences = [g2p.phonemize(text, language="en-us").phonemes for text in sentences]
target_units = {phoneme for sequence in sequences for phoneme in sequence}
traj = compute_coverage_trajectory(
    sequences,
    target_units=target_units,
    unit="phoneme",
)

# Easy plotting
import matplotlib.pyplot as plt
plt.plot(range(len(traj.coverages)), traj.coverages)
plt.xlabel("Sentences")
plt.ylabel("Coverage")
plt.title("Coverage Saturation Curve")
plt.show()

# Access marginal gains per sentence
print(traj.gains)
```

### Evaluate text quality

```python
import corpusgen

report = corpusgen.evaluate(
    ["The cat sat on the mat.", "Big dogs dig deep holes."],
    language="en-us",
)

# Text quality metrics are auto-computed
tq = report.text_quality
print(f"Type-Token Ratio: {tq.type_token_ratio:.3f}")
print(f"Flesch Reading Ease: {tq.flesch_reading_ease:.1f}")
print(f"Avg sentence length: {tq.sentence_length_words_mean:.1f} words")
```

### Measure corpus perplexity

```python
from corpusgen.evaluate.perplexity import compute_corpus_perplexity

# Simple — loads GPT-2 automatically
# Requires: python -m pip install "corpusgen[local]"
metrics = compute_corpus_perplexity(
    ["The cat sat on the mat.", "Big dogs dig deep holes."],
    model_name="gpt2",
)

print(f"Corpus perplexity: {metrics.corpus_perplexity:.2f}")  # token-weighted (standard LM metric)
print(f"Mean perplexity:   {metrics.mean_perplexity:.2f}")    # sentence-weighted
print(f"Median:            {metrics.median_perplexity:.2f}")
print(f"Total tokens:      {metrics.num_tokens}")

# Per-sentence breakdown
for i, ppl in enumerate(metrics.per_sentence):
    print(f"  Sentence {i}: PPL = {ppl:.2f}")

# Shared model — inject the same public model and tokenizer objects into both APIs:
from transformers import AutoModelForCausalLM, AutoTokenizer
from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
scorer = PerplexityFluencyScorer.from_model(model, tokenizer)

metrics = compute_corpus_perplexity(
    ["The cat sat on the mat.", "Big dogs dig deep holes."],
    model=model,
    tokenizer=tokenizer,
)
```

### Compare transcriptions with error rates

```python
from corpusgen.evaluate.error_rates import compute_error_rates

result = compute_error_rates(
    references=["the cat sat on the mat", "big dogs dig deep holes"],
    hypotheses=["the cat sat on a mat", "big dog dig deep hole"],
)

print(f"WER: {result.wer:.2%}")   # corpus-level, micro-averaged
print(f"CER: {result.cer:.2%}")
print(f"SER: {result.ser:.2%}")

# With phoneme-level comparison
result = compute_error_rates(
    references=["the cat"],
    hypotheses=["a cat"],
    reference_phonemes=[["\u00f0", "\u0259", "k", "\u00e6", "t"]],
    hypothesis_phonemes=[["\u0259", "k", "\u00e6", "t"]],
)
print(f"PER: {result.per:.2%}")

# Per-sentence breakdown
for d in result.details:
    print(f"  [{d.index}] WER={d.wer:.2%} CER={d.cer:.2%}")
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
corpusgen select -f pool.txt -l en-us --algorithm distribution --target-distribution '{"p":0.6,"t":0.4}'
corpusgen select -f pool.txt -l en-us --output selected.txt --format json

# Generate sentences targeting phoneme coverage
# --- Repository backend (sentence pool) ---
corpusgen generate -b repository -l en-us --file pool.txt --max-sentences 50
corpusgen generate -b repository -l en-us --file pool.txt --unit diphone --format json
corpusgen generate -b repository -l en-us --file pool.txt --phonemes "ʃ,ʒ,θ" --weights "ʃ:2.0,θ:1.5"
corpusgen generate -b repository -l en-us --file pool.txt --output generated.txt

# --- Repository backend with HuggingFace dataset ---
# Requires: python -m pip install "corpusgen[repository]"
corpusgen generate -b repository -l en-us --dataset ag_news --split train --max-samples 1000

# --- LLM API backend (requires API key) ---
corpusgen generate -b llm_api -l en-us --model openai/gpt-4o-mini --max-sentences 20
corpusgen generate -b llm_api -l en-us --model openai/gpt-4o-mini --api-key sk-... --llm-temperature 0.9 --max-sentences 20

# --- Local model backend (requires torch) ---
corpusgen generate -b local -l en-us --model gpt2 --device cuda --max-sentences 30
corpusgen generate -b local -l en-us --model gpt2 --quantization 4bit --local-temperature 0.7 --max-sentences 30

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
corpusgen generate -b local -l en-us --model gpt2 --guidance datg --datg-boost 5.0 --max-sentences 30
corpusgen generate -b local -l en-us --model gpt2 --guidance rl --rl-adapter-path ./checkpoint --max-sentences 30
corpusgen generate -b local -l en-us --model gpt2 --guidance datg --guidance-config datg.json --max-sentences 30

# --- Custom prompt templates ---
corpusgen generate -b llm_api -l en-us --model openai/gpt-4o-mini \
  --prompt-template "Write {k} English sentences containing: {target_units}" --max-sentences 20
corpusgen generate -b llm_api -l en-us --model openai/gpt-4o-mini \
  --prompt-template prompt.txt --max-sentences 20
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
│   ├── report.py         # EvaluationReport, Verbosity
│   ├── distribution.py   # DistributionMetrics — JSD, entropy, PCD, Pearson
│   ├── trajectory.py     # CoverageTrajectory — step-by-step saturation curves
│   ├── text_quality.py   # TextQualityMetrics — TTR, readability, sentence stats
│   ├── error_rates.py    # WER, CER, PER, SER with edit distance
│   └── perplexity.py     # Corpus-level perplexity (batched, GPU-accelerated)
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
│   │   ├── fluency.py    # PerplexityFluencyScorer (lazy LM, model sharing)
│   │   └── readability.py # ReadabilityScorer (Python API hook)
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

G2P supports eSpeak NG voices. PHOIBLE-targeted workflows additionally require
a voice represented in corpusgen's bundled eSpeak-to-ISO mapping, or a direct
ISO/Glottocode inventory lookup where the API accepts one.

- **G2P (espeak-ng):** 100+ languages
- **Inventories (PHOIBLE):** 2,186 languages, 3,020 inventories, 8 sources
- **Tested across:** 40 languages, 12 language families, 10+ writing systems

The bundled mapping covers 131 eSpeak voice codes and 115 ISO codes, with
automatic macrolanguage resolution (for example, `ms` maps to Standard Malay).

## Reproducibility

For reproducible results across machines:

1. **Pin corpusgen version** in your dependency file
2. **Pin espeak-ng version**: Record `espeak-ng --version` in experiment logs
3. **Use `poetry.lock`**: Pins all transitive dependencies
4. **Record PHOIBLE version**: corpusgen downloads a pinned, checksum-verified revision; record your corpusgen version alongside experiments

## Citation

If you use `corpusgen` in your research, please cite:

```bibtex
@software{corpusgen2026,
  title={corpusgen: Language-Agnostic Speech Corpus Generation with Maximal Phoneme Coverage},
  author={Syed, Muntaser and Silaghi, Marius and Abujar, Sheikh and Khushbu, Sharun Akter and Jaigirdar, Fariha Tasmin},
  year={2026},
  doi={10.5281/zenodo.18881479},
  url={https://github.com/jemsbhai/corpusgen}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
