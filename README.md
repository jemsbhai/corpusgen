# corpusgen

**Language-agnostic framework for generating and evaluating speech corpora with maximal phoneme coverage.**

`corpusgen` helps you build phonetically-balanced text corpora for speech synthesis (TTS), speech recognition (ASR), and clinical speech assessment — in any language.

## Features

- **Evaluate** any text corpus against phoneme coverage targets
- **Generate** phonetically-balanced corpora using three strategies:
  - 📚 **Repository**: Select from curated multilingual sentence banks (offline, deterministic)
  - 🤖 **LLM API**: Generate targeted sentences via OpenAI/Anthropic/Ollama
  - 🧠 **Local Model**: Fine-tuned Phon-CTG model on your own GPU
- **Multi-dialect support**: Handles pronunciation variants automatically
- **Flexible targeting**: Balanced, natural-distribution, or custom-weighted phoneme distributions
- **100+ languages** via PHOIBLE inventories and multi-backend G2P

## Installation

```bash
# Core (evaluation + selection + rule-based G2P)
pip install corpusgen

# With HuggingFace sentence bank access
pip install corpusgen[repository]

# With LLM generation
pip install corpusgen[llm]

# Everything
pip install corpusgen[all]
```

## Quick Start

```python
from corpusgen import evaluate, generate

# Evaluate an existing corpus
report = evaluate(
    text=["The quick brown fox jumps over the lazy dog."],
    language="en",
)
print(report.coverage)       # 0.42 (42% of English phonemes covered)
print(report.missing)        # ['ʒ', 'θ', 'ð', ...]

# Generate a phonetically-balanced corpus
corpus = generate(
    language="en",
    strategy="repository",   # or "llm" or "local"
    target_coverage=0.95,
    n_sentences=200,
)
print(corpus.sentences)
print(corpus.coverage)       # 0.97
```

## Documentation

*Coming soon*

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
