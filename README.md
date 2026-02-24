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

## Prerequisites

### espeak-ng (required)

corpusgen uses [espeak-ng](https://github.com/espeak-ng/espeak-ng) as its core grapheme-to-phoneme engine. It converts text to IPA phonemes for 100+ languages. **You must install it before using corpusgen.**

<details>
<summary><strong>Windows</strong></summary>

1. Download the latest `.msi` installer from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)
2. Run the installer (default path: `C:\Program Files\eSpeak NG\`)
3. Set the environment variable so Python can find the shared library:

```powershell
# Permanent (restart terminal after running)
[Environment]::SetEnvironmentVariable("PHONEMIZER_ESPEAK_LIBRARY", "C:\Program Files\eSpeak NG\libespeak-ng.dll", "User")
```

4. Verify installation:

```powershell
espeak-ng --version
# Expected: eSpeak NG text-to-speech: 1.51+  (or newer)
```

</details>

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install espeak-ng
```

Verify:
```bash
espeak-ng --version
```

The library is typically found automatically at `/opt/homebrew/lib/libespeak-ng.dylib`. If not:
```bash
export PHONEMIZER_ESPEAK_LIBRARY=$(brew --prefix espeak-ng)/lib/libespeak-ng.dylib
```

Add the export line to your `~/.zshrc` or `~/.bashrc` to make it permanent.

</details>

<details>
<summary><strong>Linux (Ubuntu/Debian)</strong></summary>

```bash
sudo apt-get update
sudo apt-get install espeak-ng
```

Verify:
```bash
espeak-ng --version
```

The library is typically found automatically at `/usr/lib/x86_64-linux-gnu/libespeak-ng.so`.

</details>

<details>
<summary><strong>Linux (Fedora/RHEL)</strong></summary>

```bash
sudo dnf install espeak-ng
```

</details>

<details>
<summary><strong>Docker / CI</strong></summary>

```dockerfile
RUN apt-get update && apt-get install -y espeak-ng && rm -rf /var/lib/apt/lists/*
```

For GitHub Actions:
```yaml
- name: Install espeak-ng
  run: sudo apt-get update && sudo apt-get install -y espeak-ng
```

</details>

### Verifying the full setup

After installing espeak-ng, run this to confirm everything works:

```python
python -c "from phonemizer.backend import EspeakBackend; b = EspeakBackend('en-us'); print(b.phonemize(['hello world'])); print('OK')"
```

Expected output:
```
['həloʊ wɜːld ']
OK
```

If you see `RuntimeError: espeak not installed on your system`, the `PHONEMIZER_ESPEAK_LIBRARY` environment variable is not set correctly. See the platform-specific instructions above.

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

### Development setup

```bash
git clone https://github.com/jemsbhai/corpusgen.git
cd corpusgen
poetry install          # installs core + dev dependencies
poetry run pytest       # run tests
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

## Reproducibility

For reproducible results across machines:

1. **Pin corpusgen version**: `pip install corpusgen==0.1.0`
2. **Pin espeak-ng version**: Record the output of `espeak-ng --version` in your experiment logs
3. **Use `poetry.lock`**: If developing with Poetry, the lock file pins all transitive dependencies
4. **Repository strategy is deterministic**: Given the same sentence bank version and parameters, output is identical

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
