# Contributing to corpusgen

Thank you for your interest in contributing to **corpusgen**! This document provides guidelines and instructions for contributing to the project. Whether you're reporting a bug, suggesting a feature, improving documentation, or submitting code, your contributions are welcome and appreciated.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Development Setup](#development-setup)
  - [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Improving Documentation](#improving-documentation)
  - [Submitting Code Changes](#submitting-code-changes)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Commit Messages](#commit-messages)
  - [Code Style](#code-style)
  - [Testing](#testing)
  - [Pull Request Process](#pull-request-process)
- [Architecture Overview](#architecture-overview)
- [Optional Dependencies](#optional-dependencies)
- [Releasing](#releasing)
- [Getting Help](#getting-help)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior by opening an issue.

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or later** (tested on 3.10, 3.12, 3.13)
- **[Poetry](https://python-poetry.org/docs/#installation)** for dependency management
- **[espeak-ng](https://github.com/espeak-ng/espeak-ng)** for grapheme-to-phoneme conversion

**Installing espeak-ng:**

| Platform | Command |
|----------|---------|
| **Ubuntu/Debian** | `sudo apt-get update && sudo apt-get install -y espeak-ng` |
| **macOS** | `brew install espeak-ng` |
| **Windows** | Download the `.msi` installer from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases), then set the environment variable: `[Environment]::SetEnvironmentVariable("PHONEMIZER_ESPEAK_LIBRARY", "C:\Program Files\eSpeak NG\libespeak-ng.dll", "User")` |

Verify the installation:

```bash
espeak-ng --version
```

### Development Setup

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/corpusgen.git
   cd corpusgen
   ```

2. **Install dependencies (including dev tools):**

   ```bash
   poetry install --with dev
   ```

   To install all optional dependency groups for full development:

   ```bash
   poetry install --with dev,llm,local,repository,optimization,eval
   ```

3. **Verify your setup by running the test suite:**

   ```bash
   poetry run pytest tests/ -q --no-header -m "not slow"
   ```

   All fast tests should pass. If any fail due to espeak-ng not being found, double-check your installation and environment variables.

4. **Run the linter:**

   ```bash
   poetry run ruff check src/
   ```

### Project Structure

```
corpusgen/
├── src/corpusgen/          # Source code
│   ├── cli/                # Click-based CLI (evaluate, select, generate, inventory)
│   ├── coverage/           # CoverageTracker — phoneme/diphone/triphone tracking
│   ├── evaluate/           # Evaluation metrics (distribution, trajectory, text quality, perplexity, error rates)
│   ├── g2p/                # G2PManager — espeak-ng backend via phonemizer
│   ├── generate/           # Generation framework
│   │   ├── backends/       # Repository, LLM API, and local model backends
│   │   ├── phon_ctg/       # Orchestration loop, scorer, targets, constraints
│   │   ├── phon_datg/      # Inference-time logit steering
│   │   ├── phon_rl/        # PPO-based RL training (reward, trainer, value head, policy)
│   │   └── scorers/        # Phonotactic and fluency scoring
│   ├── inventory/          # PHOIBLE integration, segment/inventory models, espeak mapping
│   ├── select/             # 6 selection algorithms (greedy, CELF, stochastic, ILP, distribution, NSGA-II)
│   └── weights/            # Phoneme weighting strategies
├── tests/                  # Test suite (~50 test files)
├── examples/               # Example scripts
├── paper.md                # JOSS paper
├── paper.bib               # JOSS bibliography
├── pyproject.toml          # Poetry config, tool settings (ruff, pytest, mypy)
├── CITATION.cff            # Citation metadata
├── CHANGELOG.md            # Release history
├── LICENSE                 # Apache 2.0
└── README.md               # User-facing documentation
```

---

## How to Contribute

### Reporting Bugs

If you find a bug, please [open an issue](https://github.com/jemsbhai/corpusgen/issues/new) with:

- **A clear, descriptive title.**
- **Steps to reproduce** the problem, including minimal code if possible.
- **Expected behavior** vs. **actual behavior.**
- **Environment details:** Python version, OS, espeak-ng version (`espeak-ng --version`), corpusgen version (`python -c "import corpusgen; print(corpusgen.__version__)"`).
- **Full traceback** if an exception was raised.

### Suggesting Features

Feature requests are welcome. Please [open an issue](https://github.com/jemsbhai/corpusgen/issues/new) describing:

- **The problem** your feature would solve.
- **Your proposed solution** (if you have one).
- **Alternatives** you've considered.
- **Which module** the feature relates to (evaluation, selection, generation, etc.).

### Improving Documentation

Documentation improvements are highly valued. This includes:

- Fixing typos or unclear explanations in the README, docstrings, or this file.
- Adding usage examples to the `examples/` directory.
- Improving docstrings in the source code.

For small fixes (typos, wording), a direct pull request is fine. For larger documentation restructuring, please open an issue first to discuss the approach.

### Submitting Code Changes

For bug fixes and features:

1. **Open an issue first** (unless the change is trivial) to discuss the approach.
2. **Fork the repo** and create a feature branch.
3. **Write tests** before writing implementation code (see [Testing](#testing)).
4. **Ensure all tests pass** and the linter is clean.
5. **Submit a pull request** with a clear description.

---

## Development Workflow

### Branching Strategy

- `main` is the stable branch. All releases are tagged from `main`.
- Create feature branches from `main` with descriptive names:
  - `feat/add-neural-g2p-backend`
  - `fix/coverage-tracker-diphone-count`
  - `docs/add-quickstart-example`

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

**Types:**

| Type | Use for |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `test` | Adding or updating tests |
| `docs` | Documentation changes |
| `refactor` | Code restructuring without behavior change |
| `ci` | CI/CD changes |
| `chore` | Maintenance tasks (dependency updates, etc.) |

**Examples:**

```
feat(select): add beam search selector with configurable width
fix(g2p): handle empty string input without crashing
test(evaluate): add edge case for single-sentence corpus
docs(readme): add Windows installation troubleshooting
```

### Code Style

We use **[Ruff](https://docs.astral.sh/ruff/)** for linting. The configuration is in `pyproject.toml`:

- **Target:** Python 3.10
- **Line length:** 100 characters
- **Rules:** E, F, I, N, W, UP, B, SIM (with specific exclusions — see `pyproject.toml`)

**Before submitting, always run:**

```bash
poetry run ruff check src/
poetry run ruff check tests/
```

To auto-fix issues:

```bash
poetry run ruff check --fix src/
```

**Additional style guidelines:**

- Use type hints for all public function signatures.
- Write docstrings for all public classes and functions (Google style preferred).
- Use `from __future__ import annotations` sparingly — the project targets Python 3.10+ which supports `X | Y` union syntax natively.
- Keep imports organized: standard library, third-party, then local (Ruff's `I` rules handle this).

### Testing

We use **[pytest](https://docs.pytest.org/)** with a two-tier test strategy:

**Fast tests** (default — run in CI):

```bash
poetry run pytest tests/ -q --no-header -m "not slow"
```

**Slow tests** (require real models, GPU, or network access):

```bash
poetry run pytest tests/ -q --no-header -m "slow"
```

**Full suite:**

```bash
poetry run pytest tests/ -q --no-header
```

**Running a single test file:**

```bash
poetry run pytest tests/test_coverage.py -q --no-header --tb=short
```

**Testing guidelines:**

- **Write tests first (TDD).** Red → Green → Refactor.
- **Every new feature must have tests.** Bug fixes should include a regression test.
- Mark tests that require GPU, large model downloads, or network access with `@pytest.mark.slow`.
- Use hand-computed expected values for mathematical correctness — do not "test" by running the code and copying the output.
- Keep tests deterministic. Pin random seeds where randomness is involved.
- Use the shared fixtures in `tests/conftest.py` (e.g., `sample_english_sentences`, `sample_english_phonemes`).
- Tests that require optional dependencies (e.g., `pulp`, `pymoo`, `torch`) should skip gracefully:

  ```python
  pytest.importorskip("pulp")
  ```

**Test coverage:**

```bash
poetry run pytest tests/ -q --no-header -m "not slow" --cov=corpusgen --cov-report=term-missing
```

### Pull Request Process

1. **Ensure your branch is up to date with `main`:**

   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run the full quality check locally:**

   ```bash
   poetry run ruff check src/ tests/
   poetry run pytest tests/ -q --no-header -m "not slow"
   ```

3. **Open a pull request** against `main` with:
   - A clear title following the conventional commit format.
   - A description of **what** changed and **why**.
   - A link to any related issue (e.g., "Closes #42").
   - Confirmation that tests pass.

4. **CI must pass.** The GitHub Actions workflow runs ruff and pytest across Python 3.10, 3.12, and 3.13.

5. **Respond to review feedback** promptly. We aim for constructive, collaborative reviews.

---

## Architecture Overview

Understanding the module boundaries helps you contribute effectively:

| Module | Responsibility | Key Classes |
|--------|---------------|-------------|
| `g2p` | Text → IPA phoneme conversion | `G2PManager`, `G2PResult` |
| `inventory` | PHOIBLE phoneme inventories & espeak mapping | `PhoibleDataset`, `Inventory`, `Segment`, `EspeakMapping` |
| `coverage` | Phoneme/diphone/triphone frequency tracking | `CoverageTracker` |
| `evaluate` | Corpus quality metrics & reporting | `evaluate()`, `EvaluationReport`, `DistributionMetrics`, `TextQualityMetrics`, `CoverageTrajectory` |
| `select` | Sentence selection algorithms | `GreedySelector`, `CELFSelector`, `StochasticGreedySelector`, `ILPSelector`, `DistributionAwareSelector`, `NSGA2Selector` |
| `weights` | Phoneme weighting strategies | `uniform_weights()`, `frequency_inverse_weights()`, `linguistic_class_weights()` |
| `generate` | Text generation framework | `GenerationLoop`, `PhoneticScorer`, `PhoneticTargetInventory` |
| `generate.backends` | Pluggable generation engines | `RepositoryBackend`, `LLMBackend`, `LocalBackend` |
| `generate.phon_datg` | Inference-time logit steering | `DATGStrategy`, `AttributeWordIndex`, `LogitModulator` |
| `generate.phon_rl` | RL-based fine-tuning | `PhonRLTrainer`, `PhoneticReward`, `ValueHead` |
| `cli` | Command-line interface | `corpusgen evaluate`, `select`, `generate`, `inventory` |

**Key design principles:**

- **Modularity:** Each module has a clear boundary and can be used independently.
- **Optional dependencies:** Heavy dependencies (torch, transformers, pulp, pymoo) are optional. Core functionality works with only the base dependencies.
- **Graceful degradation:** Modules that depend on optional packages must handle `ImportError` gracefully and provide informative error messages.

---

## Optional Dependencies

corpusgen uses pip extras to keep the core package lightweight. When working on a specific module, install the relevant group:

| Extra | Modules it enables | Install command |
|-------|-------------------|-----------------|
| `llm` | LLM API backend (litellm, openai, anthropic) | `poetry install --with llm` |
| `local` | Local model backend (torch, transformers, bitsandbytes, peft) | `poetry install --with local` |
| `repository` | HuggingFace dataset backend | `poetry install --with repository` |
| `optimization` | ILP and NSGA-II selectors (pulp, pymoo) | `poetry install --with optimization` |
| `eval` | Distribution metrics (scipy, matplotlib) | `poetry install --with eval` |
| `full` | Everything | `poetry install --with full` |

When adding a new optional dependency:

1. Add it to `[tool.poetry.dependencies]` as `optional = true`.
2. Add it to the appropriate extras group in `[tool.poetry.extras]`.
3. Create a corresponding `[tool.poetry.group.<name>.dependencies]` section.
4. Guard the import in the source code with a try/except block.
5. Add a `pytest.importorskip()` call at the top of affected test files.

---

## Releasing

Releases are managed by the maintainers. The process is:

1. Update `version` in `pyproject.toml` and `src/corpusgen/__init__.py`.
2. Update `CHANGELOG.md` with the new version and changes.
3. Update `CITATION.cff` with the new version and date.
4. Commit, tag, and push: `git tag v0.x.y && git push origin main --tags`.
5. Build and publish: `poetry build && poetry publish`.

---

## Getting Help

- **Questions about contributing?** Open a [discussion](https://github.com/jemsbhai/corpusgen/issues) or issue.
- **Stuck on a bug?** Include the full traceback, your environment details, and what you've already tried.
- **Not sure where to start?** Look for issues labeled `good first issue` or `help wanted`.

Thank you for contributing to corpusgen!
