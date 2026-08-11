# Examples

Runnable example scripts are in the [`examples/`](https://github.com/jemsbhai/corpusgen/tree/main/examples) directory.

## Prerequisites

The scripts live in the repository rather than the installed wheel. Clone the
repository, enter it, and install the package:

```bash
git clone https://github.com/jemsbhai/corpusgen.git
cd corpusgen
python -m pip install -e .
```

The per-example prerequisites are:

| Example | eSpeak NG | PHOIBLE data |
|---|---:|---:|
| 01 — inventory exploration | No | Yes |
| 02 — sentence selection | Yes | No |
| 03 — corpus evaluation | Yes | Yes |

For Examples 02 and 03, install eSpeak NG using the
[installation guide](index.md#installation). For Examples 01 and 03, download
the pinned, checksum-verified PHOIBLE snapshot once:

```bash
python -c "from corpusgen.inventory import PhoibleDataset; PhoibleDataset().download()"
```

---

## 01 — Explore a Phoneme Inventory

**File:** [`examples/01_explore_inventory.py`](https://github.com/jemsbhai/corpusgen/blob/main/examples/01_explore_inventory.py)

Load a PHOIBLE phoneme inventory, inspect segments, and query by distinctive features.

```bash
python examples/01_explore_inventory.py
```

**What it demonstrates:**

- Loading inventories with `get_inventory()`
- Listing consonants, vowels, and tones
- Querying segments by single or multiple distinctive features
- Comparing inventory sizes across languages

---

## 02 — Select Sentences for Maximal Coverage

**File:** [`examples/02_select_sentences.py`](https://github.com/jemsbhai/corpusgen/blob/main/examples/02_select_sentences.py)

Use the greedy selection algorithm to pick a minimal subset of sentences that maximizes phoneme coverage.

```bash
python examples/02_select_sentences.py
```

**What it demonstrates:**

- Selecting sentences with `select_sentences()`
- Inspecting `SelectionResult` fields (coverage, indices, timing)
- Comparing phoneme-level vs diphone-level coverage

---

## 03 — Evaluate a Corpus

**File:** [`examples/03_evaluate_corpus.py`](https://github.com/jemsbhai/corpusgen/blob/main/examples/03_evaluate_corpus.py)

Evaluate an existing corpus against a PHOIBLE target inventory and inspect the full report.

```bash
python examples/03_evaluate_corpus.py
```

**What it demonstrates:**

- Evaluating with `evaluate()` using auto-discovered and PHOIBLE inventories
- Reading distribution metrics (entropy, JSD, PCD)
- Reading text quality metrics (sentence length, type-token ratio)
- Per-sentence coverage breakdown
