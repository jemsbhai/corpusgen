# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Phase 1 — G2P Pipeline
- `G2PResult` dataclass: phonemes, diphones, triphones, unique_phonemes
- `G2PManager`: espeak-ng backend via phonemizer, batch processing, multi-dialect variant support, `supported_languages()`
- Helpful error messages when espeak-ng is not installed or `PHONEMIZER_ESPEAK_LIBRARY` is not set
- Smoke tests and G2P tests

#### Phase 2 — Evaluation & Inventory
- `CoverageTracker`: phoneme/diphone/triphone coverage tracking with frequency counts and sentence-level provenance
- `EvaluationReport`: three verbosity levels (minimal/normal/verbose), `to_dict()`, `to_json()`, `to_jsonld_ex()` export
- `SentenceDetail`: per-sentence coverage breakdown with new_phonemes attribution
- `evaluate()`: top-level API wiring G2P + CoverageTracker + EvaluationReport
  - Accepts `target_phonemes=None` (derive from corpus), explicit list, or `"phoible"` shortcut
  - Supports `unit="phoneme"`, `"diphone"`, or `"triphone"`
- `Segment` dataclass: 38 PHOIBLE distinctive features, allophones, marginal flag, segment class, glyph ID
- `Inventory` dataclass: computed properties for consonants, vowels, tones, marginal/non-marginal filtering, feature queries, summary statistics, `to_dict()` serialization
- `PhoibleDataset`: PHOIBLE CSV loader with download, caching (~24 MB at `~/.corpusgen/phoible.csv`), and query API
  - Lookup by ISO 639-3 or Glottocode, with optional source filtering
  - Default selection: largest inventory per language
  - `get_union_inventory()`: merge all inventories into maximally inclusive set
  - `get_inventory_for_espeak()`: espeak voice code → PHOIBLE inventory
  - Language search (case-insensitive partial match), `available_languages()`
  - Auto-load on first query (lazy loading)
- `EspeakMapping`: bidirectional espeak ↔ ISO 639-3 mapping
  - 131 espeak voice codes mapped to ISO 639-3
  - Macrolanguage resolution (e.g., `ms` → `zsm`, `sw` → `swh`)
  - Dialect grouping (e.g., `en-us`/`en-gb` → `eng`)
  - Bundled JSON at `corpusgen/data/espeak_iso_mapping.json`
- Top-level convenience API: `corpusgen.evaluate()`, `corpusgen.get_inventory()`

#### Testing
- 57 tests for `evaluate()`: return types, coverage invariants, derived inventory, sentence details, phoneme counts/sources, diphone/triphone units, edge cases, report usability, cross-component consistency, determinism
- 628 multilingual robustness tests across 40 languages and 12 language families
- 75 tests for `Segment` and `Inventory` data models
- 58 tests for `PhoibleDataset` loader with fixture CSV
- 38 tests for `EspeakMapping`
- 209 integration tests with real PHOIBLE data across 32 languages
- 31 wiring tests for convenience API

### Infrastructure
- Project scaffold: `src/corpusgen/` layout with Poetry 2.3.2
- Apache 2.0 license
- GitHub repo: `jemsbhai/corpusgen`
- espeak-ng as core G2P backend (phonemizer wrapper)
- PHOIBLE 2.0 for phoneme inventories (3,020 inventories, 2,186 languages)
