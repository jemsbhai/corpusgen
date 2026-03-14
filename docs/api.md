# API Reference

This page documents the public Python API for `corpusgen`.

---

## Top-Level Functions

These are the primary entry points, importable directly from `corpusgen`.

### evaluate

::: corpusgen.evaluate.evaluate.evaluate

### select_sentences

::: corpusgen.select.select_sentences

### get_inventory

::: corpusgen.get_inventory

---

## Data Models

### Inventory

::: corpusgen.inventory.models.Inventory

### Segment

::: corpusgen.inventory.models.Segment

---

## Evaluation Results

### EvaluationReport

::: corpusgen.evaluate.report.EvaluationReport

### DistributionMetrics

::: corpusgen.evaluate.distribution.DistributionMetrics

### TextQualityMetrics

::: corpusgen.evaluate.text_quality.TextQualityMetrics

### Verbosity

::: corpusgen.evaluate.report.Verbosity

---

## Selection Results

### SelectionResult

::: corpusgen.select.result.SelectionResult

---

## Generation

### GenerationLoop

::: corpusgen.generate.phon_ctg.loop.GenerationLoop

### StoppingCriteria

::: corpusgen.generate.phon_ctg.loop.StoppingCriteria

### PhoneticTargetInventory

::: corpusgen.generate.phon_ctg.targets.PhoneticTargetInventory

### PhoneticScorer

::: corpusgen.generate.phon_ctg.scorer.PhoneticScorer

---

## Backends

### RepositoryBackend

::: corpusgen.generate.backends.repository.RepositoryBackend

### LLMBackend

::: corpusgen.generate.backends.llm_api.LLMBackend

### LocalBackend

::: corpusgen.generate.backends.local.LocalBackend

---

## G2P

### G2PManager

::: corpusgen.g2p.manager.G2PManager

### G2PResult

::: corpusgen.g2p.result.G2PResult

---

## Coverage Tracking

### CoverageTracker

::: corpusgen.coverage.tracker.CoverageTracker
