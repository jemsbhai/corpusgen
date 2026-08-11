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

### PhoibleDataset

::: corpusgen.inventory.phoible.PhoibleDataset

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

### compute_distribution_metrics

::: corpusgen.evaluate.distribution.compute_distribution_metrics

### CoverageTrajectory

::: corpusgen.evaluate.trajectory.CoverageTrajectory

### compute_coverage_trajectory

::: corpusgen.evaluate.trajectory.compute_coverage_trajectory

### ErrorRateResult

::: corpusgen.evaluate.error_rates.ErrorRateResult

### compute_error_rates

::: corpusgen.evaluate.error_rates.compute_error_rates

### CorpusPerplexityMetrics

::: corpusgen.evaluate.perplexity.CorpusPerplexityMetrics

### compute_corpus_perplexity

::: corpusgen.evaluate.perplexity.compute_corpus_perplexity

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

### NgramPhonotacticScorer

::: corpusgen.generate.scorers.phonotactic.NgramPhonotacticScorer

### PerplexityFluencyScorer

::: corpusgen.generate.scorers.fluency.PerplexityFluencyScorer

### ReadabilityScorer

::: corpusgen.generate.scorers.readability.ReadabilityScorer

`ReadabilityScorer` is available as a Python scorer or candidate-filter hook;
the CLI does not currently expose a readability flag.

### PhoneticReward

::: corpusgen.generate.phon_rl.reward.PhoneticReward

### TrainingConfig

::: corpusgen.generate.phon_rl.trainer.TrainingConfig

### TrainingResult

::: corpusgen.generate.phon_rl.trainer.TrainingResult

### PhonRLTrainer

::: corpusgen.generate.phon_rl.trainer.PhonRLTrainer

### AttributeWordIndex

::: corpusgen.generate.phon_datg.attribute_words.AttributeWordIndex

### DATGStrategy

::: corpusgen.generate.phon_datg.graph.DATGStrategy

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
