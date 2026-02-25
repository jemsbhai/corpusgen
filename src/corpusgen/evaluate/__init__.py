"""Corpus evaluation: analyze text against phoneme coverage targets."""

from corpusgen.evaluate.evaluate import evaluate
from corpusgen.evaluate.report import EvaluationReport, SentenceDetail, Verbosity

__all__ = ["evaluate", "EvaluationReport", "SentenceDetail", "Verbosity"]
