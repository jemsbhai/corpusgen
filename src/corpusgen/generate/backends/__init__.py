"""Generation backends: pluggable engines for the Phon-CTG loop."""

from corpusgen.generate.backends.local import LocalBackend
from corpusgen.generate.backends.llm_api import LLMBackend
from corpusgen.generate.backends.repository import RepositoryBackend

__all__ = ["LocalBackend", "LLMBackend", "RepositoryBackend"]
