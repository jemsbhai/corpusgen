"""Smoke tests: verify corpusgen installs and imports correctly."""

import corpusgen


def test_version_exists():
    """Package should expose a version string."""
    assert hasattr(corpusgen, "__version__")
    assert isinstance(corpusgen.__version__, str)
    assert corpusgen.__version__ == "0.1.0"


def test_subpackages_importable():
    """All subpackages should be importable."""
    from corpusgen import g2p  # noqa: F401
    from corpusgen import inventory  # noqa: F401
    from corpusgen import coverage  # noqa: F401
    from corpusgen import evaluate  # noqa: F401
    from corpusgen import select  # noqa: F401
    from corpusgen import generate  # noqa: F401
    from corpusgen import weights  # noqa: F401
    from corpusgen import cli  # noqa: F401
