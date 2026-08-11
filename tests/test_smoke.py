"""Smoke tests: verify corpusgen installs and imports correctly."""

from unittest.mock import patch

import pytest

import corpusgen


def test_version_exists():
    """Package should expose a version string."""
    assert hasattr(corpusgen, "__version__")
    assert isinstance(corpusgen.__version__, str)
    assert corpusgen.__version__ == "0.1.6"


def test_subpackages_importable():
    """All subpackages should be importable."""
    from corpusgen import (
        cli,  # noqa: F401
        coverage,  # noqa: F401
        evaluate,  # noqa: F401
        g2p,  # noqa: F401
        generate,  # noqa: F401
        inventory,  # noqa: F401
        select,  # noqa: F401
        weights,  # noqa: F401
    )


def test_get_inventory_preserves_mapped_source_error():
    """A valid espeak mapping must not hide an invalid PHOIBLE source."""
    error = KeyError("No inventory with source 'missing' for 'eng'")
    with (
        patch(
            "corpusgen.inventory.mapping.EspeakMapping.to_iso",
            return_value="eng",
        ),
        patch(
            "corpusgen.inventory.phoible.PhoibleDataset.get_inventory",
            side_effect=error,
        ) as get_inventory,
        pytest.raises(KeyError, match="source 'missing'"),
    ):
        corpusgen.get_inventory("en-us", source="missing")

    get_inventory.assert_called_once_with("eng", source="missing")
