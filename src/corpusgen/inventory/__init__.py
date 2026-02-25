"""Phoneme inventory management: PHOIBLE integration and custom inventories."""

from corpusgen.inventory.mapping import EspeakMapping
from corpusgen.inventory.models import Segment, Inventory, FEATURE_NAMES
from corpusgen.inventory.phoible import PhoibleDataset

__all__ = [
    "EspeakMapping",
    "FEATURE_NAMES",
    "Inventory",
    "PhoibleDataset",
    "Segment",
]
