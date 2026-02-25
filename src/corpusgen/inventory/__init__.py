"""Phoneme inventory management: PHOIBLE integration and custom inventories."""

from corpusgen.inventory.models import Segment, Inventory, FEATURE_NAMES
from corpusgen.inventory.phoible import PhoibleDataset

__all__ = ["Segment", "Inventory", "FEATURE_NAMES", "PhoibleDataset"]
