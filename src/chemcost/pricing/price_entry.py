"""Record type for one supplier quote in the frozen pricing database."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class PriceEntry:
    """A single price point from a supplier."""

    chemical_name: str
    cas: str | None
    supplier: str
    catalog_number: str | None = None
    quantity_g: float = 0.0  # quantity in grams
    price_usd: float = 0.0
    price_per_gram_usd: float = 0.0
    purity: str | None = None  # e.g., ">=99%"
    grade: str | None = None  # e.g., "reagent grade", "ACS grade"
    url: str | None = None
    snapshot_date: date = field(default_factory=date.today)
