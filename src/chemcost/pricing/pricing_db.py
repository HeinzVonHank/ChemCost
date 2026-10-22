"""SQLite-backed frozen pricing database for the benchmark."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from pathlib import Path

from .price_entry import PriceEntry

# Override with the CHEMCOST_PRICING_DB environment variable. Without it: data/ of a source
# checkout, else data/ under the current directory (non-editable installs).
_CHECKOUT_DB = Path(__file__).parents[3] / "data" / "pricing_db.sqlite"
DEFAULT_DB_PATH = Path(
    os.environ.get(
        "CHEMCOST_PRICING_DB",
        _CHECKOUT_DB if _CHECKOUT_DB.exists() else Path.cwd() / "data" / "pricing_db.sqlite",
    )
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chemicals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    cas TEXT,
    smiles TEXT,
    molecular_weight REAL,
    UNIQUE(cas)
);

CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chemical_id INTEGER NOT NULL,
    supplier TEXT NOT NULL,
    catalog_number TEXT,
    quantity_g REAL NOT NULL,
    price_usd REAL NOT NULL,
    price_per_gram_usd REAL NOT NULL,
    purity TEXT,
    grade TEXT,
    url TEXT,
    snapshot_date TEXT NOT NULL,
    FOREIGN KEY (chemical_id) REFERENCES chemicals(id)
);

CREATE TABLE IF NOT EXISTS normalized_prices (
    chemical_id INTEGER PRIMARY KEY,
    price_per_gram_usd REAL NOT NULL,
    method TEXT NOT NULL,
    n_sources INTEGER NOT NULL,
    sources_json TEXT NOT NULL,
    FOREIGN KEY (chemical_id) REFERENCES chemicals(id)
);

CREATE INDEX IF NOT EXISTS idx_chemicals_cas ON chemicals(cas);
CREATE INDEX IF NOT EXISTS idx_chemicals_name ON chemicals(name);
CREATE INDEX IF NOT EXISTS idx_prices_chemical_id ON prices(chemical_id);

CREATE TABLE IF NOT EXISTS chemical_names (
    chemical_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    FOREIGN KEY (chemical_id) REFERENCES chemicals(id)
);

CREATE INDEX IF NOT EXISTS idx_chemical_names_name ON chemical_names(name);
"""


_EXCLUSION_KEYWORDS = ("isotope", "labeled", "labelled", "reference standard", "certified")


def _purity_percent(purity: str | None) -> float | None:
    """Parse a purity string to percent; None if missing, unknown, or excluded material."""
    if purity is None:
        return None
    text = str(purity).strip().lower()
    if any(kw in text for kw in _EXCLUSION_KEYWORDS):
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = float(match.group())
    if value <= 0:
        return None  # "0.0" marks an unreported purity
    return value * 100 if value <= 1 else value


# Reagents whose ordinary commercial grade is below 95%: the grade's own minimum applies instead.
COMMERCIAL_GRADES = {
    "[K+].[OH-]": 85.0,                # potassium hydroxide, 85-90% (balance water)
    "[H-].[Na+]": 55.0,                # sodium hydride, 60% dispersion in mineral oil
    "[NaH]": 55.0,                     # (sodium hydride drawn as a covalent hydride)
    "O=C(OO)c1cccc(Cl)c1": 70.0,       # mCPBA, <= 77% (balance 3-chlorobenzoic acid and water)
}
_GRADE_KEYS: dict[str, float] = {}


def grade_min_purity(smiles: str | None, min_purity: float) -> float:
    """Minimum purity for a chemical: min_purity, or its lower commercial grade (see COMMERCIAL_GRADES)."""
    if not _GRADE_KEYS:
        _GRADE_KEYS.update({compound_key(k): v for k, v in COMMERCIAL_GRADES.items()})
    grade = _GRADE_KEYS.get(compound_key(smiles)) if smiles else None
    return min(min_purity, grade) if grade is not None else min_purity


_STANDARDIZE = {}
_METALS = {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", "Al", "Zn", "Cu", "Ag", "Au", "Ni",
           "Pd", "Pt", "Fe", "Co", "Mn", "Cr", "Ti", "Zr", "Sn", "Rh", "Ir", "Ru", "Os", "Cd", "Hg", "In",
           "Ce", "Sc", "Y", "La", "Bi", "Pb", "Tl", "Ga", "V", "Mo", "W"}


def _inorganic_formula(smiles: str | None) -> str | None:
    """Molecular formula without charges for a neutral, carbon-free compound; None otherwise."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or Chem.GetFormalCharge(mol) != 0 or any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
        return None
    return CalcMolFormula(mol).replace("+", "").replace("-", "")


def compound_key(smiles: str) -> str | None:
    """Stereo-free key that treats tautomers, charge forms and metal bonding of one compound alike."""
    if smiles in _STANDARDIZE:
        return _STANDARDIZE[smiles]
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    key = None
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is not None:
        try:
            mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
            # Ionic and covalent drawings of a metal salt or organometallic should agree:
            # neutralize isolated metal atoms and drop radicals, then uncharge the rest.
            mol = Chem.RWMol(mol)
            for atom in mol.GetAtoms():
                if atom.GetDegree() == 0 and atom.GetSymbol() in _METALS:
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(0)
                atom.SetNumRadicalElectrons(0)
                atom.SetNoImplicit(False)
            Chem.SanitizeMol(mol)
            mol = rdMolStandardize.Uncharger().uncharge(mol.GetMol())
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
            key = Chem.MolToSmiles(mol, isomericSmiles=False)
        except Exception:  # standardization is best effort; fall back to plain flat SMILES
            key = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
    _STANDARDIZE[smiles] = key
    return key


def _canonical_smiles(smiles: str, isomeric: bool = True) -> str | None:
    """RDKit canonical SMILES (isomeric by default), or None if unparseable."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric) if mol is not None else None


class PricingDB:
    """SQLite database for frozen chemical pricing data."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, read_only: bool = False) -> None:
        """Open the pricing database; read_only=True never writes and requires the file to exist."""
        self.db_path = Path(db_path)
        if (read_only or self.db_path == DEFAULT_DB_PATH) and not self.db_path.exists():
            raise FileNotFoundError(
                f"Frozen pricing database not found at {self.db_path}. "
                "Run `python3 scripts/download_data.py` before evaluation."
            )
        if read_only:
            self.conn = sqlite3.connect(
                self.db_path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False
            )
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if not read_only:
            self.conn.executescript(SCHEMA)
        self._canon_ids: dict[str, int] | None = None
        self._flat_ids: dict[str, list[int]] | None = None
        self._compound_ids: dict[str, list[int]] | None = None
        self._formula_ids: dict[str, list[int]] | None = None

    def add_chemical(
        self,
        name: str,
        cas: str | None = None,
        smiles: str | None = None,
        mw: float | None = None,
    ) -> int:
        """Add or update a chemical and return its ID."""
        if cas:
            row = self.conn.execute("SELECT id FROM chemicals WHERE cas = ?", (cas,)).fetchone()
            if row:
                return row["id"]

        cursor = self.conn.execute(
            "INSERT OR IGNORE INTO chemicals (name, cas, smiles, molecular_weight) VALUES (?, ?, ?, ?)",
            (name, cas, smiles, mw),
        )
        self.conn.commit()
        self._canon_ids = None
        self._flat_ids = None
        self._compound_ids = None
        self._formula_ids = None

        if cursor.lastrowid:
            return cursor.lastrowid

        row = self.conn.execute(
            "SELECT id FROM chemicals WHERE name = ? AND cas IS ?", (name, cas)
        ).fetchone()
        return row["id"] if row else -1

    def add_price(self, chemical_id: int, entry: PriceEntry) -> None:
        """Add a price entry for a chemical."""
        self.conn.execute(
            """INSERT INTO prices
            (chemical_id, supplier, catalog_number, quantity_g, price_usd,
             price_per_gram_usd, purity, grade, url, snapshot_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chemical_id,
                entry.supplier,
                entry.catalog_number,
                entry.quantity_g,
                entry.price_usd,
                entry.price_per_gram_usd,
                entry.purity,
                entry.grade,
                entry.url,
                entry.snapshot_date.isoformat(),
            ),
        )
        self.conn.commit()

    def set_normalized_price(
        self,
        chemical_id: int,
        price_per_gram: float,
        method: str,
        n_sources: int,
        sources: list[str],
    ) -> None:
        """Set the normalized $/g price for a chemical."""
        self.conn.execute(
            """INSERT OR REPLACE INTO normalized_prices
            (chemical_id, price_per_gram_usd, method, n_sources, sources_json)
            VALUES (?, ?, ?, ?, ?)""",
            (chemical_id, price_per_gram, method, n_sources, json.dumps(sources)),
        )
        self.conn.commit()

    def get_price(
        self,
        cas: str | None = None,
        name: str | None = None,
        smiles: str | None = None,
    ) -> float | None:
        """Get the normalized price per gram for a chemical."""
        if cas:
            row = self.conn.execute(
                """SELECT np.price_per_gram_usd FROM normalized_prices np
                JOIN chemicals c ON c.id = np.chemical_id
                WHERE c.cas = ?""",
                (cas,),
            ).fetchone()
        elif smiles:
            row = self.conn.execute(
                """SELECT np.price_per_gram_usd FROM normalized_prices np
                JOIN chemicals c ON c.id = np.chemical_id
                WHERE c.smiles = ?""",
                (smiles,),
            ).fetchone()
        elif name:
            row = self.conn.execute(
                """SELECT np.price_per_gram_usd FROM normalized_prices np
                JOIN chemicals c ON c.id = np.chemical_id
                WHERE LOWER(c.name) = LOWER(?)""",
                (name,),
            ).fetchone()
        else:
            return None

        return row["price_per_gram_usd"] if row else None

    def get_all_chemicals(self) -> list[dict]:
        """List all chemicals in the database."""
        rows = self.conn.execute(
            """SELECT c.*, np.price_per_gram_usd as normalized_price
            FROM chemicals c
            LEFT JOIN normalized_prices np ON c.id = np.chemical_id
            ORDER BY c.name"""
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Get database statistics."""
        n_chemicals = self.conn.execute("SELECT COUNT(*) FROM chemicals").fetchone()[0]
        n_prices = self.conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        n_normalized = self.conn.execute("SELECT COUNT(*) FROM normalized_prices").fetchone()[0]
        return {
            "n_chemicals": n_chemicals,
            "n_price_entries": n_prices,
            "n_normalized": n_normalized,
        }

    # ── Pack-level procurement queries ─────────────────────────────────────

    def _lookup_chemical_id(
        self,
        smiles: str | None = None,
        cas: str | None = None,
        name: str | None = None,
    ) -> int | None:
        """Return the chemical id for the first identifier that matches.

        Order: CAS, exact SMILES, canonical SMILES (same molecule written differently),
        then name (primary name, then synonyms). Structure always wins over name.
        """
        if cas:
            row = self.conn.execute("SELECT id FROM chemicals WHERE cas = ?", (cas,)).fetchone()
            if row:
                return row["id"]
        if smiles:
            row = self.conn.execute(
                "SELECT id FROM chemicals WHERE smiles = ? ORDER BY id", (smiles,)
            ).fetchone()
            if row:
                return row["id"]
            key = _canonical_smiles(smiles)
            if key and key in self._canonical_index():
                return self._canonical_index()[key]
            # Same constitution with different or missing stereo labels, if unambiguous.
            flat = _canonical_smiles(smiles, isomeric=False)
            if flat and len(self._flat_index().get(flat, ())) == 1:
                return self._flat_index()[flat][0]
            # Same compound written as another tautomer, charge form, or with metal bonds.
            ckey = compound_key(smiles)
            if ckey and len(self._compound_index().get(ckey, ())) == 1:
                return self._compound_index()[ckey][0]
            # Carbon-free compound drawn ionic vs covalent ([Li+].[AlH4-] vs [Li+].[Al+3].4[H-]).
            fkey = _inorganic_formula(smiles)
            if fkey and len(self._formula_index().get(fkey, ())) == 1:
                return self._formula_index()[fkey][0]
        if name:
            row = self.conn.execute(
                "SELECT id FROM chemicals WHERE LOWER(name) = LOWER(?) ORDER BY id", (name,)
            ).fetchone()
            if row:
                return row["id"]
            row = self.conn.execute(
                "SELECT chemical_id AS id FROM chemical_names WHERE LOWER(name) = LOWER(?)"
                " ORDER BY chemical_id",
                (name,),
            ).fetchone()
            if row:
                return row["id"]
        return None

    def _compound_index(self) -> dict[str, list[int]]:
        """Map compound_key -> chemical ids, built once per connection."""
        if self._compound_ids is None:
            index: dict[str, list[int]] = {}
            for row in self.conn.execute(
                "SELECT id, smiles FROM chemicals WHERE smiles IS NOT NULL ORDER BY id"
            ):
                key = compound_key(row["smiles"])
                if key:
                    index.setdefault(key, []).append(row["id"])
            self._compound_ids = index
        return self._compound_ids

    def _formula_index(self) -> dict[str, list[int]]:
        """Map charge-free molecular formula of carbon-free chemicals -> ids, built once per connection."""
        if self._formula_ids is None:
            index: dict[str, list[int]] = {}
            for row in self.conn.execute("SELECT id, smiles FROM chemicals WHERE smiles IS NOT NULL ORDER BY id"):
                key = _inorganic_formula(row["smiles"])
                if key:
                    index.setdefault(key, []).append(row["id"])
            self._formula_ids = index
        return self._formula_ids

    def _flat_index(self) -> dict[str, list[int]]:
        """Map stereo-free canonical SMILES -> chemical ids, built once per connection."""
        if self._flat_ids is None:
            index: dict[str, list[int]] = {}
            for row in self.conn.execute(
                "SELECT id, smiles FROM chemicals WHERE smiles IS NOT NULL ORDER BY id"
            ):
                key = _canonical_smiles(row["smiles"], isomeric=False)
                if key:
                    index.setdefault(key, []).append(row["id"])
            self._flat_ids = index
        return self._flat_ids

    def _canonical_index(self) -> dict[str, int]:
        """Map canonical SMILES -> lowest chemical id, built once per connection."""
        if self._canon_ids is None:
            index: dict[str, int] = {}
            for row in self.conn.execute(
                "SELECT id, smiles FROM chemicals WHERE smiles IS NOT NULL ORDER BY id"
            ):
                key = _canonical_smiles(row["smiles"])
                if key:
                    index.setdefault(key, row["id"])
            self._canon_ids = index
        return self._canon_ids

    def get_pack_quotes(
        self,
        smiles: str | None = None,
        cas: str | None = None,
        name: str | None = None,
        min_purity: float = 95.0,
    ) -> list[dict]:
        """Return qualifying pack-level quotes sorted by quantity_g ascending.

        A quote qualifies if its purity is known and >= min_purity (fractions such as
        "0.98" are read as 98%) and it is not a reference standard or isotope-labeled
        material; reagents in COMMERCIAL_GRADES use their commercial grade's minimum instead.
        For each pack size only the cheapest quote is kept (ties: supplier name).
        """
        chemical_id = self._lookup_chemical_id(smiles=smiles, cas=cas, name=name)
        if chemical_id is None:
            return []
        stored = self.conn.execute("SELECT smiles FROM chemicals WHERE id = ?", (chemical_id,)).fetchone()
        min_purity = grade_min_purity(stored[0] if stored else smiles, min_purity)

        rows = self.conn.execute(
            """SELECT quantity_g, price_usd, price_per_gram_usd, purity, supplier
               FROM prices
               WHERE chemical_id = ?
               ORDER BY quantity_g ASC, price_usd ASC, supplier ASC""",
            (chemical_id,),
        ).fetchall()

        best: dict[float, dict] = {}
        for row in rows:
            purity = _purity_percent(row["purity"])
            if purity is None or purity < min_purity:
                continue
            best.setdefault(row["quantity_g"], dict(row))  # rows are price-ordered per size
        return sorted(best.values(), key=lambda e: e["quantity_g"])

    def get_raw_quotes(self, smiles: str | None = None, name: str | None = None) -> list[dict]:
        """Every stored quote for a chemical (all suppliers, purities and pack sizes), unfiltered."""
        chemical_id = self._lookup_chemical_id(smiles=smiles, name=name)
        if chemical_id is None:
            return []
        rows = self.conn.execute(
            "SELECT supplier, quantity_g, price_usd, purity FROM prices WHERE chemical_id = ? "
            "ORDER BY supplier, quantity_g, price_usd",
            (chemical_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_procurement_price(
        self,
        required_mass_g: float,
        smiles: str | None = None,
        cas: str | None = None,
        name: str | None = None,
        min_purity: float = 95.0,
    ) -> dict:
        """Return the cheapest purchase of whole packs, in any combination of pack sizes, that gives at
        least required_mass_g (see cheapest_purchase). If no qualifying quote exists, the chemical is
        unpriced.

        Returns a dict with:
            total_cost_usd   – dollars spent
            cost_per_g       – total_cost_usd / quantity_g
            quantity_g       – total grams purchased
            supplier         – the packs bought, e.g. "2 x 5 g (Supplier A) + 1 x 1 g (Supplier B)"
            n_packs          – number of packs purchased
            packs            – [(quantity_g, count, supplier, price_usd), ...]
            tier             – "pack_based" | "unpriced"
        """
        quotes = self.get_pack_quotes(smiles=smiles, cas=cas, name=name, min_purity=min_purity)
        if not quotes:
            return {"total_cost_usd": None, "cost_per_g": None, "quantity_g": None, "supplier": None,
                    "n_packs": None, "packs": None, "tier": "unpriced"}
        total, qty, packs = cheapest_purchase(required_mass_g, quotes)
        return {
            "total_cost_usd": total,
            "cost_per_g": round(total / qty, 6),
            "quantity_g": qty,
            "supplier": " + ".join(f"{n} x {q:g} g ({sup})" for q, n, sup, _ in packs),
            "n_packs": sum(n for _, n, _, _ in packs),
            "packs": packs,
            "tier": "pack_based",
        }

    def close(self) -> None:
        self.conn.close()


def cheapest_purchase(required_mass_g: float, quotes: list[dict]) -> tuple[float, float, list]:
    """Cheapest multiset of whole packs (any sizes, possibly from different suppliers) whose total mass
    is at least required_mass_g.

    quotes: one entry per pack size (quantity_g, price_usd, supplier), e.g. from get_pack_quotes.
    Returns (total_usd, total_quantity_g, [(quantity_g, count, supplier, price_usd), ...]).
    Ties: smaller total quantity, then fewer packs. Exact: depth-first search over pack counts from
    the largest size down, pruned with the lowest price per gram among the sizes still available.
    """
    packs = sorted(((q["quantity_g"], q["price_usd"], q["supplier"]) for q in quotes), key=lambda x: -x[0])
    best_rate = [min(p / g for g, p, _ in packs[i:]) for i in range(len(packs))]
    best: list = [None]

    def key(cost, qty, combo):
        return (round(cost, 6), round(qty, 9), sum(n for _, n, _, _ in combo))

    def dfs(i: int, rem: float, cost: float, qty: float, combo: list) -> None:
        if rem <= 1e-12:
            cand = (cost, qty, list(combo))
            if best[0] is None or key(*cand) < key(*best[0]):
                best[0] = cand
            return
        if i == len(packs):
            return
        if best[0] is not None and round(cost + rem * best_rate[i], 6) > round(best[0][0], 6):
            return
        g, p, sup = packs[i]
        for k in range(math.ceil(rem / g - 1e-12), -1, -1):
            if k:
                combo.append((g, k, sup, p))
            dfs(i + 1, rem - k * g, cost + k * p, qty + k * g, combo)
            if k:
                combo.pop()

    dfs(0, required_mass_g, 0.0, 0.0, [])
    total, qty, combo = best[0]
    return round(total, 6), round(qty, 9), combo
