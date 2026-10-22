"""Tools available to LLM agents during the benchmark evaluation.

These tools wrap real APIs/databases and are the agent's interface
for looking up chemical information and performing calculations.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import operator
import re
import threading
import time
from functools import lru_cache
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_pricing_db_local = threading.local()
_http_client = None

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def _get_pricing_db():
    if not hasattr(_pricing_db_local, "db"):
        from ..pricing.pricing_db import PricingDB
        _pricing_db_local.db = PricingDB(read_only=True)
    return _pricing_db_local.db


def _get_http_client():
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=30)
    return _http_client


_search_cache: dict[str, str] = {}
_search_cache_lock = threading.Lock()


def _search_chemical_cached(query: str) -> str:
    """PubChem lookup returning a JSON string; caches results and definitive misses only."""
    with _search_cache_lock:
        if query in _search_cache:
            return _search_cache[query]
    result = _search_chemical_uncached(query)
    if '"error"' not in result or "not found in PubChem" in result:
        with _search_cache_lock:
            _search_cache[query] = result
    return result


# PubChem allows 5 requests/s per client; parallel evaluation workers share one budget.
_PUBCHEM_MIN_INTERVAL_S = 0.25
_pubchem_lock = threading.Lock()
_pubchem_last = [0.0]


def _pubchem_get(client, url: str, retries: int = 5):
    """GET with a process-wide rate limit; throttling (429/503) is waited out, never handed to the agent."""
    resp = None
    for _ in range(retries):
        with _pubchem_lock:
            wait = _pubchem_last[0] + _PUBCHEM_MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _pubchem_last[0] = time.monotonic()
        resp = client.get(url)
        if resp.status_code not in (429, 503):
            return resp
        time.sleep(min(float(resp.headers.get("retry-after") or 5), 30.0))
    return resp


def _search_chemical_uncached(query: str) -> str:
    client = _get_http_client()
    try:
        resp = _pubchem_get(
            client,
            f"{PUBCHEM_BASE}/compound/name/{quote(query, safe='')}/property/"
            "SMILES,IUPACName/JSON",
        )
        if resp.status_code in (429, 503):
            return json.dumps({"error": "PubChem is temporarily unavailable (rate limited); retry this search"})
        if resp.status_code == 404:
            return json.dumps({"error": f"Chemical '{query}' not found in PubChem"})
        resp.raise_for_status()
        props = resp.json()["PropertyTable"]["Properties"][0]
        return json.dumps({
            "name": props.get("IUPACName", query),
            # PubChem renamed SMILES properties in 2025: "SMILES" is the isomeric form.
            "smiles": props.get("SMILES")
            or props.get("IsomericSMILES")
            or props.get("CanonicalSMILES")
            or props.get("ConnectivitySMILES"),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_chemical(query: str) -> dict:
    """Search for a chemical by name or CAS number.

    Returns the SMILES and IUPAC name from PubChem (no molecular weight or formula: an agent works
    those out from the structure).

    Args:
        query: Chemical name, CAS number, or common abbreviation.

    Returns:
        Dict with keys: name, smiles. On failure, returns dict with 'error' key.
    """
    query = str(query).strip().lower()
    if not query or len(query) > 500:
        return {"error": "Query must be a non-empty name or CAS number of at most 500 characters"}
    return json.loads(_search_chemical_cached(query))


def _pack_text(grams: float) -> str:
    """A pack size as a supplier lists it: "250 mg", "5 g", "2.5 kg"."""
    if grams < 1:
        return f"{grams * 1000:.6g} mg"
    if grams >= 1000:
        return f"{grams / 1000:.6g} kg"
    return f"{grams:.6g} g"


@lru_cache(maxsize=1024)
def _get_supplier_quotes_cached(smiles_or_name: str) -> str:
    """Cached supplier quotes lookup; returns JSON string."""
    return json.dumps(_get_supplier_quotes_impl(smiles_or_name))


def _get_supplier_quotes_impl(smiles_or_name: str) -> dict:
    db = _get_pricing_db()

    import re
    looks_like_smiles = bool(re.search(r"[=#@\[\]\\/()\+]", smiles_or_name)) or \
                        smiles_or_name.startswith(("C", "N", "O", "S", "F", "Cl", "Br", "I", "c", "n"))

    def _lookup(smiles=None, name=None):
        quotes = db.get_raw_quotes(smiles=smiles, name=name)
        if not quotes:
            return None
        return {"quotes": [{"supplier": q["supplier"], "pack": _pack_text(q["quantity_g"]),
                            "price_usd": q["price_usd"], "purity": q["purity"]} for q in quotes]}

    if looks_like_smiles:
        result = _lookup(smiles=smiles_or_name) or _lookup(name=smiles_or_name)
    else:
        result = _lookup(name=smiles_or_name) or _lookup(smiles=smiles_or_name)

    if result is None:
        return {"error": f"No pricing data found for '{smiles_or_name}' in the database"}
    return result


def get_supplier_quotes(smiles_or_name: str) -> dict:
    """Every supplier quote for a chemical in the frozen pricing database, as listed.

    Quotes are not filtered or cleaned: all suppliers, pack sizes (with their units) and purities
    as the suppliers reported them (a missing purity is null). Deciding which offers qualify and
    which purchase is cheapest is up to the caller.

    Args:
        smiles_or_name: SMILES string (preferred) or chemical name.

    Returns:
        {"quotes": [{"supplier": str, "pack": str, "price_usd": float, "purity": str | None}, ...]}
        or {"error": str}
    """
    query = str(smiles_or_name).strip()
    if not query or len(query) > 500:
        return {"error": "Query must be a non-empty name or SMILES of at most 500 characters"}
    return json.loads(_get_supplier_quotes_cached(query))


_FORMULA = re.compile(r"^(?:[A-Z][a-z]?\d*)+[+-]?\d*$")


def _formula_weight(formula: str) -> float | None:
    """Average molecular weight of a formula like "C10H12N2O" (a trailing charge is ignored)."""
    if not _FORMULA.match(formula or ""):
        return None
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    total = 0.0
    for symbol, count in re.findall(r"([A-Z][a-z]?)(\d*)", re.sub(r"[+-]\d*$", "", formula)):
        try:
            total += table.GetAtomicWeight(symbol) * (int(count) if count else 1)
        except RuntimeError:
            return None  # not an element symbol
    return total if total > 0 else None


def compute_molar_mass(smiles: str) -> dict:
    """Compute the average molecular weight from a SMILES string or a molecular formula.

    Args:
        smiles: A valid SMILES string, or a molecular formula such as "C10H12N2O".

    Returns:
        Dict with keys: smiles (or formula), molecular_weight, formula.
        On failure, returns dict with 'error' key.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdMolDescriptors

        RDLogger.DisableLog("rdApp.*")  # a formula is tried as SMILES first
        text = str(smiles).strip()
        mol = Chem.MolFromSmiles(text) if text else None
        if mol is None:
            mw = _formula_weight(text)
            if mw is None:
                return {"error": f"Not a valid SMILES or molecular formula: {smiles}"}
            return {"formula": text, "molecular_weight": round(mw, 4)}
        mw = Descriptors.MolWt(mol)  # average MW, as reported by PubChem
        formula = rdMolDescriptors.CalcMolFormula(mol)
        return {
            "smiles": smiles,
            "molecular_weight": round(mw, 4),
            "formula": formula,
        }
    except ImportError:
        return {"error": "RDKit is not installed"}
    except Exception as e:
        return {"error": str(e)}


_CALC_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
}
_CALC_CONSTS = {"pi": math.pi, "e": math.e}
_CALC_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_CALC_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CALC_MAX_LEN = 500
_CALC_MAX_ABS = 1e300
_CALC_MAX_EXPONENT = 100


def _calc_eval(node: ast.AST) -> float:
    """Evaluate a whitelisted arithmetic AST node; anything else raises ValueError."""
    if isinstance(node, ast.Expression):
        return _calc_eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = float(node.value)
    elif isinstance(node, ast.Name) and node.id in _CALC_CONSTS:
        value = _CALC_CONSTS[node.id]
    elif isinstance(node, ast.BinOp) and type(node.op) in _CALC_BINOPS:
        left, right = _calc_eval(node.left), _calc_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _CALC_MAX_EXPONENT:
            raise ValueError("exponent too large")
        value = _CALC_BINOPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_UNARYOPS:
        value = _CALC_UNARYOPS[type(node.op)](_calc_eval(node.operand))
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _CALC_FUNCS
        and not node.keywords
    ):
        args = [_calc_eval(a) for a in node.args]
        if node.func.id == "round" and len(args) == 2:
            args[1] = int(args[1])  # ndigits must be an integer
        value = _CALC_FUNCS[node.func.id](*args)
    else:
        raise ValueError(f"unsupported syntax: {type(node).__name__}")
    if isinstance(value, complex) or not math.isfinite(value) or abs(value) > _CALC_MAX_ABS:
        raise ValueError("result is not a finite real number")
    return float(value)


def calculate(expression: str) -> dict:
    """Evaluate an arithmetic expression without executing code.

    Supports numbers, +, -, *, /, ** (exponent up to 100), parentheses, the constants
    pi and e, and the functions sqrt, log, log10, exp, abs, round. Attribute access,
    names, and all other syntax are rejected.

    Args:
        expression: A mathematical expression string.

    Returns:
        Dict with keys: expression, result.
        On failure, returns dict with 'error' key.
    """
    try:
        if len(expression) > _CALC_MAX_LEN:
            raise ValueError("expression too long")
        result = _calc_eval(ast.parse(expression, mode="eval"))
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"Cannot evaluate '{expression}': {e}"}


# Tool registry for agent frameworks. There is no name-search tool: every visible name is a systematic
# name the agent must turn into a structure itself (search_chemical stays available for ablations).
TOOL_REGISTRY = {
    "get_supplier_quotes": {
        "function": get_supplier_quotes,
        "description": (
            "Get every supplier quote for a chemical from the frozen pricing database, as the suppliers "
            "listed them: supplier, pack size with its unit, price in USD, and reported purity (null when "
            "not reported). Quotes are not filtered: decide yourself which ones qualify. "
            "If no pack quotes are returned, the chemical is unpriced."
        ),
        "parameters": {
            "smiles_or_name": "SMILES string (preferred) or chemical name.",
        },
    },
    "calculate": {
        "function": calculate,
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, log.",
        "parameters": {"expression": "A mathematical expression string."},
    },
}
