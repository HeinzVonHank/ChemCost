"""Noise injection pipeline for benchmark chemical names and quantities.

Transforms clean, unambiguous chemical component names into realistic
variations that a human chemist might encounter: dropped isomer prefixes,
abbreviated or expanded names, stripped stereochemistry, and formula-based
references.

Stage 2 adds quantity noise: switching between equivalents and mol%,
converting to volume strings, rounding/approximating values, and replacing
precise quantities with vague natural-language descriptions.

Stage 4 adds format transform noise: converting structured component data
into natural-language experimental prose, simulating OCR errors on chemical
names and numbers, and mixing text/table formats.

All functions operate on benchmark record dicts (from JSONL) and return
modified *copies* — originals are never mutated.
"""

from __future__ import annotations

import copy
import random
import re
from typing import Literal

from .chemical_aliases import (
    ABBREVIATION_TO_FULL,
    AMBIGUOUS_ABBREVIATIONS,
    COMMON_TO_IUPAC,
    FORMULA_TO_NAME,
    FULL_TO_ABBREVIATION,
    ISOMER_AMBIGUOUS,
    IUPAC_TO_COMMON,
    POSITIONAL_PREFIXES,
    SALT_VARIATIONS,
    STEREO_PREFIXES,
)

NoiseLevel = Literal["low", "medium", "high", "rich"]

# Fraction of eligible components to modify at each noise level.
_NOISE_PROBABILITY: dict[NoiseLevel, float] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.80,
    "rich": 0.50,
}


def _should_apply(noise_level: NoiseLevel, rng: random.Random) -> bool:
    """Return True with probability determined by *noise_level*."""
    return rng.random() < _NOISE_PROBABILITY[noise_level]


# ── Isomer noise ─────────────────────────────────────────────────────────────

def _strip_stereo(name: str) -> str | None:
    """Strip stereochemistry prefixes like (R)-, (S)-, D-, L- etc.

    Returns the modified name, or None if no prefix was found.
    """
    for pattern in STEREO_PREFIXES:
        new_name = re.sub(r"^" + pattern, "", name)
        if new_name != name:
            return new_name.strip()
    return None


def _strip_positional(name: str) -> str | None:
    """Strip positional isomer prefixes like n-, sec-, tert- etc.

    Returns the modified name, or None if no prefix was found.
    """
    lower = name.lower()
    for prefix in POSITIONAL_PREFIXES:
        if lower.startswith(prefix):
            stripped = name[len(prefix):]
            if stripped:
                return stripped
    return None


def inject_isomer_noise(
    record: dict,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
) -> dict:
    """Replace specific chemical names with ambiguous isomer-parent names.

    Operates on the ``components`` list of a benchmark record.  For each
    component whose name matches a known isomer mapping, the specific name
    is replaced with the generic parent (e.g. "n-butanol" -> "butanol").
    At higher noise levels, stereochemistry and positional prefixes are also
    stripped even when there is no explicit mapping.

    Parameters
    ----------
    record : dict
        A benchmark reaction record with a ``components`` key.
    noise_level : "low" | "medium" | "high"
        Controls the probability that each eligible name is modified.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A deep copy of the record with modified component names.  An
        ``original_name`` field is added to each modified component.
    """
    rng = random.Random(seed)
    record = copy.deepcopy(record)

    for comp in record.get("components", []):
        name = comp.get("name", "").strip()
        if not name:
            continue

        if not _should_apply(noise_level, rng):
            continue

        # 1. Check explicit isomer mapping (case-sensitive then case-insensitive)
        replacement = ISOMER_AMBIGUOUS.get(name)
        if replacement is None:
            replacement = ISOMER_AMBIGUOUS.get(name.lower())
        if replacement is not None:
            comp["original_name"] = name
            comp["name"] = replacement
            comp["noise_type"] = "isomer_ambiguity"
            continue

        # 2. Strip stereochemistry prefixes
        stripped = _strip_stereo(name)
        if stripped is not None and stripped != name:
            comp["original_name"] = name
            comp["name"] = stripped
            comp["noise_type"] = "stereo_stripped"
            continue

        # 3. At medium+, also strip positional prefixes even without mapping
        if noise_level in ("medium", "high", "rich"):
            stripped = _strip_positional(name)
            if stripped is not None:
                comp["original_name"] = name
                comp["name"] = stripped
                comp["noise_type"] = "positional_stripped"

    return record


# ── Name variation noise ─────────────────────────────────────────────────────

def inject_name_variation(
    record: dict,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
) -> dict:
    """Replace chemical names with abbreviations, IUPAC names, or formulas.

    For abbreviations the replacement goes in the *harder* direction:
    - Abbreviation -> full IUPAC name (agent must parse systematic names)
    - Full name -> ambiguous abbreviation (agent must disambiguate)
    - Common name -> IUPAC name (and vice versa)
    - Named salts -> molecular formulas (and vice versa)

    Parameters
    ----------
    record : dict
        A benchmark reaction record.
    noise_level : "low" | "medium" | "high"
        Controls the probability and aggressiveness of replacement.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A deep copy with modified component names.
    """
    rng = random.Random(seed)
    record = copy.deepcopy(record)

    for comp in record.get("components", []):
        name = comp.get("name", "").strip()
        if not name:
            continue

        if not _should_apply(noise_level, rng):
            continue

        replacement = _find_name_variation(name, noise_level, rng)
        common = (comp.get("name_common") or "").strip().lower()
        if replacement is not None and common and replacement.strip().lower() == common:
            continue  # noise must not hand back the familiar name the visible name replaced
        if replacement is not None and replacement != name:
            comp["original_name"] = name
            comp["name"] = replacement
            comp["noise_type"] = "name_variation"

    return record


def _find_name_variation(
    name: str, noise_level: NoiseLevel, rng: random.Random
) -> str | None:
    """Find a name variation for a single chemical name.

    Tries multiple strategies in order and returns the first match.
    """
    # Strategy 1: abbreviation -> full name (expand abbreviations)
    if name in ABBREVIATION_TO_FULL:
        fulls = ABBREVIATION_TO_FULL[name]
        return rng.choice(fulls)

    # Strategy 2: full name -> abbreviation
    name_lower = name.lower()
    if name_lower in FULL_TO_ABBREVIATION:
        abbrevs = FULL_TO_ABBREVIATION[name_lower]
        # At medium+, prefer ambiguous abbreviations for extra difficulty
        if noise_level in ("medium", "high", "rich"):
            for abbrev in abbrevs:
                if abbrev in AMBIGUOUS_ABBREVIATIONS:
                    return abbrev
        return rng.choice(abbrevs)

    # Strategy 3: common name <-> IUPAC
    if name_lower in COMMON_TO_IUPAC:
        return COMMON_TO_IUPAC[name_lower]
    if name_lower in IUPAC_TO_COMMON:
        return IUPAC_TO_COMMON[name_lower]

    # Strategy 4: named salt <-> formula
    if name_lower in SALT_VARIATIONS:
        return SALT_VARIATIONS[name_lower]
    if name_lower in FORMULA_TO_NAME:
        return FORMULA_TO_NAME[name_lower]

    # Strategy 5: at high noise, also swap case conventions
    if noise_level == "high":
        # e.g. "Ethanol" -> "ethanol" or "ETHANOL"
        if name[0].isupper() and len(name) > 3 and not name.isupper():
            return name.lower()

    return None


# ── Quantity noise (Stage 2) ─────────────────────────────────────────────────

# Approximate densities (g/mL) for common solvents / liquid reagents.
_SOLVENT_DENSITIES: dict[str, float] = {
    "water": 1.00,
    "thf": 0.889,
    "tetrahydrofuran": 0.889,
    "dcm": 1.33,
    "dichloromethane": 1.33,
    "methylene chloride": 1.33,
    "toluene": 0.87,
    "methylbenzene": 0.87,
    "dmf": 0.944,
    "dimethylformamide": 0.944,
    "dmso": 1.10,
    "dimethyl sulfoxide": 1.10,
    "acetonitrile": 0.786,
    "mecn": 0.786,
    "methanol": 0.791,
    "meoh": 0.791,
    "ethanol": 0.789,
    "etoh": 0.789,
    "ethyl acetate": 0.902,
    "etoac": 0.902,
    "diethyl ether": 0.713,
    "ether": 0.713,
    "acetone": 0.784,
    "chloroform": 1.489,
    "hexane": 0.659,
    "n-hexane": 0.659,
    "pentane": 0.626,
    "benzene": 0.879,
    "acetic acid": 1.049,
    "pyridine": 0.982,
    "triethylamine": 0.726,
    "et3n": 0.726,
    "1,4-dioxane": 1.034,
    "dioxane": 1.034,
    "isopropanol": 0.786,
    "2-propanol": 0.786,
    "carbon tetrachloride": 1.594,
    "1,2-dichloroethane": 1.253,
    "nitromethane": 1.137,
}

# Default density for reagents when no specific density is known.
_DEFAULT_DENSITY = 1.0

# Per-noise-level probabilities for vague-quantity descriptions.
_VAGUE_PROB: dict[NoiseLevel, float] = {
    "low": 0.0,
    "medium": 0.0,
    "high": 1.0,
    "rich": 0.20,
}


def _get_density(name: str) -> float:
    """Look up approximate density for a chemical by name."""
    return _SOLVENT_DENSITIES.get(name.lower().strip(), _DEFAULT_DENSITY)


def _has_known_density(name: str) -> bool:
    """Return whether we have an explicit density for this component."""
    return name.lower().strip() in _SOLVENT_DENSITIES


def _amount_to_volume_str(amount_g: float, name: str) -> str:
    """A liquid's laboratory amount as a volume."""
    volume_ml = amount_g / _get_density(name)
    if volume_ml < 0.001:
        return f"{volume_ml * 1000:.1f} \u00b5L"
    if volume_ml < 1.0:
        return f"{volume_ml:.3f} mL"
    return f"{volume_ml:.2f} mL"


def _amount_to_mmol_str(amount_g: float, mw: float) -> str:
    """A laboratory amount in mmol."""
    return f"{amount_g / mw * 1000:.3g} mmol"


def _equiv_to_mol_percent_str(equiv: float) -> str:
    """Convert a sub-1.0 equivalents value to a mol% string."""
    mol_pct = equiv * 100
    # Use clean formatting: drop trailing zeros
    if mol_pct == int(mol_pct):
        return f"{int(mol_pct)} mol%"
    return f"{mol_pct:.3g} mol%"


def _make_approximate_str(amount_g: float, rng: random.Random) -> str:
    """A laboratory amount rounded to two significant figures, marked as approximate."""
    prefix = rng.choice(["~", "approx. ", "about ", "ca. "])
    value = float(f"{amount_g:.2g}")
    return f"{prefix}{value:g} g" if value >= 1 else f"{prefix}{value * 1000:g} mg"


def _make_vague_description(
    equiv: float, role: str, rng: random.Random,
) -> str | None:
    """Return a vague natural-language quantity, or None if N/A."""
    if equiv > 5.0:
        return rng.choice(["a large excess", "large excess"])
    if equiv > 2.0:
        return rng.choice(["excess", "in excess"])
    if equiv < 0.1:
        return rng.choice(["catalytic amount", "cat."])
    if equiv < 0.3 and role in ("solvent", "reagent", "reactant"):
        return "a few drops"
    return None


def inject_quantity_noise(
    record: dict,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
) -> dict:
    """Inject noise into component quantity/equivalents fields.

    Transforms precise numeric equivalents into realistic imprecise
    representations: mol% strings, volume descriptions, approximate
    values, and vague natural-language quantities.

    Parameters
    ----------
    record : dict
        A benchmark reaction record with a ``components`` key.
    noise_level : "low" | "medium" | "high"
        Controls which transformations are active and how often:
        - ``"low"``: mol%/equiv mixing for catalysts only (~25%).
        - ``"medium"``: + approximate values + unit switching for known liquids.
        - ``"high"``: + vague quantities (~80%).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A deep copy of the record with modified quantity fields.
        Modified components gain a ``quantity_description`` string
        field and have ``equivalents`` set to ``None``.  An
        ``original_equivalents`` field records the original numeric
        value.  ``noise_type`` is ``"quantity_noise"`` and
        ``quantity_noise_kind`` indicates the specific sub-type.
    """
    rng = random.Random(seed)
    record = copy.deepcopy(record)
    for comp in record.get("components", []):
        equiv, amount = comp.get("equivalents"), comp.get("amount_g")
        if equiv is None or amount is None or comp.get("role") == "solvent":
            continue
        if not _should_apply(noise_level, rng):
            continue
        mw = comp.get("mw")
        name = comp.get("name_common") or comp.get("name", "")
        role = comp.get("role", "")

        def describe(text: str, kind: str, comp=comp, amount=amount) -> None:
            comp["original_amount_g"] = amount
            comp["amount_g"] = None
            comp["quantity_description"] = text
            comp["noise_type"] = "quantity_noise"
            comp["quantity_noise_kind"] = kind

        # HIGH/RICH: vague quantities for extreme amounts
        if _VAGUE_PROB[noise_level] > 0 and rng.random() < _VAGUE_PROB[noise_level]:
            vague = _make_vague_description(equiv, role, rng)
            if vague is not None:
                describe(vague, "vague")
                continue
        # LOW+: catalysts are often given as mol% of the scale reference
        if role == "catalyst" and equiv < 1.0:
            describe(_equiv_to_mol_percent_str(equiv), "mol_percent")
            continue
        if noise_level in ("medium", "high", "rich") and role != "catalyst":
            # MEDIUM+: liquids by volume, other amounts in mmol or approximately
            if _has_known_density(name) and rng.random() < 0.5:
                describe(_amount_to_volume_str(amount, name), "unit_switch")
                continue
            if mw and rng.random() < 0.3:
                describe(_amount_to_mmol_str(amount, mw), "mmol")
                continue
            if rng.random() < 0.6:
                describe(_make_approximate_str(amount, rng), "approximate")
                continue
    return record


# ── Missing information noise (Stage 3) ─────────────────────────────────────

# Per-noise-level probabilities for missing-info sub-operations.
_MW_DROP_PROB: dict[NoiseLevel, float] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.80,
    "rich": 0.50,
}

_YIELD_DROP_PROB: dict[NoiseLevel, float] = {
    "low": 0.0,
    "medium": 0.0,
    "high": 0.60,
    "rich": 0.40,
}

_ROLE_DROP_PROB: dict[NoiseLevel, float] = {
    "low": 0.0,
    "medium": 0.0,
    "high": 0.60,
    "rich": 0.40,
}

_PRODUCT_DROP_PROB: dict[NoiseLevel, float] = {
    "low": 0.0,
    "medium": 0.0,
    "high": 0.0,
    "rich": 0.0,
}


def inject_missing_info_noise(
    record: dict,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
) -> dict:
    """Randomly drop information fields from a benchmark record.

    Simulates realistic scenarios where reaction descriptions are
    incomplete: missing molecular weights, unknown yields, unlabelled
    component roles, or vague product references.

    Parameters
    ----------
    record : dict
        A benchmark reaction record with ``components``,
        ``yield_percent``, ``product_smiles``, etc.
    noise_level : "low" | "medium" | "high"
        Controls which fields may be dropped and with what probability.

        - **low**: Drop MW from ~25 % of components.
        - **medium**: Drop MW (~50 %).
        - **high**: Drop MW (~80 %), yield (~60 %), role (~60 %).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A deep copy of the record with selected fields set to ``None``
        or generic placeholders.  A ``noise_applied`` list on the record
        and per-component ``noise_type`` tags track what was removed.
    """
    rng = random.Random(seed)
    record = copy.deepcopy(record)

    # Ensure a list exists to track record-level noise.
    if "noise_applied" not in record:
        record["noise_applied"] = []

    # ── Drop MW from components ─────────────────────────────────────
    mw_prob = _MW_DROP_PROB[noise_level]
    if mw_prob > 0:
        for comp in record.get("components", []):
            if comp.get("mw") is not None and rng.random() < mw_prob:
                comp["original_mw"] = comp["mw"]
                comp["mw"] = None
                comp["noise_type"] = "mw_dropped"

    # ── Drop yield (medium+) ────────────────────────────────────────
    yield_prob = _YIELD_DROP_PROB[noise_level]
    if yield_prob > 0 and record.get("steps"):
        # Multi-step: each step yield can go missing; the route yield then goes too.
        for st in record["steps"]:
            if st.get("yield_percent") is not None and rng.random() < yield_prob:
                st["original_yield_percent"] = st["yield_percent"]
                st["yield_percent"] = None
                if record.get("yield_percent") is not None:
                    record["original_yield_percent"] = record["yield_percent"]
                    record["yield_percent"] = None
                record["noise_applied"].append(f"yield_dropped_step_{st.get('step_number')}")
    elif yield_prob > 0 and record.get("yield_percent") is not None:
        if rng.random() < yield_prob:
            record["original_yield_percent"] = record["yield_percent"]
            record["yield_percent"] = None
            record["noise_applied"].append("yield_dropped")

    # ── Drop role labels (medium+) ──────────────────────────────────
    role_prob = _ROLE_DROP_PROB[noise_level]
    if role_prob > 0:
        for comp in record.get("components", []):
            role = comp.get("role", "")
            if role and role != "unknown" and rng.random() < role_prob:
                comp["original_role"] = role
                comp["role"] = "unknown"
                # Combine tag if MW was already dropped for this component.
                if comp.get("noise_type") == "mw_dropped":
                    comp["noise_type"] = "mw_and_role_dropped"
                else:
                    comp["noise_type"] = "role_dropped"

    # ── Drop product info (high only) ───────────────────────────────
    product_prob = _PRODUCT_DROP_PROB[noise_level]
    if product_prob > 0 and rng.random() < product_prob:
        # Replace product component names with a generic placeholder.
        for comp in record.get("components", []):
            if comp.get("role") == "product" or comp.get(
                "original_role"
            ) == "product":
                cur_name = comp.get("name")
                if cur_name and cur_name != "the desired product":
                    comp["original_name"] = cur_name
                    comp["name"] = "the desired product"
        # Remove product SMILES at record level.
        if record.get("product_smiles") is not None:
            record["original_product_smiles"] = record[
                "product_smiles"
            ]
            record["product_smiles"] = None
        record["noise_applied"].append("product_dropped")

    return record


# ── Format transform noise (Stage 4) ────────────────────────────────────

# Ground-truth fields that must never be modified by any noise function.
_GROUND_TRUTH_FIELDS = frozenset({
    "procurement_cost_usd_per_g_product",
    "total_cost_per_gram_product_usd",
    "cost_tier",
})

# OCR character-confusion maps.  Each key may be replaced by its value.
_OCR_CHAR_SWAPS: list[tuple[str, str]] = [
    ("O", "0"),   # capital O -> zero
    ("0", "O"),   # zero -> capital O
    ("l", "1"),   # lowercase L -> one
    ("1", "l"),   # one -> lowercase L
]

# OCR substring replacements (applied to full strings, not per-char).
_OCR_SUBSTR_SWAPS: list[tuple[str, str]] = [
    ("rn", "m"),
]

# Special-character OCR degradation.
_OCR_SPECIAL_CHAR: list[tuple[str, str]] = [
    ("\u2265", ">="),   # >= -> >=
    ("\u00b0C", "oC"),  # degree-C -> oC
    ("\u00b0", "o"),    # degree -> o
    ("\u03bc", "u"),    # mu -> u
    ("\u2082", "2"),    # subscript 2
    ("\u2083", "3"),    # subscript 3
    ("\u2084", "4"),    # subscript 4
    ("\u2080", "0"),    # subscript 0
    ("\u2081", "1"),    # subscript 1
]

# Procedural sentence templates for NL conversion.
def _amount_text(comp: dict) -> str | None:
    """Laboratory amount as written in a text ("1.234 g", "56.7 mg"), or a noisy description."""
    if comp.get("quantity_description"):
        return str(comp["quantity_description"])
    grams = comp.get("amount_g")
    if grams is None or comp.get("role") == "solvent":
        return None
    return f"{grams:.5g} g" if grams >= 1 else f"{grams * 1000:.5g} mg"


def _component_nl_phrase(comp: dict, rng: random.Random) -> str:
    """Build a natural-language phrase for one component: name and laboratory amount."""
    name = comp.get("name", "UNKNOWN")
    amount = _amount_text(comp)
    return f"{name} ({amount})" if amount else name


def _build_natural_language(
    record: dict, rng: random.Random,
) -> str:
    """Convert a record's components into experimental-section prose: names and laboratory amounts
    in a shuffled order, the yield, and the product formula (no reaction class or roles)."""
    components = [
        component
        for component in record.get("components", [])
        if component.get("role") != "product" and component.get("original_role") != "product"
    ]
    if not components:
        return "No components specified."
    phrases = [_component_nl_phrase(c, rng) for c in components]
    rng.shuffle(phrases)
    opener = rng.choice(["A mixture of", "The reaction combined", "To the vessel were added", "The procedure used"])
    body = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
    sentences = [f"{opener} {body}."]
    sentences.append(_product_sentence(record, rng))
    if record.get("scale_reference"):
        sentences.append(f"The scale reference is {record['scale_reference']}.")
    return " ".join(s for s in sentences if s)


def _product_sentence(record: dict, rng: random.Random) -> str:
    """The isolated product: its name (or formula) and mass; the mass is withheld when the yield is."""
    product = record.get("product") if isinstance(record.get("product"), dict) else {}
    label = product.get("display_name") or (f"a compound of formula {product['formula']}" if product.get("formula")
                                            else "the product")
    grams = product.get("mass_g")
    if grams is None or record.get("yield_percent") is None:
        return f"The product is {label}; the isolated amount was not reported."
    mass = f"{grams:.5g} g" if grams >= 1 else f"{grams * 1000:.5g} mg"
    return rng.choice([f"This gave {mass} of {label}.", f"Workup afforded {label} ({mass}).",
                       f"{label[0].upper() + label[1:]} was isolated ({mass})."])


def _apply_ocr_noise(
    text: str, rng: random.Random, prob: float = 0.20,
) -> str:
    """Apply simulated OCR errors to *text*.

    Each eligible character/substring is corrupted with probability
    *prob*.  Deterministic given the same *rng* state.
    """
    # 1. Substring replacements first (e.g. "rn" -> "m")
    for old, new in _OCR_SUBSTR_SWAPS:
        if old in text and rng.random() < prob:
            text = text.replace(old, new, 1)

    # 2. Special-character degradation (always applied)
    for old, new in _OCR_SPECIAL_CHAR:
        text = text.replace(old, new)

    # 3. Per-character confusion
    chars = list(text)
    for i, ch in enumerate(chars):
        for old, new in _OCR_CHAR_SWAPS:
            if ch == old and rng.random() < prob:
                chars[i] = new
                break  # only one swap per character
    return "".join(chars)


def _apply_ocr_to_components(
    record: dict, rng: random.Random, prob: float = 0.20,
) -> dict:
    """Apply OCR noise to component names (record already deep-copied)."""
    for comp in record.get("components", []):
        name = comp.get("name", "")
        if not name:
            continue
        noisy_name = _apply_ocr_noise(name, rng, prob)
        if noisy_name != name:
            if "original_name" not in comp:
                comp["original_name"] = name
            comp["name"] = noisy_name
            existing = comp.get("noise_type", "")
            if existing:
                comp["noise_type"] = existing + "+ocr"
            else:
                comp["noise_type"] = "ocr"
    return record


def _build_mixed_format(
    record: dict, rng: random.Random,
) -> str:
    """Build a mixed text+table description.

    Some components are described in prose, others in a markdown-style
    table, mimicking real papers where part of the info is in the
    experimental text and part in a supplementary table.
    """
    components = [
        component
        for component in record.get("components", [])
        if component.get("role") != "product" and component.get("original_role") != "product"
    ]
    if not components:
        return "No components specified."

    # Randomly split: some in text, some in table
    shuffled = list(range(len(components)))
    rng.shuffle(shuffled)
    split_point = max(1, len(shuffled) // 2)
    text_indices = set(shuffled[:split_point])
    table_indices = set(shuffled[split_point:])

    # Build text portion
    text_parts = [_component_nl_phrase(components[i], rng) for i in sorted(text_indices)]
    text_section = "The reaction used " + ", ".join(text_parts) + "."

    # Build table portion
    table_lines: list[str] = [
        "Component | Amount",
        "--- | ---",
    ]
    for i in sorted(table_indices):
        c = components[i]
        table_lines.append(f"{c.get('name', '?')} | {_amount_text(c) or 'not recorded'}")
    table_section = "\n".join(table_lines)

    return f"{text_section} {_product_sentence(record, rng)}\n\n{table_section}"


def _per_step_text(record: dict, rng: random.Random, builder) -> str:
    """Render a multi-step route step by step without revealing intermediate identities."""
    parts: list[str] = []
    steps = record.get("steps") or []
    for i, st in enumerate(steps):
        n = st.get("step_number")
        rows = [
            {**c, "name": f"product of step {c['from_step']}"} if c.get("from_step") is not None else c
            for c in record.get("components", [])
            if c.get("original_step") == n
        ]
        text = builder({"components": rows}, rng)
        product = st.get("product") or {}
        what = "the final product" if i == len(steps) - 1 else f"the product of step {n}"
        label = product.get("display_name") or (f"formula {product['formula']}" if product.get("formula") else None)
        tail = f" This gives {what}" + (f" ({label})" if label else "")
        if st.get("yield_percent") is not None and product.get("mass_g") is not None:
            g = product["mass_g"]
            tail += f": {g:.5g} g isolated" if g >= 1 else f": {g * 1000:.5g} mg isolated"
        if st.get("scale_reference"):
            tail += f"; scale reference {st['scale_reference']}"
        parts.append(f"Step {n}: {text}{tail}.")
    return "\n".join(parts)


def inject_format_noise(
    record: dict,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
    format_kind: Literal["nl_only", "ocr_only", "nl_plus_ocr"] | None = None,
) -> dict:
    """Transform the record format to simulate real-world input variation.

    Noise level behavior:

    - ``low``: no format change (structured data stays structured).
    - ``medium``: convert the structured ``components`` list into a
      natural-language experimental-section paragraph stored in a
      new ``description`` field.
    - ``high``: natural-language conversion **plus** simulated OCR
      errors on component names and description text, and a mixed
      text+table representation in ``mixed_format``.

    Ground-truth fields (``procurement_cost_usd_per_g_product``,
    ``total_cost_per_gram_product_usd``, ``cost_tier``) are never
    modified.

    Parameters
    ----------
    record : dict
        A benchmark reaction record with a ``components`` key.
    noise_level : "low" | "medium" | "high"
        Controls format transform intensity.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A deep copy of the record.  At medium+, a ``description``
        field is added containing the prose representation.  At high,
        OCR noise is also applied to component names and the
        description.
    """
    rng = random.Random(seed)
    record = copy.deepcopy(record)

    if noise_level == "low":
        # No format change at low noise
        return record

    # Resolve which format_kind applies. Explicit override wins; else level-based.
    if format_kind is None:
        if noise_level == "medium":
            format_kind = "nl_only"
        elif noise_level == "high":
            format_kind = "nl_plus_ocr"
        elif noise_level == "rich":
            format_kind = "nl_plus_ocr"
        else:
            format_kind = "nl_only"

    record["format_kind"] = format_kind

    if format_kind == "ocr_only":
        # Keep structured format; only corrupt component names with OCR noise.
        ocr_seed = rng.randint(0, 2**31)
        ocr_rng = random.Random(ocr_seed)
        record = _apply_ocr_to_components(record, ocr_rng)
        record["format_kind"] = "ocr_only"
        return record

    # nl_only / nl_plus_ocr: build natural-language description.
    multistep = bool(record.get("steps"))
    description = (
        _per_step_text(record, rng, _build_natural_language)
        if multistep else _build_natural_language(record, rng)
    )

    if format_kind == "nl_plus_ocr":
        ocr_seed = rng.randint(0, 2**31)
        ocr_rng = random.Random(ocr_seed)
        record = _apply_ocr_to_components(record, ocr_rng)
        description = _apply_ocr_noise(description, ocr_rng)

        # Build mixed-format variant
        mix_seed = rng.randint(0, 2**31)
        mix_rng = random.Random(mix_seed)
        mixed_section = (
            _per_step_text(record, mix_rng, _build_mixed_format)
            if multistep else _build_mixed_format(record, mix_rng)
        )
        record["mixed_format"] = mixed_section
        # The agent reads the mixed prose + table rendering (built from OCR-corrupted names).
        description = mixed_section

    record["description"] = description
    record["format_kind"] = format_kind
    return record


# ── Combined noise entry point ──────────────────────────────────────────

NoiseType = Literal[
    "isomer", "name_variation", "quantity", "missing_info", "format",
]

_DEFAULT_NOISE_TYPES: list[NoiseType] = ["isomer", "name_variation"]
DEFAULT_NOISE_TYPES_BY_LEVEL: dict[NoiseLevel, list[NoiseType]] = {
    "low": ["isomer", "name_variation"],
    "medium": ["isomer", "name_variation", "quantity", "missing_info", "format"],
    "high": ["isomer", "name_variation", "quantity", "missing_info", "format"],
    "rich": ["isomer", "name_variation", "quantity", "missing_info", "format"],
}


def default_noise_types_for_level(noise_level: NoiseLevel) -> list[NoiseType]:
    """Return the default enabled noise stages for a named noise level."""
    return list(DEFAULT_NOISE_TYPES_BY_LEVEL.get(noise_level, _DEFAULT_NOISE_TYPES))


def inject_noise(
    record: dict,
    noise_types: list[NoiseType] | None = None,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
    format_kind: Literal["nl_only", "ocr_only", "nl_plus_ocr"] | None = None,
) -> dict:
    """Apply one or more noise types to a benchmark record.

    This is the main entry point.  Each noise type is applied
    sequentially so they can compound (e.g. an isomer-ambiguous name
    may then also get its abbreviation expanded).

    Parameters
    ----------
    record : dict
        A benchmark reaction record.
    noise_types : list of NoiseType, optional
        Which noise types to apply.  Defaults to
        ``["isomer", "name_variation"]``.
    noise_level : "low" | "medium" | "high"
        Controls per-component probability and aggressiveness.
    seed : int or None
        Master seed; sub-seeds are derived for each noise type.

    Returns
    -------
    dict
        A deep copy of the record with noise applied.

    Examples
    --------
    >>> record = {"components": [{"name": "n-butanol", "role": "solvent"}]}
    >>> noisy = inject_noise(record, noise_level="high", seed=42)
    >>> noisy["components"][0]["name"]
    'butanol'
    """
    if noise_types is None:
        noise_types = list(_DEFAULT_NOISE_TYPES)

    master_rng = random.Random(seed)

    result = record
    for ntype in noise_types:
        sub_seed = master_rng.randint(0, 2**31)
        if ntype == "isomer":
            result = inject_isomer_noise(
                result, noise_level=noise_level, seed=sub_seed,
            )
        elif ntype == "name_variation":
            result = inject_name_variation(
                result, noise_level=noise_level, seed=sub_seed,
            )
        elif ntype == "missing_info":
            result = inject_missing_info_noise(
                result, noise_level=noise_level, seed=sub_seed,
            )
        elif ntype == "quantity":
            result = inject_quantity_noise(
                result, noise_level=noise_level, seed=sub_seed,
            )
        elif ntype == "format":
            result = inject_format_noise(
                result, noise_level=noise_level, seed=sub_seed,
                format_kind=format_kind,
            )
        else:
            raise ValueError(f"Unknown noise type: {ntype!r}")

    return result


def inject_noise_dataset(
    records: list[dict],
    noise_types: list[NoiseType] | None = None,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
) -> list[dict]:
    """Apply noise injection to an entire dataset (list of records).

    Each record gets a deterministic sub-seed derived from the master seed
    and its index, ensuring reproducibility even when records are reordered.

    Parameters
    ----------
    records : list of dict
        List of benchmark reaction records.
    noise_types : list, optional
        Noise types to apply.
    noise_level : "low" | "medium" | "high"
        Noise intensity.
    seed : int or None
        Master seed for the whole dataset.

    Returns
    -------
    list of dict
        Noisy copies of all records.
    """
    master_rng = random.Random(seed)
    noisy_records = []
    for record in records:
        rec_seed = master_rng.randint(0, 2**31)
        noisy_records.append(
            inject_noise(
                record,
                noise_types=noise_types,
                noise_level=noise_level,
                seed=rec_seed,
            )
        )
    return noisy_records
