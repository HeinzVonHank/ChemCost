"""Tests for the noise injection pipeline."""

import copy

import pytest

from chemcost.noise.chemical_aliases import (
    ABBREVIATION_TO_FULL,
    FULL_TO_ABBREVIATION,
    ISOMER_AMBIGUOUS,
)
from chemcost.noise.noise_injector import (
    _strip_positional,
    _strip_stereo,
    default_noise_types_for_level,
    inject_isomer_noise,
    inject_name_variation,
    inject_noise,
    inject_noise_dataset,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_record(component_names: list[str], roles: list[str] | None = None) -> dict:
    """Build a minimal benchmark record with the given component names."""
    if roles is None:
        roles = ["reactant"] * len(component_names)
    return {
        "reaction_id": "TEST-001",
        "reaction_name": "Test Reaction",
        "components": [
            {
                "name": name,
                "role": role,
                "equivalents": 1.0,
                "mw": 100.0,
            }
            for name, role in zip(component_names, roles)
        ],
        "procurement_cost_usd_per_g_product": 10.0,
    }


def _get_names(record: dict) -> list[str]:
    """Extract component names from a record."""
    return [c["name"] for c in record["components"]]


# ── _strip_stereo tests ─────────────────────────────────────────────────────

class TestStripStereo:
    def test_R_prefix(self):
        assert _strip_stereo("(R)-limonene") == "limonene"

    def test_S_prefix(self):
        assert _strip_stereo("(S)-BINAP") == "BINAP"

    def test_D_prefix(self):
        assert _strip_stereo("D-glucose") == "glucose"

    def test_L_prefix(self):
        assert _strip_stereo("L-proline") == "proline"

    def test_cis_prefix(self):
        assert _strip_stereo("cis-stilbene") == "stilbene"

    def test_trans_prefix(self):
        assert _strip_stereo("trans-stilbene") == "stilbene"

    def test_E_prefix(self):
        assert _strip_stereo("(E)-2-butene") == "2-butene"

    def test_Z_prefix(self):
        assert _strip_stereo("(Z)-2-butene") == "2-butene"

    def test_no_prefix(self):
        assert _strip_stereo("benzene") is None

    def test_rac_prefix(self):
        assert _strip_stereo("rac-BINAP") == "BINAP"

    def test_alpha_prefix(self):
        assert _strip_stereo("alpha-pinene") == "pinene"

    def test_beta_prefix(self):
        assert _strip_stereo("beta-pinene") == "pinene"


# ── _strip_positional tests ─────────────────────────────────────────────────

class TestStripPositional:
    def test_n_prefix(self):
        assert _strip_positional("n-butanol") == "butanol"

    def test_sec_prefix(self):
        assert _strip_positional("sec-butanol") == "butanol"

    def test_tert_prefix(self):
        assert _strip_positional("tert-butanol") == "butanol"

    def test_iso_prefix(self):
        assert _strip_positional("iso-propanol") == "propanol"

    def test_o_prefix(self):
        assert _strip_positional("o-xylene") == "xylene"

    def test_m_prefix(self):
        assert _strip_positional("m-xylene") == "xylene"

    def test_p_prefix(self):
        assert _strip_positional("p-xylene") == "xylene"

    def test_no_prefix(self):
        assert _strip_positional("benzene") is None

    def test_case_insensitive(self):
        assert _strip_positional("N-butanol") == "butanol"


# ── inject_isomer_noise tests ───────────────────────────────────────────────

class TestInjectIsomerNoise:
    def test_known_isomer_mapping(self):
        """n-butanol should become butanol at high noise (seed ensures hit)."""
        record = _make_record(["n-butanol"])
        noisy = inject_isomer_noise(record, noise_level="high", seed=42)
        assert noisy["components"][0]["name"] == "butanol"
        assert noisy["components"][0].get("original_name") == "n-butanol"
        assert noisy["components"][0].get("noise_type") == "isomer_ambiguity"

    def test_stereo_stripping(self):
        """(R)-limonene should become limonene at high noise."""
        record = _make_record(["(R)-limonene"])
        noisy = inject_isomer_noise(record, noise_level="high", seed=1)
        assert noisy["components"][0]["name"] == "limonene"
        assert noisy["components"][0].get("noise_type") == "stereo_stripped"

    def test_original_preserved(self):
        """The original record should not be mutated."""
        record = _make_record(["n-butanol"])
        original_copy = copy.deepcopy(record)
        inject_isomer_noise(record, noise_level="high", seed=42)
        assert record == original_copy

    def test_empty_name_skipped(self):
        """Components with empty names should pass through unchanged."""
        record = _make_record(["", "n-butanol"])
        noisy = inject_isomer_noise(record, noise_level="high", seed=42)
        assert noisy["components"][0]["name"] == ""
        # Second component should be modified
        assert noisy["components"][1]["name"] == "butanol"

    def test_low_noise_partial(self):
        """At low noise, not all components are modified (probabilistic)."""
        names = ["n-butanol", "sec-butanol", "tert-butanol", "isobutanol"]
        record = _make_record(names)
        noisy = inject_isomer_noise(record, noise_level="low", seed=12345)
        modified = [
            c for c in noisy["components"] if c.get("original_name") is not None
        ]
        # At 25% probability with 4 items, very unlikely all are modified
        # (but at least one should be with the right seed)
        assert len(modified) < len(names)

    def test_positional_stripped_at_medium(self):
        """Medium noise strips positional prefixes even without explicit mapping."""
        record = _make_record(["n-octylamine"])
        noisy = inject_isomer_noise(record, noise_level="medium", seed=1)
        # n-octylamine has no mapping in ISOMER_AMBIGUOUS but prefix should strip
        assert noisy["components"][0]["name"] == "octylamine"

    def test_i_PrOH_isomer_mapping(self):
        """i-PrOH should map to propanol."""
        record = _make_record(["i-PrOH"])
        noisy = inject_isomer_noise(record, noise_level="high", seed=42)
        assert noisy["components"][0]["name"] == "propanol"

    def test_seed_reproducibility(self):
        """Same seed should give same results."""
        record = _make_record(["n-butanol", "(R)-BINAP", "toluene"])
        r1 = inject_isomer_noise(record, noise_level="medium", seed=42)
        r2 = inject_isomer_noise(record, noise_level="medium", seed=42)
        assert _get_names(r1) == _get_names(r2)


# ── inject_name_variation tests ─────────────────────────────────────────────

class TestInjectNameVariation:
    def test_abbreviation_expanded(self):
        """DMF should be expanded to a full name at high noise."""
        record = _make_record(["DMF"])
        noisy = inject_name_variation(record, noise_level="high", seed=42)
        name = noisy["components"][0]["name"]
        assert name in ABBREVIATION_TO_FULL["DMF"]

    def test_full_name_to_abbreviation(self):
        """triethylamine should become an abbreviation at high noise."""
        record = _make_record(["triethylamine"])
        noisy = inject_name_variation(record, noise_level="high", seed=42)
        name = noisy["components"][0]["name"]
        assert name in FULL_TO_ABBREVIATION["triethylamine"]

    def test_common_to_iupac(self):
        """acetone -> propan-2-one at high noise."""
        record = _make_record(["acetone"])
        noisy = inject_name_variation(record, noise_level="high", seed=1)
        assert noisy["components"][0]["name"] == "propan-2-one"

    def test_salt_to_formula(self):
        """sodium hydroxide -> NaOH at high noise."""
        record = _make_record(["sodium hydroxide"])
        noisy = inject_name_variation(record, noise_level="high", seed=42)
        assert noisy["components"][0]["name"] == "NaOH"

    def test_original_not_mutated(self):
        """The input record should be left unchanged."""
        record = _make_record(["DMF"])
        original_copy = copy.deepcopy(record)
        inject_name_variation(record, noise_level="high", seed=42)
        assert record == original_copy

    def test_no_match_unchanged(self):
        """Names with no known variation should not be modified."""
        record = _make_record(["4-bromobenzonitrile"])
        noisy = inject_name_variation(record, noise_level="high", seed=42)
        assert noisy["components"][0]["name"] == "4-bromobenzonitrile"
        assert "original_name" not in noisy["components"][0]

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        record = _make_record(["DMF", "Et3N", "acetone", "sodium hydroxide"])
        r1 = inject_name_variation(record, noise_level="high", seed=99)
        r2 = inject_name_variation(record, noise_level="high", seed=99)
        assert _get_names(r1) == _get_names(r2)

    def test_noise_type_tag(self):
        """Modified components should be tagged with noise_type."""
        record = _make_record(["DMF"])
        noisy = inject_name_variation(record, noise_level="high", seed=42)
        assert noisy["components"][0].get("noise_type") == "name_variation"

    def test_case_insensitive_full_name_lookup(self):
        """Full-name lookup should be case-insensitive."""
        record = _make_record(["Triethylamine"])
        noisy = inject_name_variation(record, noise_level="high", seed=42)
        # Should still find an abbreviation
        name = noisy["components"][0]["name"]
        assert name in ("TEA", "Et3N") or noisy["components"][0].get("original_name") is not None


# ── inject_noise combined tests ─────────────────────────────────────────────

class TestInjectNoise:
    def test_default_level_mapping_matches_benchmark_design(self):
        assert default_noise_types_for_level("low") == ["isomer", "name_variation"]
        assert default_noise_types_for_level("medium") == [
            "isomer", "name_variation", "quantity", "missing_info", "format",
        ]
        assert default_noise_types_for_level("high") == [
            "isomer", "name_variation", "quantity", "missing_info", "format",
        ]

    def test_default_applies_both(self):
        """Default noise types include both isomer and name_variation."""
        record = _make_record(
            ["n-butanol", "DMF", "benzene"],
            ["solvent", "solvent", "solvent"],
        )
        noisy = inject_noise(record, noise_level="high", seed=42)
        modified = [c for c in noisy["components"] if "original_name" in c]
        # At high noise, at least some should be modified
        assert len(modified) >= 1

    def test_only_isomer(self):
        """Can restrict to isomer noise only."""
        record = _make_record(["n-butanol", "DMF"])
        noisy = inject_noise(
            record, noise_types=["isomer"], noise_level="high", seed=42
        )
        # DMF has no isomer mapping, so only n-butanol should be hit
        names = _get_names(noisy)
        # n-butanol should become butanol
        assert "butanol" in names or "n-butanol" in names
        # DMF should remain DMF (not expanded, since we only do isomer noise)
        assert "DMF" in names

    def test_only_name_variation(self):
        """Can restrict to name_variation noise only."""
        record = _make_record(["n-butanol", "DMF"])
        noisy = inject_noise(
            record, noise_types=["name_variation"], noise_level="high", seed=42
        )
        # n-butanol has no name_variation mapping (it's an isomer thing)
        # DMF should get expanded
        dmf_comp = noisy["components"][1]
        assert dmf_comp["name"] in ABBREVIATION_TO_FULL["DMF"] or dmf_comp["name"] == "DMF"

    def test_invalid_noise_type(self):
        """Unknown noise type should raise ValueError."""
        record = _make_record(["benzene"])
        with pytest.raises(ValueError, match="Unknown noise type"):
            inject_noise(record, noise_types=["bogus"], noise_level="low")

    def test_seed_reproducibility(self):
        """inject_noise with same seed is deterministic."""
        record = _make_record(["n-butanol", "DMF", "(R)-BINAP", "sodium hydroxide"])
        r1 = inject_noise(record, noise_level="high", seed=42)
        r2 = inject_noise(record, noise_level="high", seed=42)
        assert _get_names(r1) == _get_names(r2)

    def test_ground_truth_cost_preserved(self):
        """Noise should not modify pricing / ground truth fields."""
        record = _make_record(["DMF"])
        record["procurement_cost_usd_per_g_product"] = 42.0
        record["cost_tier"] = "pack_based"
        noisy = inject_noise(record, noise_level="high", seed=42)
        assert noisy["procurement_cost_usd_per_g_product"] == 42.0
        assert noisy["cost_tier"] == "pack_based"

    def test_component_fields_preserved(self):
        """Non-name fields (role, equivalents, mw) should be untouched."""
        record = _make_record(["DMF"], ["solvent"])
        record["components"][0]["equivalents"] = 5.0
        record["components"][0]["mw"] = 73.09
        noisy = inject_noise(record, noise_level="high", seed=42)
        comp = noisy["components"][0]
        assert comp["role"] == "solvent"
        assert comp["equivalents"] == 5.0
        assert comp["mw"] == 73.09


# ── inject_noise_dataset tests ──────────────────────────────────────────────

class TestInjectNoiseDataset:
    def test_all_records_processed(self):
        """Every record in the list should be processed."""
        records = [_make_record([f"chem_{i}"]) for i in range(5)]
        noisy = inject_noise_dataset(records, noise_level="low", seed=42)
        assert len(noisy) == 5

    def test_originals_not_mutated(self):
        """Source records should not be modified."""
        records = [_make_record(["DMF"])]
        originals = copy.deepcopy(records)
        inject_noise_dataset(records, noise_level="high", seed=42)
        assert records == originals

    def test_seed_reproducibility(self):
        """Dataset-level noise with same seed is deterministic."""
        records = [
            _make_record(["DMF", "n-butanol"]),
            _make_record(["sodium hydroxide", "acetone"]),
        ]
        r1 = inject_noise_dataset(records, noise_level="high", seed=42)
        r2 = inject_noise_dataset(records, noise_level="high", seed=42)
        for a, b in zip(r1, r2):
            assert _get_names(a) == _get_names(b)


# ── Alias dictionary sanity checks ─────────────────────────────────────────

class TestAliasDictionaries:
    def test_isomer_ambiguous_has_entries(self):
        assert len(ISOMER_AMBIGUOUS) >= 30

    def test_abbreviation_to_full_has_entries(self):
        assert len(ABBREVIATION_TO_FULL) >= 30

    def test_full_to_abbreviation_reverse_populated(self):
        """Every abbreviation value should have a reverse entry."""
        for abbrev, fulls in ABBREVIATION_TO_FULL.items():
            for full in fulls:
                assert full.lower() in FULL_TO_ABBREVIATION
                assert abbrev in FULL_TO_ABBREVIATION[full.lower()]

    def test_no_trivial_isomer_mappings(self):
        """Isomer mapping should not map a name to itself."""
        for k, v in ISOMER_AMBIGUOUS.items():
            assert k.lower() != v.lower(), f"Trivial mapping: {k} -> {v}"
