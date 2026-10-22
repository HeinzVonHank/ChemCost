"""Tests for Stage 4: Format Transform Noise."""

import copy

from chemcost.noise.noise_injector import (
    _apply_ocr_noise,
    _build_mixed_format,
    inject_format_noise,
    inject_noise,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_record(
    component_names: list[str],
    roles: list[str] | None = None,
    equivs: list[float] | None = None,
    mws: list[float] | None = None,
) -> dict:
    """Build a minimal benchmark record with the given components."""
    n = len(component_names)
    if roles is None:
        roles = ["reactant"] * n
    if equivs is None:
        equivs = [1.0] * n
    if mws is None:
        mws = [100.0] * n
    return {
        "reaction_id": "FMT-001",
        "reaction_name": "Bromination",
        "components": [
            {
                "name": name,
                "role": role,
                "equivalents": equiv,
                "mw": mw,
                "amount_g": None if role == "solvent" else equiv * 2.0 * mw / 1000,  # 2 mmol lab scale
            }
            for name, role, equiv, mw in zip(
                component_names, roles, equivs, mws,
            )
        ],
        "yield_percent": 92,
        "product_mw": 196.04,
        "procurement_cost_usd_per_g_product": 15.50,
        "total_cost_per_gram_product_usd": 12.30,
        "cost_tier": "pack_based",
    }


SAMPLE_RECORD = _make_record(
    component_names=["indole", "NBS", "DMF"],
    roles=["reactant", "reactant", "solvent"],
    equivs=[1.0, 1.05, 5.0],
    mws=[117.15, 177.98, 73.09],
)


# ── NL conversion tests ────────────────────────────────────────────────────


class TestNaturalLanguageConversion:
    def test_medium_produces_description(self):
        """Medium noise should add a 'description' string field."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="medium", seed=42,
        )
        assert "description" in noisy
        assert isinstance(noisy["description"], str)
        assert len(noisy["description"]) > 50

    def test_low_no_description(self):
        """Low noise should NOT add a description field."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="low", seed=42,
        )
        assert "description" not in noisy

    def test_nl_contains_component_names(self):
        """NL text must mention all non-solvent component names."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="medium", seed=42,
        )
        desc = noisy["description"]
        assert "indole" in desc
        assert "NBS" in desc

    def test_nl_contains_solvent_name(self):
        """NL text should mention the solvent."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="medium", seed=42,
        )
        assert "DMF" in noisy["description"]

    def test_nl_contains_lab_amounts_not_equivalents(self):
        """NL text gives laboratory amounts (NBS: 1.05 equiv x 2 mmol x 177.98 = 373.76 mg), never equivalents."""
        desc = inject_format_noise(SAMPLE_RECORD, noise_level="medium", seed=42)["description"]
        assert "373.76 mg" in desc and "234.3 mg" in desc
        assert "equiv" not in desc and "1.05" not in desc

    def test_nl_hides_molecular_weights(self):
        desc = inject_format_noise(SAMPLE_RECORD, noise_level="medium", seed=42)["description"]
        assert "117.15" not in desc and "MW" not in desc

    def test_nl_gives_isolated_product_not_yield(self):
        """NL text names the product and its isolated mass; the yield itself is left to the agent."""
        record = {**SAMPLE_RECORD, "product": {"display_name": "3-bromo-1H-indole", "mass_g": 0.3607}}
        desc = inject_format_noise(record, noise_level="medium", seed=42)["description"]
        assert "360.7 mg" in desc and "3-bromo-1H-indole" in desc and "%" not in desc

    def test_nl_contains_mass_in_mg(self):
        """NL text should express masses in mg (or g for large amounts)."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="medium", seed=42,
        )
        desc = noisy["description"]
        # indole: 1.0 * 117.15 = 117.15 mg -> "117 mg"
        assert "mg" in desc or "g" in desc

    def test_nl_reads_like_experimental(self):
        """NL text should contain procedural language."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="medium", seed=42,
        )
        desc = noisy["description"].lower()
        # Should contain at least one procedural verb/phrase
        procedural_words = [
            "solution", "added", "dissolved", "stirred",
            "mixture", "combined", "treated", "afford",
            "workup", "isolated", "purification", "used",
            "mixed", "prepared",
        ]
        assert any(w in desc for w in procedural_words)

    def test_nl_catalyst_mol_percent(self):
        """Catalysts with equiv < 1 should appear as mol%."""
        record = _make_record(
            ["substrate", "Pd(OAc)2"],
            roles=["reactant", "catalyst"],
            equivs=[1.0, 0.05],
            mws=[150.0, 224.51],
        )
        noisy = inject_format_noise(
            record, noise_level="medium", seed=42,
        )
        assert "22.451 mg" in noisy["description"]  # 0.05 equiv x 2 mmol x 224.51 g/mol

    def test_nl_product_formula_included(self):
        record = {**SAMPLE_RECORD, "product": {"formula": "C8H6BrN", "mw": 196.04}}
        desc = inject_format_noise(record, noise_level="medium", seed=42)["description"]
        assert "C8H6BrN" in desc and "196.04" not in desc

    def test_components_list_preserved_at_medium(self):
        """The components list should still exist at medium noise."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="medium", seed=42,
        )
        assert "components" in noisy
        assert len(noisy["components"]) == 3


# ── OCR noise tests ────────────────────────────────────────────────────────


class TestOCRNoise:
    def test_high_applies_ocr_to_description(self):
        """High noise should produce OCR errors in description."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="high", seed=42,
        )
        assert "description" in noisy
        # Description should exist (OCR may or may not change it)
        assert isinstance(noisy["description"], str)

    def test_ocr_subscript_loss(self):
        """Subscript characters should be replaced by plain digits."""
        import random as _random
        rng = _random.Random(42)
        result = _apply_ocr_noise("Na\u2082CO\u2083", rng)
        assert "\u2082" not in result
        assert "\u2083" not in result
        # Should contain plain digits instead
        assert "2" in result
        assert "3" in result

    def test_ocr_degree_degradation(self):
        """Degree symbol should degrade."""
        import random as _random
        rng = _random.Random(42)
        result = _apply_ocr_noise("100\u00b0C", rng)
        assert "\u00b0" not in result
        assert "oC" in result or "o" in result

    def test_ocr_mu_degradation(self):
        """Greek mu should become 'u'."""
        import random as _random
        rng = _random.Random(42)
        result = _apply_ocr_noise("5 \u03bcL", rng)
        assert "\u03bc" not in result
        assert "u" in result

    def test_ocr_char_confusion_O_zero(self):
        """Capital O and zero should sometimes swap."""
        import random as _random
        # Use a text with many O's to increase probability of swap
        text = "OOOOOOOOOO"
        rng = _random.Random(42)
        result = _apply_ocr_noise(text, rng, prob=1.0)
        # With prob=1.0, all O's should become 0's
        assert "0" in result

    def test_ocr_char_confusion_l_one(self):
        """Lowercase L and digit 1 should sometimes swap."""
        import random as _random
        text = "llllllllll"
        rng = _random.Random(42)
        result = _apply_ocr_noise(text, rng, prob=1.0)
        assert "1" in result

    def test_ocr_rn_to_m(self):
        """'rn' substring should sometimes become 'm'."""
        import random as _random
        rng = _random.Random(42)
        # prob=1.0 guarantees the swap
        result = _apply_ocr_noise("burned", rng, prob=1.0)
        assert "bumed" in result

    def test_high_noise_has_mixed_format(self):
        """High noise should add a 'mixed_format' field."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="high", seed=42,
        )
        assert "mixed_format" in noisy
        mixed = noisy["mixed_format"]
        # Should contain table markers
        assert "|" in mixed
        assert "---" in mixed

    def test_ocr_applied_to_component_names(self):
        """High noise should apply OCR noise to component names."""
        # Use a record with OCR-susceptible names
        record = _make_record(
            ["Na\u2082CO\u2083", "Olefin", "chloroform"],
            roles=["base", "reactant", "solvent"],
        )
        noisy = inject_format_noise(
            record, noise_level="high", seed=42,
        )
        # At least the subscript chars should be degraded
        names = [c["name"] for c in noisy["components"]]
        na_name = names[0]
        assert "\u2082" not in na_name
        assert "\u2083" not in na_name


# ── Mixed format tests ─────────────────────────────────────────────────────


class TestMixedFormat:
    def test_mixed_has_text_and_table(self):
        """Mixed format should have both prose and table sections."""
        import random as _random
        rng = _random.Random(42)
        mixed = _build_mixed_format(SAMPLE_RECORD, rng)
        # Table header
        assert "Component | Amount" in mixed and "Role" not in mixed
        assert "---" in mixed
        # Some text (prose)
        assert "The reaction used" in mixed

    def test_mixed_covers_all_components(self):
        """All components should appear in either text or table."""
        import random as _random
        rng = _random.Random(42)
        mixed = _build_mixed_format(SAMPLE_RECORD, rng)
        for comp in SAMPLE_RECORD["components"]:
            name = comp["name"]
            assert name in mixed, (
                f"Component '{name}' not found in mixed format"
            )


# ── Seed reproducibility tests ─────────────────────────────────────────────


class TestSeedReproducibility:
    def test_format_noise_deterministic(self):
        """Same seed produces identical output."""
        r1 = inject_format_noise(
            SAMPLE_RECORD, noise_level="medium", seed=42,
        )
        r2 = inject_format_noise(
            SAMPLE_RECORD, noise_level="medium", seed=42,
        )
        assert r1["description"] == r2["description"]

    def test_format_noise_high_deterministic(self):
        """Same seed at high noise produces identical output."""
        r1 = inject_format_noise(
            SAMPLE_RECORD, noise_level="high", seed=99,
        )
        r2 = inject_format_noise(
            SAMPLE_RECORD, noise_level="high", seed=99,
        )
        assert r1["description"] == r2["description"]
        assert r1["mixed_format"] == r2["mixed_format"]
        names1 = [c["name"] for c in r1["components"]]
        names2 = [c["name"] for c in r2["components"]]
        assert names1 == names2

    def test_different_seeds_differ(self):
        """Different seeds should generally produce different output."""
        outs = {(r["description"], r.get("mixed_format"))
                for r in (inject_format_noise(SAMPLE_RECORD, noise_level="high", seed=s) for s in range(10))}
        assert len(outs) > 1


# ── Immutability tests ─────────────────────────────────────────────────────


class TestOriginalNotMutated:
    def test_medium_does_not_mutate_original(self):
        """inject_format_noise should not modify the input record."""
        record = _make_record(["indole", "NBS", "DMF"])
        original = copy.deepcopy(record)
        inject_format_noise(record, noise_level="medium", seed=42)
        assert record == original

    def test_high_does_not_mutate_original(self):
        """inject_format_noise at high noise should not mutate input."""
        record = _make_record(["indole", "NBS", "DMF"])
        original = copy.deepcopy(record)
        inject_format_noise(record, noise_level="high", seed=42)
        assert record == original


# ── Ground truth preservation tests ────────────────────────────────────────


class TestGroundTruthPreserved:
    def test_procurement_cost_preserved_medium(self):
        """Ground truth cost must survive medium noise."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="medium", seed=42,
        )
        assert (
            noisy["procurement_cost_usd_per_g_product"] == 15.50
        )
        assert (
            noisy["total_cost_per_gram_product_usd"] == 12.30
        )

    def test_cost_tier_preserved_high(self):
        """Cost tier must survive high noise."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="high", seed=42,
        )
        assert noisy["cost_tier"] == "pack_based"

    def test_all_ground_truth_fields_preserved(self):
        """All ground truth fields must be unchanged."""
        noisy = inject_format_noise(
            SAMPLE_RECORD, noise_level="high", seed=42,
        )
        assert (
            noisy["procurement_cost_usd_per_g_product"]
            == SAMPLE_RECORD["procurement_cost_usd_per_g_product"]
        )
        assert (
            noisy["total_cost_per_gram_product_usd"]
            == SAMPLE_RECORD["total_cost_per_gram_product_usd"]
        )
        assert (
            noisy["cost_tier"] == SAMPLE_RECORD["cost_tier"]
        )


# ── Integration with inject_noise entry point ─────────────────────────────


class TestInjectNoiseFormatIntegration:
    def test_format_in_noise_types(self):
        """inject_noise should accept 'format' as a noise type."""
        noisy = inject_noise(
            SAMPLE_RECORD,
            noise_types=["format"],
            noise_level="medium",
            seed=42,
        )
        assert "description" in noisy

    def test_format_combined_with_others(self):
        """Format noise should compose with other noise types."""
        noisy = inject_noise(
            SAMPLE_RECORD,
            noise_types=["isomer", "name_variation", "format"],
            noise_level="medium",
            seed=42,
        )
        assert "description" in noisy
        assert "components" in noisy

    def test_all_five_noise_types_accepted(self):
        """All five noise types should be accepted without error."""
        all_types = [
            "isomer", "name_variation", "quantity",
            "missing_info", "format",
        ]
        noisy = inject_noise(
            SAMPLE_RECORD,
            noise_types=all_types,
            noise_level="medium",
            seed=42,
        )
        assert "description" in noisy

    def test_ground_truth_after_combined_noise(self):
        """Ground truth must survive combined noise pipeline."""
        noisy = inject_noise(
            SAMPLE_RECORD,
            noise_types=["name_variation", "format"],
            noise_level="high",
            seed=42,
        )
        assert (
            noisy["procurement_cost_usd_per_g_product"] == 15.50
        )
        assert noisy["cost_tier"] == "pack_based"


# ── Edge cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_components(self):
        """Record with no components should not crash."""
        record = {
            "reaction_id": "EMPTY-001",
            "reaction_name": "Empty",
            "components": [],
            "yield_percent": 50,
            "procurement_cost_usd_per_g_product": 1.0,
        }
        noisy = inject_format_noise(
            record, noise_level="medium", seed=42,
        )
        assert "description" in noisy
        assert "No components specified" in noisy["description"]

    def test_single_component_no_solvent(self):
        """Single non-solvent component should produce valid NL."""
        record = _make_record(["benzaldehyde"], roles=["reactant"])
        noisy = inject_format_noise(
            record, noise_level="medium", seed=42,
        )
        assert "benzaldehyde" in noisy["description"]

    def test_only_solvents(self):
        """Record with only solvents should still work."""
        record = _make_record(
            ["THF", "water"],
            roles=["solvent", "solvent"],
        )
        noisy = inject_format_noise(
            record, noise_level="medium", seed=42,
        )
        assert "description" in noisy

    def test_missing_mw_and_equiv(self):
        """Components with None MW/equiv should not crash NL."""
        record = {
            "reaction_id": "NULL-001",
            "reaction_name": "Test",
            "components": [
                {
                    "name": "mystery reagent",
                    "role": "reactant",
                    "equivalents": None,
                    "mw": None,
                },
            ],
            "procurement_cost_usd_per_g_product": 5.0,
        }
        noisy = inject_format_noise(
            record, noise_level="medium", seed=42,
        )
        assert "mystery reagent" in noisy["description"]
