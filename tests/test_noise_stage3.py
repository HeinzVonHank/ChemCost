"""Tests for Stage 3: Missing Information Noise injection."""

import copy

import pytest

from chemcost.noise.noise_injector import (
    inject_missing_info_noise,
    inject_noise,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_record(
    n_components: int = 4,
    *,
    yield_percent: float = 85.0,
    include_product: bool = False,
) -> dict:
    """Build a realistic benchmark record for missing-info tests."""
    roles = ["reactant", "catalyst", "base", "solvent"]
    components = [
        {
            "name": f"chemical_{i}",
            "role": roles[i % len(roles)],
            "equivalents": 1.0 + i * 0.5,
            "mw": 100.0 + i * 20.0,
        }
        for i in range(n_components)
    ]
    if include_product:
        components.append({
            "name": "target_compound",
            "role": "product",
            "equivalents": 1.0,
            "mw": 250.0,
        })
    return {
        "reaction_id": "TEST-STAGE3-001",
        "reaction_name": "Test Reaction for Missing Info",
        "yield_percent": yield_percent,
        "product_smiles": "CC(=O)O",
        "components": components,
        # Ground truth fields — must NEVER be touched.
        "procurement_cost_usd_per_g_product": 42.0,
        "cost_tier": "pack_based",
        "cost_model": "v2",
    }


# ── MW dropping tests ────────────────────────────────────────────────────────


class TestMWDropping:
    def test_low_drops_some_mw(self):
        """At low noise, ~25% of component MWs should be dropped."""
        record = _make_record(20)
        noisy = inject_missing_info_noise(record, noise_level="low", seed=42)
        dropped = [
            c for c in noisy["components"] if c["mw"] is None
        ]
        # With 20 components at 25%, expect roughly 5 — allow range.
        assert 1 <= len(dropped) <= 12

    def test_medium_drops_more_mw(self):
        """At medium noise, ~50% of component MWs should be dropped."""
        record = _make_record(20)
        noisy = inject_missing_info_noise(
            record, noise_level="medium", seed=42,
        )
        dropped = [
            c for c in noisy["components"] if c["mw"] is None
        ]
        assert len(dropped) >= 5

    def test_high_drops_most_mw(self):
        """At high noise, ~80% of component MWs should be dropped."""
        record = _make_record(20)
        noisy = inject_missing_info_noise(
            record, noise_level="high", seed=42,
        )
        dropped = [
            c for c in noisy["components"] if c["mw"] is None
        ]
        assert len(dropped) >= 10

    def test_mw_drop_tags_component(self):
        """Dropped MW components should have noise_type and original_mw."""
        record = _make_record(10)
        noisy = inject_missing_info_noise(
            record, noise_level="high", seed=42,
        )
        for comp in noisy["components"]:
            if comp["mw"] is None:
                assert comp.get("original_mw") is not None
                ntype = comp.get("noise_type", "")
                assert ntype in (
                    "mw_dropped", "mw_and_role_dropped",
                )

    def test_mw_already_none_skipped(self):
        """Components with mw=None should not be re-tagged."""
        record = _make_record(4)
        record["components"][0]["mw"] = None
        noisy = inject_missing_info_noise(
            record, noise_level="high", seed=42,
        )
        comp = noisy["components"][0]
        assert comp["mw"] is None
        assert "original_mw" not in comp


# ── Yield dropping tests ─────────────────────────────────────────────────────


class TestYieldDropping:
    def test_low_never_drops_yield(self):
        """At low noise, yield should never be dropped."""
        record = _make_record()
        # Run many seeds to be sure.
        for seed in range(50):
            noisy = inject_missing_info_noise(
                record, noise_level="low", seed=seed,
            )
            assert noisy["yield_percent"] == 85.0

    def test_medium_never_drops_yield(self):
        """At medium noise, yield stays visible in the current benchmark design."""
        record = _make_record()
        for seed in range(50):
            noisy = inject_missing_info_noise(
                record, noise_level="medium", seed=seed,
            )
            assert noisy["yield_percent"] == 85.0

    def test_high_drops_yield_more_often(self):
        """At high noise, yield is dropped ~60% of the time."""
        record = _make_record()
        dropped_count = 0
        trials = 100
        for seed in range(trials):
            noisy = inject_missing_info_noise(
                record, noise_level="high", seed=seed,
            )
            if noisy["yield_percent"] is None:
                dropped_count += 1
        assert dropped_count >= 30

    def test_yield_drop_tags_record(self):
        """Dropped yield should be tagged in noise_applied."""
        record = _make_record()
        # Find a seed that triggers yield drop at high noise.
        for seed in range(100):
            noisy = inject_missing_info_noise(
                record, noise_level="high", seed=seed,
            )
            if noisy["yield_percent"] is None:
                assert "yield_dropped" in noisy["noise_applied"]
                assert noisy["original_yield_percent"] == 85.0
                return
        pytest.fail("No seed produced a yield drop at high noise")

    def test_yield_already_none_skipped(self):
        """If yield_percent is already None, no drop should happen."""
        record = _make_record()
        record["yield_percent"] = None
        noisy = inject_missing_info_noise(
            record, noise_level="high", seed=42,
        )
        assert "yield_dropped" not in noisy.get("noise_applied", [])
        assert "original_yield_percent" not in noisy


# ── Role dropping tests ──────────────────────────────────────────────────────


class TestRoleDropping:
    def test_low_never_drops_role(self):
        """At low noise, roles should never be dropped."""
        record = _make_record(10)
        for seed in range(50):
            noisy = inject_missing_info_noise(
                record, noise_level="low", seed=seed,
            )
            for comp in noisy["components"]:
                assert comp["role"] != "unknown"

    def test_medium_never_drops_roles(self):
        """At medium noise, role labels are preserved."""
        record = _make_record(20)
        for seed in range(50):
            noisy = inject_missing_info_noise(
                record, noise_level="medium", seed=seed,
            )
            assert all(c["role"] != "unknown" for c in noisy["components"])

    def test_high_drops_many_roles(self):
        """At high noise, ~60% of roles become 'unknown'."""
        record = _make_record(20)
        noisy = inject_missing_info_noise(
            record, noise_level="high", seed=42,
        )
        unknown_count = sum(
            1 for c in noisy["components"] if c["role"] == "unknown"
        )
        assert unknown_count >= 5

    def test_role_drop_tags_component(self):
        """Dropped roles should be tagged with original_role."""
        record = _make_record(10)
        noisy = inject_missing_info_noise(
            record, noise_level="high", seed=42,
        )
        for comp in noisy["components"]:
            if comp["role"] == "unknown":
                assert comp.get("original_role") is not None
                assert comp["original_role"] != "unknown"
                ntype = comp.get("noise_type", "")
                assert "role_dropped" in ntype

    def test_already_unknown_skipped(self):
        """Components with role='unknown' should not be re-tagged."""
        record = _make_record(4)
        record["components"][0]["role"] = "unknown"
        noisy = inject_missing_info_noise(
            record, noise_level="high", seed=42,
        )
        comp = noisy["components"][0]
        assert comp["role"] == "unknown"
        assert "original_role" not in comp

    def test_combined_mw_and_role_drop_tag(self):
        """A component with both MW and role dropped gets combined tag."""
        record = _make_record(20)
        noisy = inject_missing_info_noise(
            record, noise_level="high", seed=42,
        )
        combined = [
            c for c in noisy["components"]
            if c.get("noise_type") == "mw_and_role_dropped"
        ]
        # At high noise with 20 components, it is very likely at least
        # one component gets both dropped.
        assert len(combined) >= 1


# ── Product info dropping tests ──────────────────────────────────────────────


class TestProductDropping:
    @pytest.mark.parametrize("noise_level", ["low", "medium", "high"])
    def test_default_pipeline_never_drops_product(self, noise_level):
        """Product info is not part of the current default missing-info benchmark."""
        record = _make_record(4, include_product=True)
        for seed in range(50):
            noisy = inject_missing_info_noise(
                record, noise_level=noise_level, seed=seed,
            )
            assert noisy["product_smiles"] == "CC(=O)O"
            assert "product_dropped" not in noisy.get("noise_applied", [])
            product_names = [
                c["name"]
                for c in noisy["components"]
                if c.get("role") == "product" or c.get("original_role") == "product"
            ]
            assert product_names == ["target_compound"]


# ── Seed reproducibility ─────────────────────────────────────────────────────


class TestSeedReproducibility:
    def test_same_seed_same_result(self):
        """Identical seed produces identical output."""
        record = _make_record(10, include_product=True)
        r1 = inject_missing_info_noise(
            record, noise_level="high", seed=12345,
        )
        r2 = inject_missing_info_noise(
            record, noise_level="high", seed=12345,
        )
        assert r1 == r2

    def test_different_seed_different_result(self):
        """Different seeds produce different output (probabilistically)."""
        record = _make_record(10, include_product=True)
        r1 = inject_missing_info_noise(
            record, noise_level="high", seed=1,
        )
        r2 = inject_missing_info_noise(
            record, noise_level="high", seed=999,
        )
        # At least one field should differ.
        assert r1 != r2


# ── Original record not mutated ──────────────────────────────────────────────


class TestOriginalNotMutated:
    def test_direct_call_no_mutation(self):
        """inject_missing_info_noise must not modify the input record."""
        record = _make_record(6, include_product=True)
        original = copy.deepcopy(record)
        inject_missing_info_noise(record, noise_level="high", seed=42)
        assert record == original

    def test_via_inject_noise_no_mutation(self):
        """inject_noise with missing_info must not modify the input."""
        record = _make_record(6, include_product=True)
        original = copy.deepcopy(record)
        inject_noise(
            record,
            noise_types=["missing_info"],
            noise_level="high",
            seed=42,
        )
        assert record == original


# ── Ground truth fields NEVER touched ────────────────────────────────────────


class TestGroundTruthPreserved:
    @pytest.mark.parametrize("noise_level", ["low", "medium", "high"])
    def test_cost_fields_untouched(self, noise_level):
        """procurement_cost, cost_tier, cost_model must survive."""
        record = _make_record(10, include_product=True)
        noisy = inject_missing_info_noise(
            record, noise_level=noise_level, seed=42,
        )
        assert noisy["procurement_cost_usd_per_g_product"] == 42.0
        assert noisy["cost_tier"] == "pack_based"
        assert noisy["cost_model"] == "v2"

    @pytest.mark.parametrize("noise_level", ["low", "medium", "high"])
    def test_cost_fields_via_inject_noise(self, noise_level):
        """Ground truth survives the combined inject_noise entry point."""
        record = _make_record(10, include_product=True)
        noisy = inject_noise(
            record,
            noise_types=["missing_info"],
            noise_level=noise_level,
            seed=42,
        )
        assert noisy["procurement_cost_usd_per_g_product"] == 42.0
        assert noisy["cost_tier"] == "pack_based"
        assert noisy["cost_model"] == "v2"


# ── inject_noise integration ─────────────────────────────────────────────────


class TestInjectNoiseIntegration:
    def test_missing_info_via_inject_noise(self):
        """inject_noise dispatches to inject_missing_info_noise."""
        record = _make_record(10, include_product=True)
        noisy = inject_noise(
            record,
            noise_types=["missing_info"],
            noise_level="high",
            seed=42,
        )
        # At high noise some MWs should be dropped.
        dropped = [
            c for c in noisy["components"] if c["mw"] is None
        ]
        assert len(dropped) >= 1

    def test_combined_with_name_noise(self):
        """missing_info can be combined with other noise types."""
        record = _make_record(10, include_product=True)
        noisy = inject_noise(
            record,
            noise_types=["isomer", "name_variation", "missing_info"],
            noise_level="high",
            seed=42,
        )
        # Should still deep-copy and have components.
        assert len(noisy["components"]) == len(record["components"])
        assert noisy is not record

    def test_quantity_noise_dispatches(self):
        """Quantity noise should dispatch without error."""
        record = _make_record(4)
        noisy = inject_noise(
            record,
            noise_types=["quantity"],
            noise_level="high",
            seed=42,
        )
        assert len(noisy["components"]) == 4

    def test_unknown_noise_type_still_raises(self):
        """Unknown noise type should still raise ValueError."""
        record = _make_record(4)
        with pytest.raises(ValueError, match="Unknown noise type"):
            inject_noise(
                record,
                noise_types=["bogus"],
                noise_level="low",
            )
