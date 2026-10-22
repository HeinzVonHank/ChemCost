from chemcost.pricing.price_entry import PriceEntry
from chemcost.pricing.pricing_db import PricingDB


def _make_db(tmp_path):
    """Create a DB with two pack sizes for anisole."""
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="anisole", smiles="COc1ccccc1", mw=108.14)
    db.set_normalized_price(chem_id, 5.0, "single", 1, ["TestSupplier"])
    for qty, price, purity in [
        (1.0, 5.0, "99"),
        (5.0, 20.0, "99"),
        (25.0, 80.0, "99"),
    ]:
        db.add_price(
            chem_id,
            PriceEntry(
                chemical_name="anisole",
                cas=None,
                supplier="TestSupplier",
                quantity_g=qty,
                price_usd=price,
                price_per_gram_usd=price / qty,
                purity=purity,
            ),
        )
    return db, chem_id


def test_get_price_can_match_by_smiles(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="anisole", smiles="COc1ccccc1", mw=108.14)
    db.add_price(
        chem_id,
        PriceEntry(
            chemical_name="anisole",
            cas=None,
            supplier="TestSupplier",
            quantity_g=1.0,
            price_usd=5.0,
            price_per_gram_usd=5.0,
        ),
    )
    db.set_normalized_price(
        chemical_id=chem_id,
        price_per_gram=5.0,
        method="single",
        n_sources=1,
        sources=["TestSupplier"],
    )

    assert db.get_price(smiles="COc1ccccc1") == 5.0
    db.close()


def test_get_pack_quotes_returns_sorted_packs(tmp_path):
    db, _ = _make_db(tmp_path)
    quotes = db.get_pack_quotes(smiles="COc1ccccc1")
    assert len(quotes) == 3
    assert quotes[0]["quantity_g"] == 1.0
    assert quotes[-1]["quantity_g"] == 25.0
    db.close()


def test_get_pack_quotes_filters_low_purity(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="crude", smiles="CCO", mw=46.07)
    for qty, purity in [(1.0, "85"), (1.0, "97")]:
        db.add_price(
            chem_id,
            PriceEntry(chemical_name="crude", cas=None, supplier="S",
                       quantity_g=qty, price_usd=10.0, price_per_gram_usd=10.0, purity=purity),
        )
    quotes = db.get_pack_quotes(smiles="CCO", min_purity=95.0)
    assert len(quotes) == 1
    assert quotes[0]["purity"] == "97"
    db.close()


def test_get_pack_quotes_excludes_unknown_purity(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="mystery", smiles="CCC", mw=44.06)
    db.add_price(
        chem_id,
        PriceEntry(chemical_name="mystery", cas=None, supplier="S",
                   quantity_g=1.0, price_usd=3.0, price_per_gram_usd=3.0, purity=None),
    )
    quotes = db.get_pack_quotes(smiles="CCC")
    assert quotes == []  # purity >= 95% cannot be confirmed, so the quote does not qualify


def test_procurement_picks_cheapest_option_for_small_need(tmp_path):
    db, _ = _make_db(tmp_path)
    # Require 0.05 g → one 1 g pack at $5 is the cheapest option
    result = db.get_procurement_price(required_mass_g=0.05, smiles="COc1ccccc1")
    assert result["tier"] == "pack_based"
    assert result["total_cost_usd"] == 5.0
    assert result["quantity_g"] == 1.0
    assert result["n_packs"] == 1
    db.close()


def test_procurement_buys_several_small_packs_when_cheaper(tmp_path):
    db, _ = _make_db(tmp_path)
    # Require 3 g → 3 x 1 g = $15 beats 1 x 5 g = $20 and 1 x 25 g = $80
    result = db.get_procurement_price(required_mass_g=3.0, smiles="COc1ccccc1")
    assert result["tier"] == "pack_based"
    assert result["total_cost_usd"] == 15.0
    assert result["n_packs"] == 3
    assert result["quantity_g"] == 3.0
    db.close()


def test_procurement_mixes_pack_sizes(tmp_path):
    db, _ = _make_db(tmp_path)
    # Require 30 g → 25 g + 5 g = $100 beats 6 x 5 g = $120, 2 x 25 g = $160 and 30 x 1 g = $150
    result = db.get_procurement_price(required_mass_g=30.0, smiles="COc1ccccc1")
    assert result["tier"] == "pack_based"
    assert result["n_packs"] == 2 and result["quantity_g"] == 30.0
    assert result["total_cost_usd"] == 100.0
    # Require 6 g → 5 g + 1 g = $25 beats 2 x 5 g = $40 and the 25 g pack at $80
    assert db.get_procurement_price(required_mass_g=6.0, smiles="COc1ccccc1")["total_cost_usd"] == 25.0
    db.close()


def test_cheapest_purchase_is_exact():
    import itertools

    from chemcost.pricing.pricing_db import cheapest_purchase

    sizes = [(0.1, 4.0), (0.25, 7.0), (1.0, 18.0), (5.0, 60.0)]
    packs = [{"quantity_g": g, "price_usd": p, "supplier": "S"} for g, p in sizes]
    for need in (0.05, 0.3, 0.6, 1.2, 2.6, 7.3):
        brute = min(sum(n * p["price_usd"] for n, p in zip(ns, packs))
                    for ns in itertools.product(range(0, 80), range(0, 32), range(0, 9), range(0, 3))
                    if sum(n * p["quantity_g"] for n, p in zip(ns, packs)) >= need - 1e-12)
        assert cheapest_purchase(need, packs)[0] == round(brute, 6), need


def test_procurement_ignores_normalized_price_without_pack_quotes(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="benzene", smiles="c1ccccc1", mw=78.11)
    db.set_normalized_price(chem_id, 20.0, "median", 1, ["TestSupplier"])

    result = db.get_procurement_price(required_mass_g=0.1, smiles="c1ccccc1")
    assert result["tier"] == "unpriced"
    assert result["total_cost_usd"] is None


def test_procurement_returns_unpriced_when_no_data(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    result = db.get_procurement_price(required_mass_g=0.1, smiles="c1ccccc1")
    assert result["tier"] == "unpriced"
    assert result["total_cost_usd"] is None


def test_purity_fraction_is_read_as_percent(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="propane", smiles="CCC", mw=44.1)
    for purity, price in [("0.97", 4.0), ("0.90", 1.0), ("0.0", 0.5)]:
        db.add_price(chem_id, PriceEntry(chemical_name="propane", cas=None, supplier=f"S{price}",
                                         quantity_g=1.0, price_usd=price,
                                         price_per_gram_usd=price, purity=purity))
    quotes = db.get_pack_quotes(smiles="CCC")
    assert [q["price_usd"] for q in quotes] == [4.0]


def test_commercial_grade_reagents_use_their_grade(tmp_path):
    from chemcost.pricing.pricing_db import grade_min_purity

    db = PricingDB(tmp_path / "g.sqlite")
    koh = db.add_chemical(name="potassium hydroxide", smiles="[K+].[OH-]", mw=56.106)
    naoh = db.add_chemical(name="sodium hydroxide", smiles="[Na+].[OH-]", mw=39.997)
    for cid in (koh, naoh):
        db.add_price(cid, PriceEntry(chemical_name="x", cas=None, supplier="S", quantity_g=25.0, price_usd=20.0,
                                     price_per_gram_usd=0.8, purity="85"))
    assert [q["purity"] for q in db.get_pack_quotes(smiles="[K+].[OH-]")] == ["85"]
    assert [q["purity"] for q in db.get_pack_quotes(smiles="O[K]")] == ["85"]  # same compound, other drawing
    assert db.get_pack_quotes(smiles="[Na+].[OH-]") == []  # NaOH is sold at >= 97%: no exception
    assert grade_min_purity("[K+].[OH-]", 95.0) == 85.0
    assert grade_min_purity("[K+].[OH-]", 80.0) == 80.0  # never stricter than asked


def test_inorganic_compound_found_in_ionic_or_covalent_drawing(tmp_path):
    db = PricingDB(tmp_path / "f.sqlite")
    lah = db.add_chemical(name="lithium aluminium hydride", smiles="[Li+].[AlH4-]", mw=37.95)
    assert db._lookup_chemical_id(smiles="[H-].[H-].[H-].[H-].[Li+].[Al+3]") == lah
    assert db._lookup_chemical_id(smiles="[Cl-]") is None  # a bare ion is not the neutral compound
