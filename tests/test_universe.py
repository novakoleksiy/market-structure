import pytest

from universe import INSTRUMENT_SPECS, UNIVERSE, InstrumentSpec, Symbol, build_universe


def test_instrument_specs_are_unique_and_enabled_specs_are_resolved():
    assert len(INSTRUMENT_SPECS) == len(
        {spec.reference_name for spec in INSTRUMENT_SPECS}
    )

    enabled_specs = [spec for spec in INSTRUMENT_SPECS if spec.enabled]
    assert enabled_specs
    assert all(spec.provider_symbol for spec in enabled_specs)


def test_build_universe_returns_enabled_symbols_only():
    universe = build_universe()

    assert universe == UNIVERSE
    assert all(isinstance(symbol, Symbol) for symbol in universe)
    assert [symbol.name for symbol in universe] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "HYPEUSDT",
        "EUR_USD",
        "GBP_USD",
        "EUR_JPY",
        "GBP_JPY",
        "USD_CHF",
        "USD_CAD",
        "AUD_USD",
        "USD_JPY",
        "NZD_USD",
        "SPX500_USD",
        "NAS100_USD",
        "US30_USD",
        "DE30_EUR",
        "JP225_USD",
        "HK33_HKD",
        "UK100_GBP",
        "WTICO_USD",
        "XAU_USD",
        "XAG_USD",
    ]


def test_verified_index_mappings_are_enabled():
    mappings = {spec.reference_name: spec for spec in INSTRUMENT_SPECS}

    assert mappings["GER40"].provider_symbol == "DE30_EUR"
    assert mappings["GER40"].enabled
    assert mappings["HK50"].provider_symbol == "HK33_HKD"
    assert mappings["HK50"].enabled
    assert "Verified" in mappings["GER40"].note
    assert "Verified" in mappings["HK50"].note


def test_build_universe_rejects_enabled_spec_without_provider_symbol():
    with pytest.raises(ValueError, match="missing a provider symbol"):
        build_universe([InstrumentSpec("TEST", None, "oanda", "forex", enabled=True)])
