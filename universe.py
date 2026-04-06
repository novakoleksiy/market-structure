"""Universe configuration derived from instruments.txt reference names."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Symbol:
    name: str
    source: str  # "binance" or "oanda"
    source_param: str  # "futures-usdt" | "forex" | "indices" | "commodities"


@dataclass(frozen=True)
class InstrumentSpec:
    reference_name: str
    provider_symbol: str | None
    source: str
    source_param: str
    enabled: bool = True
    note: str = ""


INSTRUMENT_SPECS = [
    InstrumentSpec("BTCUSDT", "BTCUSDT", "binance", "futures-usdt"),
    InstrumentSpec("ETHUSDT", "ETHUSDT", "binance", "futures-usdt"),
    InstrumentSpec("SOLUSDT", "SOLUSDT", "binance", "futures-usdt"),
    InstrumentSpec("HYPEUSDT", "HYPEUSDT", "binance", "futures-usdt"),
    InstrumentSpec("EURUSD", "EUR_USD", "oanda", "forex"),
    InstrumentSpec("GBPUSD", "GBP_USD", "oanda", "forex"),
    InstrumentSpec("EURJPY", "EUR_JPY", "oanda", "forex"),
    InstrumentSpec("GBPJPY", "GBP_JPY", "oanda", "forex"),
    InstrumentSpec("USDCHF", "USD_CHF", "oanda", "forex"),
    InstrumentSpec("USDCAD", "USD_CAD", "oanda", "forex"),
    InstrumentSpec("AUDUSD", "AUD_USD", "oanda", "forex"),
    InstrumentSpec("USDJPY", "USD_JPY", "oanda", "forex"),
    InstrumentSpec("NZDUSD", "NZD_USD", "oanda", "forex"),
    InstrumentSpec("USD500", "SPX500_USD", "oanda", "indices"),
    InstrumentSpec("NAS100", "NAS100_USD", "oanda", "indices"),
    InstrumentSpec("US30", "US30_USD", "oanda", "indices"),
    InstrumentSpec(
        "GER40",
        "DE30_EUR",
        "oanda",
        "indices",
        note="Verified against OANDA via universe smoke test.",
    ),
    InstrumentSpec("JPN225", "JP225_USD", "oanda", "indices"),
    InstrumentSpec(
        "HK50",
        "HK33_HKD",
        "oanda",
        "indices",
        note="Verified against OANDA via universe smoke test.",
    ),
    InstrumentSpec(
        "UK100",
        "UK100_GBP",
        "oanda",
        "indices",
        note="OANDA quotes UK100 against GBP.",
    ),
    InstrumentSpec(
        "XTIUSD",
        "WTICO_USD",
        "oanda",
        "commodities",
        note="WTI crude oil CFD on OANDA.",
    ),
    InstrumentSpec("XAUUSD", "XAU_USD", "oanda", "commodities"),
    InstrumentSpec("XAGUSD", "XAG_USD", "oanda", "commodities"),
]


def build_universe(specs: list[InstrumentSpec] | None = None) -> list[Symbol]:
    """Return runtime symbols from enabled instrument specs."""
    specs = INSTRUMENT_SPECS if specs is None else specs

    universe: list[Symbol] = []
    for spec in specs:
        if not spec.enabled:
            continue
        if not spec.provider_symbol:
            raise ValueError(
                f"Enabled instrument '{spec.reference_name}' is missing a provider symbol"
            )
        universe.append(Symbol(spec.provider_symbol, spec.source, spec.source_param))
    return universe


UNIVERSE = build_universe()
