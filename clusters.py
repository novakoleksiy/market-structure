"""Shared cluster configuration for multi-timeframe signal generation."""

_CLUSTER_1 = {"low": "5min", "med": "30min", "high": "4h"}
_CLUSTER_2 = {"low": "30min", "med": "4h", "high": "1D"}
_CLUSTER_3 = {"low": "4h", "med": "1D", "high": "1W"}
_CLUSTER_4 = {"low": "1D", "med": "1W", "high": "1ME"}

ALL_CLUSTERS = {
    "C1": _CLUSTER_1,
    "C2": _CLUSTER_2,
    "C3": _CLUSTER_3,
    "C4": _CLUSTER_4,
}

ALL_TIMEFRAMES = list({tf for c in ALL_CLUSTERS.values() for tf in c.values()})


def get_cluster(cluster_name: str) -> dict[str, str]:
    """Return cluster definition by name."""
    try:
        return ALL_CLUSTERS[cluster_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown cluster '{cluster_name}'. Choose from: {sorted(ALL_CLUSTERS)}"
        ) from exc
