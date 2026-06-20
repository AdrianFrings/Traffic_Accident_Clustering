"""DuckDB-backed query layer over the processor's Parquet output.

DuckDB is an *embedded* library (no server, no daemon, no idle RAM): it reads
the Parquet files directly and only materialises the rows a query asks for.
clusters.parquet (~1 MB) is cheap to scan; the 3.2M-row point set is never
loaded whole -- queries hit a single per-city partition produced by
src/partition_points.py.
"""
import json
import os
from pathlib import Path

import duckdb

# Data dir is resolved relative to the dashboard root, overridable for Docker.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DATA_DIR = Path(os.getenv("DATA_DIR", str(_DEFAULT_DATA_DIR)))

CLUSTERS_PARQUET = str(DATA_DIR / "clusters.parquet")
POINTS_GLOB = str(DATA_DIR / "points" / "*" / "*.parquet")

# UTYP1 accident-type codes -> human labels (mirrors the original dashboard).
ACCIDENT_TYPES = {
    "1": "Driving accident",
    "2": "Turning accident",
    "3": "Crossing / turning-in accident",
    "4": "Pedestrian crossing accident",
    "5": "Stationary vehicle accident",
    "6": "Accident between moving vehicles",
    "7": "Other accident",
}

# Padding (metres-ish, in degrees) added around a cluster's bounds when fetching
# nearby noise points for the zoom view, so we never read a city's full noise set.
_NOISE_PAD_DEG = 0.003  # ~330 m

# One shared connection; per-request work uses .cursor() for thread safety.
_con = duckdb.connect(database=":memory:")


def _cursor():
    return _con.cursor()


def _has_street_name() -> bool:
    """StreetName is added incrementally by the enrichment job; tolerate absence.

    Also tolerate the clusters file being missing at startup (e.g. before the
    data volume is mounted/populated) so the service can still boot.
    """
    try:
        cols = _cursor().execute(
            f"SELECT * FROM read_parquet('{CLUSTERS_PARQUET}') LIMIT 0"
        ).description
        return any(c[0] == "StreetName" for c in cols)
    except Exception:
        return False


_STREET_EXPR = (
    "COALESCE(NULLIF(StreetName, ''), LocationName)"
    if _has_street_name()
    else "LocationName"
)


def list_cities() -> list[str]:
    rows = _cursor().execute(
        f"SELECT DISTINCT City FROM read_parquet('{CLUSTERS_PARQUET}') ORDER BY City"
    ).fetchall()
    return [r[0] for r in rows]


def list_modes(city: str) -> list[str]:
    rows = _cursor().execute(
        f"SELECT DISTINCT Mode FROM read_parquet('{CLUSTERS_PARQUET}') "
        "WHERE City = ? ORDER BY Mode",
        [city],
    ).fetchall()
    return [r[0] for r in rows]


def _parse_type_stats(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        stats = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [
        {"code": code, "label": ACCIDENT_TYPES.get(str(code), f"Code {code}"), "count": count}
        for code, count in stats.items()
    ]


def get_clusters(city: str, mode: str) -> list[dict]:
    """Hotspot summaries for one city/mode, sorted worst-first."""
    rows = _cursor().execute(
        f"""
        SELECT Cluster_ID, Lat, Lon, AccidentCount, RiskScore,
               {_STREET_EXPR} AS StreetName, LocationName,
               AccidentTypeStats, GoogleMapsLink
        FROM read_parquet('{CLUSTERS_PARQUET}')
        WHERE City = ? AND Mode = ?
        ORDER BY AccidentCount DESC
        """,
        [city, mode],
    ).fetchall()
    return [
        {
            "cluster_id": cid,
            "lat": lat,
            "lon": lon,
            "accident_count": count,
            "risk_score": risk,
            "street_name": street,
            "location_name": location,
            "accident_types": _parse_type_stats(type_stats),
            "google_maps_link": gmaps,
        }
        for cid, lat, lon, count, risk, street, location, type_stats, gmaps in rows
    ]


def get_cluster_points(city: str, mode: str, cluster_id: int) -> dict:
    """Points for one cluster plus nearby noise, read from a single city partition.

    Noise is bounded to a padded box around the cluster so we never return a
    city's entire (often huge) noise set.
    """
    cur = _cursor()
    cluster_rows = cur.execute(
        f"""
        SELECT Lat, Lon FROM read_parquet('{POINTS_GLOB}', hive_partitioning = true)
        WHERE City = ? AND Mode = ? AND Cluster_ID = ?
        """,
        [city, mode, cluster_id],
    ).fetchall()
    cluster_points = [{"lat": lat, "lon": lon} for lat, lon in cluster_rows]

    noise_points: list[dict] = []
    if cluster_points:
        lats = [p["lat"] for p in cluster_points]
        lons = [p["lon"] for p in cluster_points]
        min_lat, max_lat = min(lats) - _NOISE_PAD_DEG, max(lats) + _NOISE_PAD_DEG
        min_lon, max_lon = min(lons) - _NOISE_PAD_DEG, max(lons) + _NOISE_PAD_DEG
        noise_rows = cur.execute(
            f"""
            SELECT Lat, Lon FROM read_parquet('{POINTS_GLOB}', hive_partitioning = true)
            WHERE City = ? AND Mode = ? AND Cluster_ID = -1
              AND Lat BETWEEN ? AND ? AND Lon BETWEEN ? AND ?
            """,
            [city, mode, min_lat, max_lat, min_lon, max_lon],
        ).fetchall()
        noise_points = [{"lat": lat, "lon": lon} for lat, lon in noise_rows]

    return {"cluster": cluster_points, "noise": noise_points}
