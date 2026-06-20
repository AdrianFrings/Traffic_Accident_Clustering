"""Bake street-level names into clusters.parquet, offline and once.

Today the Streamlit app reverse-geocodes every visible cluster live, via
Nominatim with a 1 req/sec limit, on every session -> multi-second spinners.
This script does that geocoding a single time, ahead of serving, and persists
the result so the runtime serves a precomputed `StreetName` instantly.

It is resumable: every lookup is cached to disk (keyed by rounded lat/lon), so
re-runs and interruptions never repeat a request. Respects Nominatim's policy
(<= 1 req/sec, descriptive user-agent).

Usage:
    python src/enrich_street_names.py            # enrich all missing clusters
    python src/enrich_street_names.py --limit 5  # smoke-test on 5 clusters
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROCESSED_DIR = Path("data/processed")
CLUSTERS_FILE = PROCESSED_DIR / "clusters.parquet"
CACHE_FILE = PROCESSED_DIR / "street_name_cache.json"

USER_AGENT = "traffic_accident_clustering_dashboard"
CACHE_PRECISION = 5  # ~1 m; clusters are centroids so collisions are intended


def cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, CACHE_PRECISION)},{round(lon, CACHE_PRECISION)}"


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache))


def resolve_street(geocode, lat: float, lon: float) -> str:
    """Return the finest available place name for a coordinate."""
    try:
        loc = geocode((lat, lon), language="de", timeout=5)
        if loc and "address" in loc.raw:
            addr = loc.raw["address"]
            return addr.get("road") or addr.get("pedestrian") or addr.get("path") \
                or addr.get("suburb") or "Unknown Street"
    except Exception as exc:  # network/timeout/parse - cache nothing, retry next run
        logging.warning("Geocode failed for (%s, %s): %s", lat, lon, exc)
        raise
    return "Unknown Street"


def main(limit: int | None) -> None:
    if not CLUSTERS_FILE.exists():
        raise FileNotFoundError(f"{CLUSTERS_FILE} not found.")

    df = pd.read_parquet(CLUSTERS_FILE)
    if "StreetName" not in df.columns:
        df["StreetName"] = pd.NA

    cache = load_cache()
    geolocator = Nominatim(user_agent=USER_AGENT)
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1.0)

    todo = df[df["StreetName"].isna()]
    if limit is not None:
        todo = todo.head(limit)
    logging.info("Clusters needing a street name: %s (of %s total).", len(todo), len(df))

    processed = 0
    try:
        for idx, row in todo.iterrows():
            key = cache_key(row["Lat"], row["Lon"])
            if key not in cache:
                cache[key] = resolve_street(geocode, row["Lat"], row["Lon"])
                if processed % 50 == 0:
                    save_cache(cache)  # periodic checkpoint
            df.at[idx, "StreetName"] = cache[key]
            processed += 1
    finally:
        save_cache(cache)
        df.to_parquet(CLUSTERS_FILE, index=False)
        logging.info("Saved %s/%s street names; cache holds %s entries.",
                     processed, len(todo), len(cache))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process N clusters (smoke test).")
    main(parser.parse_args().limit)
