"""Partition the monolithic cluster_points.parquet into per-city files and
pre-compute WGS84 lat/lon so the runtime never has to reproject or load the
full 39 MB point set into memory.

Input : data/processed/cluster_points.parquet  (X_Meters/Y_Meters, EPSG:25832)
Output: data/processed/points/City=<name>/data.parquet  (adds Lat/Lon)

The runtime backend reads exactly one city file per request via DuckDB's
hive-partitioning, instead of holding all 3.2M points in RAM.
"""
import logging
from pathlib import Path

import pandas as pd
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROCESSED_DIR = Path("data/processed")
POINTS_FILE = PROCESSED_DIR / "cluster_points.parquet"
OUTPUT_DIR = PROCESSED_DIR / "points"

SOURCE_CRS = "EPSG:25832"  # UTM 32N (easting/northing in metres)
TARGET_CRS = "EPSG:4326"   # WGS84 lat/lon


def add_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Reproject X_Meters/Y_Meters (EPSG:25832) to Lat/Lon (WGS84)."""
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    lon, lat = transformer.transform(df["X_Meters"].values, df["Y_Meters"].values)
    df = df.copy()
    df["Lat"] = lat
    df["Lon"] = lon
    return df


def write_partitions(points_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> int:
    """Reproject and write a points DataFrame as per-city Parquet partitions.

    Reusable by the offline processor so a full pipeline run produces the
    runtime layout directly. Returns the number of city partitions written.
    """
    df = add_latlon(points_df)

    # Sort so each city's file groups Mode/Cluster_ID together; this gives DuckDB
    # tight per-row-group statistics for predicate pushdown within a city file.
    df = df.sort_values(["City", "Mode", "Cluster_ID"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        output_dir,
        partition_cols=["City"],
        index=False,
        existing_data_behavior="delete_matching",
    )
    return sum(1 for _ in output_dir.glob("City=*"))


def main() -> None:
    if not POINTS_FILE.exists():
        raise FileNotFoundError(f"{POINTS_FILE} not found.")

    logging.info("Loading %s ...", POINTS_FILE)
    df = pd.read_parquet(POINTS_FILE)
    logging.info("Loaded %s points across %s cities.", len(df), df["City"].nunique())

    logging.info("Writing per-city partitions to %s ...", OUTPUT_DIR)
    n_parts = write_partitions(df, OUTPUT_DIR)
    logging.info("Done. Wrote %s city partitions.", n_parts)


if __name__ == "__main__":
    main()
