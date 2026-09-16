"""Refreshes population/population_density_per_km2 for every already-
persisted grid region using the newly-extended data/demography/demography.csv
(now covers 30 kecamatan across all 10 regions, including the main Dukuh
Atas/Blok M study area's own -- previously it had zero overlap there).

Deliberately does NOT re-fetch OSM POI/walk-isochrone or WorldCover --
those are unaffected by a demography.csv update, so re-running them would
just waste time re-fetching identical data. Reads each region's grid back
from spatial_grids (already has everything else), re-runs ONLY
join_demography() on it, and writes back via write_spatial_grids (which
upserts every column, so poi_count/dist_to_station/etc. round-trip
unchanged).

Run: .venv/Scripts/python.exe -m scripts.refresh_demography
"""

import pandas as pd

from src.ingestion.demography import join_demography
from src.persistence.writer import get_engine, ensure_schema, load_grid_from_db, write_spatial_grids

ALL_REGION_PREFIXES = [
    "grid_", "cibubur_", "bojongnangka_", "sentul_", "baranangsiang_",
    "tmii_", "depokselatan_", "depokpusat_", "bogorutara_", "blokmselatan_",
]


def run() -> None:
    engine = get_engine()
    ensure_schema(engine)

    total_cells = 0
    total_matched = 0
    for prefix in ALL_REGION_PREFIXES:
        grid = load_grid_from_db(engine, prefix)
        if grid.empty:
            print(f"{prefix}: no cells found, skipping")
            continue
        before_matched = grid["population_density_per_km2"].notna().sum()
        grid = join_demography(grid)
        after_matched = grid["population_density_per_km2"].notna().sum()
        write_spatial_grids(engine, grid)
        total_cells += len(grid)
        total_matched += after_matched
        print(f"{prefix:15s} {len(grid):5d} cells -- demography matched: {before_matched} -> {after_matched}")

    print(f"\nDone. {total_matched}/{total_cells} cells across all regions now have real demography "
          f"({total_matched / total_cells * 100:.1f}%).")


if __name__ == "__main__":
    run()
