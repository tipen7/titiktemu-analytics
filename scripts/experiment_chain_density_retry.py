"""Retries the 4 regions that failed with a network timeout in
scripts/experiment_chain_density.py's first run (depokselatan, depokpusat,
bogorutara, blokmselatan all hit `overpass-api.de` connect timeouts and
were filled with chain_poi_count=0/chain_ratio=0.0 -- a FAKE zero, not a
real "no chains here" finding), then re-runs the chain_ratio variant
(the better-performing of the two chain features) with clean data across
all 10 regions, to confirm whether the first run's below-baseline result
holds up once every region has a real fetch instead of a fallback.
"""

import time

import pandas as pd

from src.ingestion.osm import fetch_chain_poi_counts
from src.persistence.writer import get_engine, load_grid_from_db
from scripts.experiment_chain_density import (
    REGIONS, load_v3_survey, run_variant, BASE_FEATURE_COLS,
)
from src.ingestion.umkm_survey_v5 import load_v5_survey

RETRY_PREFIXES = {"depokselatan_", "depokpusat_", "bogorutara_", "blokmselatan_"}


def run() -> None:
    engine = get_engine()

    print("[1/3] Reading already-persisted grids + retrying the 4 failed regions...")
    chain_grids = []
    for prefix, bbox in REGIONS:
        g = load_grid_from_db(engine, prefix)
        if prefix in RETRY_PREFIXES:
            for attempt in range(3):
                try:
                    g = fetch_chain_poi_counts(g, bbox_wgs84=bbox)
                    print(f"      {prefix}: RETRY OK -- {int(g['chain_poi_count'].sum())} chain POIs "
                          f"(mean ratio {g['chain_ratio'].mean():.3f})")
                    break
                except Exception as e:
                    print(f"      {prefix}: retry {attempt+1}/3 failed ({e})")
                    time.sleep(10)
            else:
                print(f"      {prefix}: giving up after 3 retries -- filling 0")
                g["chain_poi_count"] = 0
                g["chain_ratio"] = 0.0
        else:
            g = fetch_chain_poi_counts(g, bbox_wgs84=bbox)
            print(f"      {prefix}: {int(g['chain_poi_count'].sum())} chain POIs (mean ratio {g['chain_ratio'].mean():.3f})")
        chain_grids.append(g)

    combined_grid = pd.concat(chain_grids, ignore_index=True)

    print("[2/3] Loading real survey data: v3 + v5 (v4 excluded)...")
    v3_survey = load_v3_survey(combined_grid)
    v5_survey = load_v5_survey(combined_grid)
    v5_survey = v5_survey.merge(
        combined_grid[["grid_id", "chain_poi_count", "chain_ratio"]], on="grid_id", how="left",
    )
    survey = pd.concat([v3_survey, v5_survey], ignore_index=True)
    print(f"      {len(survey)} total rows, {survey['vulnerability'].notna().sum()} with computable vulnerability.")

    print("[3/3] Re-running chain_ratio variant with clean data (CV bandwidth)...")
    result = run_variant(BASE_FEATURE_COLS + ["chain_ratio"], survey, "+ chain_ratio (clean retry)")
    print("\n=== FINAL ===")
    if result.get("accuracy_pct") is not None:
        print(f"  {result['label']}: {result['accuracy_pct']}% (n={result['n']}, "
              f"CI [{result['ci_95_low_pct']}, {result['ci_95_high_pct']}])")
        print("  Reference baseline (no chain feature): 64.7% (n=119, CI [55.8, 72.7])")


if __name__ == "__main__":
    run()
