"""Single entry point for the offline analytics pipeline. Called by a
scheduled job (GitHub Actions cron, or manually) -- per our earlier
architecture decision, this is a batch script, not a hosted service.

Pipeline stages, per PRD 'Technology Architecture':
  1. Ingestion     -- MAPID API (stub, pending real access) + OSM (network-blocked here)
  2. Preprocessing -- grid construction + district tagging + feature extraction
  3. Modeling      -- GWR (Adaptive Bisquare) -> full-grid scoring -> XGBoost EWS/matching
  4. Narrative     -- Gemini synthesis per flagged grid cell
  5. Persistence   -- write scored grid + dashboard metrics to Supabase Postgres/PostGIS

Run with: python run_pipeline.py
"""

import sys
import numpy as np
import pandas as pd
from src.config import settings
from src.preprocessing.grid import build_study_grid
from src.preprocessing.districts import tag_grid_with_district, STATIONS
from src.modeling.gwr import fit_gwr, predict_gwr_surface
from src.modeling.xgboost_ews import vulnerability_to_ews, train_xgboost_ews, score_matching
from src.narrative.gemini_client import generate_narrative
from src.persistence.writer import get_engine, ensure_schema, write_spatial_grids, write_risk_scores, write_narratives, write_reallocation_candidates
from src.modeling.zones import precompute_reallocations
from src.persistence.dashboard_metrics import compute_dashboard_metrics, write_dashboard_metrics

SURVEY_CSV_PATH = "data/umkm_survey_mock.csv"  # swap to the real complete file when available


def load_umkm_survey(grid: pd.DataFrame) -> pd.DataFrame:
    """Loads, cleans, imputes, and geocodes the UMKM survey CSV into the
    shape src/modeling/gwr.py's fit_gwr() expects. Cleaning logic itself
    lives in src/ingestion/umkm_survey.py -- this only orchestrates."""
    from src.ingestion.umkm_survey import (
        load_raw_survey, clean_survey, compute_vulnerability_ratio, assign_grid_coords,
    )

    raw = load_raw_survey(SURVEY_CSV_PATH)
    clean = clean_survey(raw)
    clean["vulnerability"] = compute_vulnerability_ratio(clean)
    # dist_to_station computed PER POINT here (not grid-cell-borrowed) --
    # see assign_grid_coords docstring for why that matters (GWR singularity).
    survey = assign_grid_coords(clean, grid, stations=STATIONS)
    survey = survey.merge(grid[["grid_id", "poi_count"]], on="grid_id", how="left")
    return survey


def run() -> None:
    print(f"[1/5] Building study grid ({settings.grid_cell_size_m}m cells)...")
    grid = build_study_grid()
    grid["x"] = grid.geometry.centroid.x
    grid["y"] = grid.geometry.centroid.y
    grid = tag_grid_with_district(grid)
    print(f"      {len(grid)} cells built, tagged across {grid['district_name'].nunique()} district(s).")

    print("[2/5] Ingestion (OSM POI, Sentinel-2 NDBI, MAPID)...")
    print("      SKIPPED -- requires live network access not available in this "
          "environment. See src/ingestion/{osm,sentinel2,mapid}.py.")
    # Placeholder until real OSM ingestion runs -- flagged, not silently real.
    rng = np.random.default_rng(0)
    grid["poi_count"] = rng.integers(5, 60, len(grid))

    print("[3/5] Loading UMKM survey data for GWR training...")
    survey = load_umkm_survey(grid)
    print(f"      {len(survey)} survey rows geocoded to the grid.")
    print(f"      {survey['vulnerability'].notna().sum()} rows have a computable vulnerability ratio.")

    try:
        fit = fit_gwr(survey, y_col="vulnerability", x_cols=["dist_to_station", "poi_count"])
    except ValueError as e:
        print(f"      STOPPED: {e}")
        sys.exit(1)

    grid["vulnerability_index"] = predict_gwr_surface(
        fit, list(zip(grid.x, grid.y)), grid[["dist_to_station", "poi_count"]].values
    )
    grid["ews_code"] = vulnerability_to_ews(grid["vulnerability_index"])

    feature_cols = ["dist_to_station", "poi_count"]
    ews_fit = train_xgboost_ews(grid, feature_cols)
    grid["matching_score"] = score_matching(grid, feature_cols, ews_fit)
    print(f"      GWR bandwidth={fit['bandwidth']}, XGBoost test acc={ews_fit['test_accuracy']:.3f}")

    print("[4/5] Generating narratives for flagged (waspada/bahaya) grids...")
    flagged = grid[grid["ews_code"] > 0]
    narratives = []
    for _, row in flagged.iterrows():
        result = generate_narrative(row["grid_id"], row["ews_code"], row["vulnerability_index"], row["matching_score"])
        narratives.append({"grid_id": row["grid_id"], **result})
    print(f"      {len(narratives)} narratives generated.")

    print("[5/5] Persisting to Postgres/PostGIS...")
    engine = get_engine()
    ensure_schema(engine)
    write_spatial_grids(engine, grid)
    write_risk_scores(engine, grid)
    write_narratives(engine, narratives)
    reallocations = precompute_reallocations(grid)
    write_reallocation_candidates(engine, reallocations)
    print(f"      {len(reallocations)} reallocation candidates precomputed for "
          f"{reallocations['origin_grid_id'].nunique() if not reallocations.empty else 0} danger cells.")
    metrics = compute_dashboard_metrics(grid, survey)
    write_dashboard_metrics(engine, metrics)
    print(f"      Done. Dashboard summary: {metrics}")


if __name__ == "__main__":
    run()
