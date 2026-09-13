"""Single entry point for the offline analytics pipeline. Called by a
scheduled job (GitHub Actions cron, or manually) -- per our earlier
architecture decision, this is a batch script, not a hosted service.

Pipeline stages, per PRD 'Technology Architecture':
  1. Ingestion     -- OSM POI + walk isochrone (live), WorldCover built-up % (local raster);
                      MAPID still stubbed (pending real access -- see src/ingestion/mapid.py)
  2. Preprocessing -- grid construction + district tagging + feature extraction
  3. Modeling      -- GWR (Adaptive Bisquare) -> full-grid scoring -> XGBoost EWS/matching
  4. Narrative     -- Gemini synthesis per flagged grid cell
  5. Persistence   -- write scored grid + dashboard metrics to Supabase Postgres/PostGIS

Run with: python run_pipeline.py
"""

import sys
import pandas as pd
from src.config import settings
from src.preprocessing.grid import build_study_grid
from src.preprocessing.districts import tag_grid_with_district, tag_grid_with_kecamatan, STATIONS
from src.preprocessing.isochrone import tag_grid_within_isochrone
from src.ingestion.osm import fetch_poi_counts, fetch_walk_network
from src.ingestion.sentinel2 import add_builtup_pct
from src.ingestion.umkm_survey_v5 import load_v5_survey
from src.modeling.gwr import fit_gwr, predict_gwr_surface, loocv_predict
from src.modeling.xgboost_ews import vulnerability_to_ews, train_xgboost_ews, score_matching, validate_ews_against_survey, real_survey_ews_cutoffs
from src.narrative.gemini_client import generate_narrative, GeminiQuotaExceededError
from src.persistence.writer import get_engine, ensure_schema, write_spatial_grids, write_risk_scores, write_narratives, write_reallocation_candidates, load_grid_from_db, write_umkm_businesses, delete_umkm_businesses_by_source
from src.modeling.zones import precompute_reallocations
from src.persistence.dashboard_metrics import compute_dashboard_metrics, write_dashboard_metrics

SURVEY_CSV_PATH = "data/umkm_survey_v3.geojson"  # real Geo MAPID export, 68 rows (supersedes the mock CSV)
WORLDCOVER_RASTER_PATH = "data/worldcover/ESA_WorldCover_10m_2021_v200_S09E105_Map.tif"

# Real, wired-in modeling features -- keep GWR's x_cols and XGBoost's
# feature_cols identical so both models see the same signal. Extending
# this list requires all 3 of: load_umkm_survey()'s merge below, the
# ingestion stage that computes it, and this list itself.
FEATURE_COLS = ["dist_to_station", "poi_count", "within_walk_isochrone", "builtup_pct"]

# All non-main-study-area grids -- already ingested with real structural
# data (OSM POI/isochrone, WorldCover, demography) by
# scripts/ingest_cibubur_bogor_corridor.py (the LRT Cibubur-Bogor corridor)
# and scripts/ingest_other_jabodetabek_areas.py (the rest of
# umkm_survey_v5.geojson's real Jabodetabek clusters -- v5 is the refined
# re-survey of the same real areas v4 covered; v4 itself is EXCLUDED from
# training, see src/ingestion/umkm_survey_v5.py's docstring for why),
# then persisted to spatial_grids. Loaded back here (not re-fetched) and
# combined with the main study area grid so GWR fits ONE model across
# every real, geographically-separated region.
CORRIDOR_GRID_PREFIXES = [
    "cibubur_", "bojongnangka_", "sentul_", "baranangsiang_",
    "tmii_", "depokselatan_", "depokpusat_", "bogorutara_", "blokmselatan_",
]


def region_of(grid_id: str) -> str:
    """grid_id format is '{region}_{i:03d}_{j:03d})' -- strips the two
    trailing numeric segments to recover the region prefix (e.g.
    'cibubur_014_002' -> 'cibubur', 'grid_000_010' -> 'grid'). Used to keep
    reallocation search within one geographically contiguous region -- see
    run_pipeline.py's reallocation step."""
    return grid_id.rsplit("_", 2)[0]


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
    survey = survey.merge(
        grid[["grid_id", "district_name", "kecamatan", "poi_count", "within_walk_isochrone", "builtup_pct"]],
        on="grid_id", how="left",
    )
    return survey


def run() -> None:
    print(f"[1/5] Building study grid ({settings.grid_cell_size_m}m cells)...")
    grid = build_study_grid()
    grid["x"] = grid.geometry.centroid.x
    grid["y"] = grid.geometry.centroid.y
    grid = tag_grid_with_district(grid)
    grid = tag_grid_with_kecamatan(grid)
    print(f"      {len(grid)} cells built, tagged across {grid['district_name'].nunique()} district(s) "
          f"and {grid['kecamatan'].nunique()} real kecamatan.")

    print("[2/5] Ingestion (OSM POI, OSM walk isochrone, WorldCover built-up %)...")
    bbox = (
        settings.study_area_bbox_west, settings.study_area_bbox_south,
        settings.study_area_bbox_east, settings.study_area_bbox_north,
    )
    grid = fetch_poi_counts(grid, bbox_wgs84=bbox)
    print(f"      {int(grid['poi_count'].sum())} real OSM commercial POIs found "
          f"(mean {grid['poi_count'].mean():.1f}/cell).")

    walk_graph = fetch_walk_network(bbox)
    grid = tag_grid_within_isochrone(grid, walk_graph, STATIONS, max_distance_m=800.0)
    print(f"      {int(grid['within_walk_isochrone'].sum())}/{len(grid)} cells within an "
          f"800m/~10-min walk isochrone of a station ({len(walk_graph.nodes)} network nodes).")

    grid = add_builtup_pct(grid, WORLDCOVER_RASTER_PATH)
    print(f"      WorldCover built-up %% joined -- mean {grid['builtup_pct'].mean():.1f}%% built-up per cell.")
    # MAPID (Properti Go/Struk Go/Menu Go) still stubbed -- real access
    # confirmed reachable, but the actual dataset returned covers Kota
    # Tangerang / Bogor / Jakarta Timur / Depok, none of which overlap
    # this study area's bbox, so it's deferred until scope expands there.

    print("[2b/5] Loading LRT Cibubur-Bogor corridor grids (already ingested, real structural data)...")
    engine = get_engine()
    corridor_grids = [load_grid_from_db(engine, prefix) for prefix in CORRIDOR_GRID_PREFIXES]
    for prefix, cgrid in zip(CORRIDOR_GRID_PREFIXES, corridor_grids):
        print(f"      {prefix.rstrip('_')}: {len(cgrid)} cells")
    combined_grid = pd.concat([grid] + corridor_grids, ignore_index=True)
    print(f"      Combined grid: {len(combined_grid)} cells across {combined_grid['grid_id'].map(region_of).nunique()} regions.")

    print("[3/5] Loading UMKM survey data for GWR training (main study area + Cibubur-Bogor corridor)...")
    main_survey = load_umkm_survey(combined_grid)
    # Joined against the FULL combined_grid, not just corridor_grids --
    # verified some v5 rows (e.g. a Senayan City/Plaza Senayan cluster,
    # nearest_station_verified="blok_m_selatan") actually fall inside the
    # MAIN study area's own bounds, not any of the 9 extra regions. Joining
    # against corridor_grids alone silently left those rows' poi_count/
    # within_walk_isochrone/builtup_pct null.
    corridor_survey = load_v5_survey(combined_grid)
    survey = pd.concat([main_survey, corridor_survey], ignore_index=True)
    print(f"      {len(main_survey)} main-area + {len(corridor_survey)} v5 survey rows "
          f"= {len(survey)} total geocoded to the grid.")
    print(f"      {survey['vulnerability'].notna().sum()} rows have a computable vulnerability ratio.")

    # CV (not AICc) bandwidth selection: verified via LOOCV-against-real-
    # survey experiment (scripts/experiment_v3_v5_accuracy.py) to raise
    # genuine accuracy from 57.1% to 64.7% on this data -- CV directly
    # optimizes leave-one-out predictive accuracy, which is exactly what
    # ews_validation below measures, where AICc optimizes a different
    # (fit-vs-complexity) objective.
    try:
        fit = fit_gwr(survey, y_col="vulnerability", x_cols=FEATURE_COLS, bw_criterion="CV")
    except ValueError as e:
        print(f"      STOPPED: {e}")
        sys.exit(1)
    if fit["dropped_x_cols"]:
        print(f"      NOTE: GWR is using {fit['used_x_cols']} (excluded "
              f"{fit['dropped_x_cols']} -- constant across real survey rows, "
              f"see gwr.fit_gwr docstring). XGBoost still uses the full "
              f"{FEATURE_COLS}.")

    combined_grid["vulnerability_index"] = predict_gwr_surface(
        fit, list(zip(combined_grid.x, combined_grid.y)), combined_grid
    )

    # EWS cutoffs are anchored to REAL survey vulnerability, not the grid's
    # own distribution -- most cells sit far enough from any real training
    # point that GWR's extrapolated prediction clips to a shared floor (see
    # predict_gwr_surface's non-negativity clip), so the grid's own terciles
    # are neither stable nor meaningful (can even fail to have 3 distinct
    # values). "waspada"/"bahaya" should mean the same real rent-burden
    # level regardless of how much of the grid GWR can currently reach with
    # real signal -- see xgboost_ews.real_survey_ews_cutoffs.
    valid_survey = survey.dropna(subset=["vulnerability"] + FEATURE_COLS).reset_index(drop=True)
    ews_cutoffs = real_survey_ews_cutoffs(valid_survey["vulnerability"].values)
    combined_grid["ews_code"] = vulnerability_to_ews(combined_grid["vulnerability_index"], cutoffs=ews_cutoffs)
    print(f"      EWS cutoffs (real-survey-anchored): waspada >= {ews_cutoffs[0]:.3f}, "
          f"bahaya >= {ews_cutoffs[1]:.3f}")

    ews_fit = train_xgboost_ews(combined_grid, FEATURE_COLS)
    combined_grid["matching_score"] = score_matching(combined_grid, FEATURE_COLS, ews_fit)
    print(f"      GWR bandwidth={fit['bandwidth']}, XGBoost surface-fit acc={ews_fit['test_accuracy']:.3f} "
          f"(fidelity to the GWR surface, NOT real-world accuracy -- see ews_validation below)")

    print("      Running leave-one-out cross-validation against real survey ground truth "
          "(the genuine accuracy check)...")
    loocv_preds = loocv_predict(survey, y_col="vulnerability", x_cols=FEATURE_COLS, bw_criterion="CV")
    ews_validation = validate_ews_against_survey(valid_survey["vulnerability"].values, loocv_preds)
    print(f"      Real-ground-truth EWS accuracy: {ews_validation['accuracy_pct']}% "
          f"(n={ews_validation['n']}, 95% CI [{ews_validation['ci_95_low_pct']}, "
          f"{ews_validation['ci_95_high_pct']}], confidence={ews_validation['confidence_level']})")

    print("[4/5] Generating narratives for flagged (waspada/bahaya) grids...")
    flagged = combined_grid[combined_grid["ews_code"] > 0]
    narratives = []
    narrative_failures = 0
    for _, row in flagged.iterrows():
        try:
            result = generate_narrative(row["grid_id"], row["ews_code"], row["vulnerability_index"], row["matching_score"])
        except GeminiQuotaExceededError as e:
            # A 429 on ONE cell means every remaining cell this run will
            # also 429 (same key, same quota window) -- verified this cost
            # ~25 minutes retrying a quota already exhausted on the first
            # call, across 1,404 flagged cells. Stop immediately rather
            # than burn the rest of the list one failing HTTP round-trip
            # at a time; grid/scores/reallocations are already computed
            # and unaffected, only narratives for the remaining cells are
            # skipped.
            remaining = len(flagged) - len(narratives) - narrative_failures
            narrative_failures += remaining
            print(f"      WARNING: Gemini quota exceeded ({e}) -- stopping narrative generation early, "
                  f"skipping {remaining} remaining flagged cell(s) instead of retrying each individually.")
            break
        except RuntimeError as e:
            # One malformed/unreachable Gemini response shouldn't discard an
            # entire batch run's GWR/XGBoost output (grid/scores/reallocations
            # are independent of narratives) -- log it and keep going. The
            # malformed response itself is still never persisted, per
            # generate_narrative()'s own "don't guess" contract.
            narrative_failures += 1
            print(f"      WARNING: narrative failed for {row['grid_id']}, skipping ({e})")
            continue
        narratives.append({"grid_id": row["grid_id"], **result})
    print(f"      {len(narratives)}/{len(flagged)} narratives generated ({narrative_failures} failed and were skipped).")

    print("[5/5] Persisting to Postgres/PostGIS...")
    ensure_schema(engine)
    write_spatial_grids(engine, combined_grid)
    write_risk_scores(engine, combined_grid)
    write_narratives(engine, narratives)
    # Individual real business records -- same survey rows GWR trains on,
    # not a separate dataset. See writer.py's umkm_businesses schema note
    # for why this exists (Discovery Map favorites, UMKM Self-Tracker) and
    # what it deliberately excludes (raw rent/revenue). v4 is retired (its
    # revenue field was a mechanical proxy, not an independent
    # measurement -- see umkm_survey_v5.py) -- delete_umkm_businesses_by_source
    # removes its stale rows so the table doesn't keep showing since-closed/
    # duplicate businesses alongside the real v3+v5 records.
    delete_umkm_businesses_by_source(engine, "v4")
    write_umkm_businesses(engine, main_survey, source="v3")
    write_umkm_businesses(engine, corridor_survey, source="v5")
    print(f"      {len(main_survey.dropna(subset=['grid_id']))} + "
          f"{len(corridor_survey.dropna(subset=['grid_id']))} real UMKM business records persisted "
          f"(stale v4 records removed).")

    # Reallocation search is run PER REGION, not on the combined grid --
    # zones.py's expanding-radius search has no concept of "too far to be a
    # sensible recommendation," so searching across the full combined grid
    # could recommend a Dukuh Atas tenant relocate to Sentul, ~40km away,
    # once the search radius expands past everything nearby. Keeping it
    # region-scoped (same regions load_grid_from_db/CORRIDOR_GRID_PREFIXES
    # already partition by) preserves the existing "expand radius, fall
    # back to nearest-available" behavior within a geographically coherent
    # area, per region.
    reallocations = pd.concat(
        [precompute_reallocations(region_grid) for _, region_grid in combined_grid.groupby(combined_grid["grid_id"].map(region_of))],
        ignore_index=True,
    )
    write_reallocation_candidates(engine, reallocations)
    print(f"      {len(reallocations)} reallocation candidates precomputed for "
          f"{reallocations['origin_grid_id'].nunique() if not reallocations.empty else 0} danger cells "
          f"(searched within each of {combined_grid['grid_id'].map(region_of).nunique()} regions separately).")
    metrics = compute_dashboard_metrics(
        combined_grid, survey, ews_test_accuracy=ews_fit["test_accuracy"], ews_validation=ews_validation,
    )
    write_dashboard_metrics(engine, metrics)
    print(f"      Done. Dashboard summary: {metrics}")


if __name__ == "__main__":
    run()
