"""Read-only experiment: fetches a real OSM chain/franchise-presence
feature (see src/ingestion/osm.py's fetch_chain_poi_counts) for all 10
already-persisted grid regions, merges it in-memory only, and re-runs the
v3+v5 accuracy experiment to see whether it improves on the current best
(64.7%, CV-selected GWR bandwidth, the 4 existing structural features --
see scripts/experiment_v3_v5_accuracy.py).

Does NOT write anything to spatial_grids -- this is purely to decide
whether the feature is worth persisting at all before touching the schema
or the production pipeline.

Run: .venv/Scripts/python.exe -m scripts.experiment_chain_density
"""

import pandas as pd

from src.config import settings
from src.ingestion.osm import fetch_chain_poi_counts
from src.ingestion.umkm_survey import load_raw_survey, clean_survey, compute_vulnerability_ratio, assign_grid_coords
from src.ingestion.umkm_survey_v5 import load_v5_survey
from src.modeling.gwr import fit_gwr, predict_gwr_surface, loocv_predict
from src.modeling.xgboost_ews import validate_ews_against_survey
from src.persistence.writer import get_engine, load_grid_from_db
from src.preprocessing.districts import STATIONS, CIBUBUR_BOGOR_BBOX

SURVEY_V3_PATH = "data/umkm_survey_v3.geojson"
BASE_FEATURE_COLS = ["dist_to_station", "poi_count", "within_walk_isochrone", "builtup_pct"]

# (grid_id_prefix, real bbox) for every region already persisted -- same
# bboxes as run_pipeline.py's main study area + the 9 corridor ingest scripts.
REGIONS = [
    ("grid_", (settings.study_area_bbox_west, settings.study_area_bbox_south,
               settings.study_area_bbox_east, settings.study_area_bbox_north)),
    ("cibubur_", CIBUBUR_BOGOR_BBOX),
    ("bojongnangka_", (106.885, -6.4500, 106.9450, -6.4000)),
    ("sentul_", (106.8450, -6.5450, 106.8650, -6.5150)),
    ("baranangsiang_", (106.8000, -6.6150, 106.8200, -6.5950)),
    ("tmii_", (106.870, -6.300, 106.890, -6.280)),
    ("depokselatan_", (106.790, -6.460, 106.815, -6.430)),
    ("depokpusat_", (106.810, -6.400, 106.840, -6.370)),
    ("bogorutara_", (106.820, -6.600, 106.850, -6.560)),
    ("blokmselatan_", (106.785, -6.260, 106.805, -6.246)),
]


def load_v3_survey(grid: pd.DataFrame) -> pd.DataFrame:
    raw = load_raw_survey(SURVEY_V3_PATH)
    clean = clean_survey(raw)
    clean["vulnerability"] = compute_vulnerability_ratio(clean)
    survey = assign_grid_coords(clean, grid, stations=STATIONS)
    survey = survey.merge(
        grid[["grid_id", "district_name", "kecamatan", "poi_count", "within_walk_isochrone",
              "builtup_pct", "chain_poi_count", "chain_ratio"]],
        on="grid_id", how="left",
    )
    return survey


def run_variant(feature_cols: list[str], survey: pd.DataFrame, label: str) -> dict:
    print(f"\n=== {label} (features: {feature_cols}) ===")
    valid = survey.dropna(subset=["vulnerability"] + feature_cols).reset_index(drop=True)
    print(f"    {len(valid)} usable rows.")
    if len(valid) < 10:
        print("    SKIPPED: too few rows.")
        return {"label": label, "accuracy_pct": None}

    fit = fit_gwr(survey, y_col="vulnerability", x_cols=feature_cols, verbose=True, bw_criterion="CV")
    if fit["dropped_x_cols"]:
        print(f"    GWR using {fit['used_x_cols']} (excluded {fit['dropped_x_cols']}).")
    loocv_preds = loocv_predict(survey, y_col="vulnerability", x_cols=feature_cols, bw_criterion="CV")
    ews_validation = validate_ews_against_survey(valid["vulnerability"].values, loocv_preds)
    print(f"    REAL accuracy: {ews_validation['accuracy_pct']}% (n={ews_validation['n']}, "
          f"95% CI [{ews_validation['ci_95_low_pct']}, {ews_validation['ci_95_high_pct']}])")
    return {"label": label, **ews_validation}


def run() -> None:
    engine = get_engine()

    print("[1/4] Reading already-persisted grids (read-only)...")
    grids = {prefix: load_grid_from_db(engine, prefix) for prefix, _ in REGIONS}
    for prefix, g in grids.items():
        print(f"      {prefix}: {len(g)} cells")

    print("[2/4] Fetching real OSM chain/franchise POI data per region (live network)...")
    chain_grids = []
    for prefix, bbox in REGIONS:
        g = grids[prefix]
        try:
            g_chain = fetch_chain_poi_counts(g, bbox_wgs84=bbox)
        except Exception as e:
            print(f"      {prefix}: FAILED ({e}) -- filling chain_poi_count=0, chain_ratio=0.0")
            g_chain = g.copy()
            g_chain["chain_poi_count"] = 0
            g_chain["chain_ratio"] = 0.0
        n_chains = int(g_chain["chain_poi_count"].sum())
        print(f"      {prefix}: {n_chains} real OSM chain/franchise POIs found "
              f"(mean chain_ratio {g_chain['chain_ratio'].mean():.3f})")
        chain_grids.append(g_chain)
    combined_grid = pd.concat(chain_grids, ignore_index=True)

    print("[3/4] Loading real survey data: v3 + v5 (v4 excluded)...")
    v3_survey = load_v3_survey(combined_grid)
    v5_survey = load_v5_survey(combined_grid)
    v5_survey = v5_survey.merge(
        combined_grid[["grid_id", "chain_poi_count", "chain_ratio"]], on="grid_id", how="left",
    )
    survey = pd.concat([v3_survey, v5_survey], ignore_index=True)
    print(f"      {len(survey)} total rows, {survey['vulnerability'].notna().sum()} with computable vulnerability.")

    print("[4/4] Running experiments (CV bandwidth throughout)...")
    results = []
    results.append(run_variant(BASE_FEATURE_COLS, survey, "Baseline 4 features (reference: 64.7%)"))
    results.append(run_variant(BASE_FEATURE_COLS + ["chain_poi_count"], survey, "+ chain_poi_count"))
    results.append(run_variant(BASE_FEATURE_COLS + ["chain_ratio"], survey, "+ chain_ratio"))

    print("\n=== SUMMARY ===")
    for r in results:
        if r.get("accuracy_pct") is not None:
            print(f"  {r['label']}: {r['accuracy_pct']}% (n={r['n']}, CI [{r['ci_95_low_pct']}, {r['ci_95_high_pct']}])")
        else:
            print(f"  {r['label']}: could not be computed")


if __name__ == "__main__":
    run()
