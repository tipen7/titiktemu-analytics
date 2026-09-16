"""Diagnostic: v6+v5's real accuracy (51.4%, n=179) came in well BELOW the
old v3+v5's 64.7% (n=119) -- isolates whether that's specifically caused
by v6's 32 new rows (previously-null fields now filled, but their
vulnerability values cluster tightly low -- 0.001-0.056 -- unlike v3's
wide 0.0008-1.01 spread) versus some other effect of the larger n.

Tests, all with CV bandwidth:
  A. v6 restricted to only the original 68 v3 IDs (i.e. v3's rows with
     their now-filled-in fields) + v5 -- isolates the effect of the 32
     NEW rows by excluding them.
  B. v5 only -- corridor signal alone.
  C. v6 only -- main study area signal alone (with its 32 new rows).

Run: .venv/Scripts/python.exe -m scripts.experiment_v6_diagnose
"""

import json

import pandas as pd

from src.ingestion.umkm_survey import load_raw_survey, clean_survey, compute_vulnerability_ratio, assign_grid_coords
from src.ingestion.umkm_survey_v5 import load_v5_survey
from src.modeling.gwr import fit_gwr, loocv_predict
from src.modeling.xgboost_ews import validate_ews_against_survey
from src.persistence.writer import get_engine, load_grid_from_db
from src.preprocessing.districts import STATIONS

SURVEY_V6_PATH = "data/umkm_survey_v6.geojson"
SURVEY_V3_PATH = "data/umkm_survey_v3.geojson"

CORRIDOR_GRID_PREFIXES = [
    "cibubur_", "bojongnangka_", "sentul_", "baranangsiang_",
    "tmii_", "depokselatan_", "depokpusat_", "bogorutara_", "blokmselatan_",
]
BASE_FEATURE_COLS = ["dist_to_station", "poi_count", "within_walk_isochrone", "builtup_pct"]


def load_v6_survey(grid: pd.DataFrame, restrict_to_v3_ids: bool = False) -> pd.DataFrame:
    raw = load_raw_survey(SURVEY_V6_PATH)
    if restrict_to_v3_ids:
        with open(SURVEY_V3_PATH, encoding="utf-8") as f:
            v3_ids = {f["properties"]["id"] for f in json.load(f)["features"]}
        raw = raw[raw["id"].astype(str).isin(v3_ids)].copy()
    clean = clean_survey(raw)
    clean["vulnerability"] = compute_vulnerability_ratio(clean)
    survey = assign_grid_coords(clean, grid, stations=STATIONS)
    survey = survey.merge(
        grid[["grid_id", "district_name", "kecamatan", "poi_count", "within_walk_isochrone", "builtup_pct"]],
        on="grid_id", how="left",
    )
    return survey


def run_variant(survey: pd.DataFrame, label: str) -> dict:
    print(f"\n=== {label} ===")
    valid = survey.dropna(subset=["vulnerability"] + BASE_FEATURE_COLS).reset_index(drop=True)
    print(f"    {len(valid)} usable rows.")
    if len(valid) < 10:
        print("    SKIPPED: too few rows.")
        return {"label": label, "accuracy_pct": None}
    fit = fit_gwr(survey, y_col="vulnerability", x_cols=BASE_FEATURE_COLS, verbose=False, bw_criterion="CV")
    loocv_preds = loocv_predict(survey, y_col="vulnerability", x_cols=BASE_FEATURE_COLS, bw_criterion="CV")
    ews_validation = validate_ews_against_survey(valid["vulnerability"].values, loocv_preds)
    print(f"    REAL accuracy: {ews_validation['accuracy_pct']}% (n={ews_validation['n']}, "
          f"95% CI [{ews_validation['ci_95_low_pct']}, {ews_validation['ci_95_high_pct']}], bw={fit['bandwidth']})")
    return {"label": label, **ews_validation}


def run() -> None:
    engine = get_engine()
    main_grid = load_grid_from_db(engine, "grid_")
    corridor_grids = [load_grid_from_db(engine, prefix) for prefix in CORRIDOR_GRID_PREFIXES]
    combined_grid = pd.concat([main_grid] + corridor_grids, ignore_index=True)

    v6_v3ids_survey = load_v6_survey(main_grid, restrict_to_v3_ids=True)
    v5_survey = load_v5_survey(combined_grid)
    v6_only_survey = load_v6_survey(main_grid, restrict_to_v3_ids=False)

    results = []
    results.append(run_variant(pd.concat([v6_v3ids_survey, v5_survey], ignore_index=True),
                                "A: v6 (v3 IDs only, fields refilled) + v5"))
    results.append(run_variant(v5_survey, "B: v5 only"))
    results.append(run_variant(v6_only_survey, "C: v6 only (all 100 rows)"))

    print("\n=== SUMMARY ===")
    for r in results:
        if r.get("accuracy_pct") is not None:
            print(f"  {r['label']}: {r['accuracy_pct']}% (n={r['n']}, CI [{r['ci_95_low_pct']}, {r['ci_95_high_pct']}])")
        else:
            print(f"  {r['label']}: could not be computed")


if __name__ == "__main__":
    run()
