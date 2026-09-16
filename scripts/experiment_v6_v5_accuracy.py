"""Read-only accuracy experiment: trains GWR + XGBoost EWS on real
umkm_survey_v6 (main study area, supersedes v3) + umkm_survey_v5
(corridor) data ONLY, evaluates the genuine LOOCV-against-real-survey
accuracy, and prints a report. v6 has 95/100 rows with a computable
vulnerability ratio (up from v3's 35/68), and population_density_per_km2
now has real coverage across 99% of the grid (up from ~65%, 0% for the
main study area) after data/demography/demography.csv was extended.

Deliberately does NOT call write_spatial_grids / write_risk_scores /
write_umkm_businesses / write_dashboard_metrics -- purely a local accuracy
experiment. Grid data is read back from spatial_grids (already refreshed
by scripts/refresh_demography.py) via load_grid_from_db, read-only.

Run: .venv/Scripts/python.exe -m scripts.experiment_v6_v5_accuracy
"""

import pandas as pd

from src.ingestion.umkm_survey import load_raw_survey, clean_survey, compute_vulnerability_ratio, assign_grid_coords
from src.ingestion.umkm_survey_v5 import load_v5_survey
from src.modeling.gwr import fit_gwr, predict_gwr_surface, loocv_predict
from src.modeling.xgboost_ews import (
    vulnerability_to_ews, train_xgboost_ews, validate_ews_against_survey, real_survey_ews_cutoffs,
)
from src.persistence.writer import get_engine, load_grid_from_db
from src.preprocessing.districts import STATIONS

SURVEY_V6_PATH = "data/umkm_survey_v6.geojson"

CORRIDOR_GRID_PREFIXES = [
    "cibubur_", "bojongnangka_", "sentul_", "baranangsiang_",
    "tmii_", "depokselatan_", "depokpusat_", "bogorutara_", "blokmselatan_",
]

BASE_FEATURE_COLS = ["dist_to_station", "poi_count", "within_walk_isochrone", "builtup_pct"]
WITH_DEMOGRAPHY_COLS = BASE_FEATURE_COLS + ["population_density_per_km2"]


def load_v6_survey(grid: pd.DataFrame) -> pd.DataFrame:
    raw = load_raw_survey(SURVEY_V6_PATH)
    clean = clean_survey(raw)
    clean["vulnerability"] = compute_vulnerability_ratio(clean)
    survey = assign_grid_coords(clean, grid, stations=STATIONS)
    survey = survey.merge(
        grid[["grid_id", "district_name", "kecamatan", "poi_count", "within_walk_isochrone",
              "builtup_pct", "population_density_per_km2"]],
        on="grid_id", how="left",
    )
    return survey


def run_experiment(feature_cols: list[str], survey: pd.DataFrame, combined_grid: pd.DataFrame, label: str) -> dict:
    print(f"\n=== {label} (features: {feature_cols}) ===")
    valid_survey = survey.dropna(subset=["vulnerability"] + feature_cols).reset_index(drop=True)
    print(f"    {len(valid_survey)} real survey rows have a computable vulnerability ratio AND every feature.")
    if len(valid_survey) < 10:
        print("    SKIPPED: fewer than 10 usable rows.")
        return {"label": label, "n": len(valid_survey), "accuracy_pct": None}

    fit = fit_gwr(survey, y_col="vulnerability", x_cols=feature_cols, verbose=True, bw_criterion="CV")
    if fit["dropped_x_cols"]:
        print(f"    GWR using {fit['used_x_cols']} (excluded {fit['dropped_x_cols']}).")

    print("    Running LOOCV against real survey ground truth...")
    loocv_preds = loocv_predict(survey, y_col="vulnerability", x_cols=feature_cols, bw_criterion="CV")
    ews_validation = validate_ews_against_survey(valid_survey["vulnerability"].values, loocv_preds)
    print(f"    REAL accuracy: {ews_validation['accuracy_pct']}% (n={ews_validation['n']}, "
          f"95% CI [{ews_validation['ci_95_low_pct']}, {ews_validation['ci_95_high_pct']}])")

    grid_scoreable = combined_grid.dropna(subset=feature_cols).copy()
    grid_scoreable["vulnerability_index"] = predict_gwr_surface(
        fit, list(zip(grid_scoreable.x, grid_scoreable.y)), grid_scoreable
    )
    ews_cutoffs = real_survey_ews_cutoffs(valid_survey["vulnerability"].values)
    grid_scoreable["ews_code"] = vulnerability_to_ews(grid_scoreable["vulnerability_index"], cutoffs=ews_cutoffs)
    try:
        ews_fit = train_xgboost_ews(grid_scoreable, feature_cols)
        print(f"    (diagnostic only) XGBoost surface-fit accuracy: {ews_fit['test_accuracy']:.3f}")
    except Exception as e:
        print(f"    (diagnostic only) XGBoost surface-fit skipped: {e}")

    return {"label": label, **ews_validation}


def run() -> None:
    engine = get_engine()

    print("[1/3] Reading already-persisted, demography-refreshed grids (read-only)...")
    main_grid = load_grid_from_db(engine, "grid_")
    corridor_grids = [load_grid_from_db(engine, prefix) for prefix in CORRIDOR_GRID_PREFIXES]
    combined_grid = pd.concat([main_grid] + corridor_grids, ignore_index=True)
    print(f"      {len(combined_grid)} cells loaded across {1 + len(corridor_grids)} regions.")

    print("[2/3] Loading real survey data: v6 (main study area) + v5 (corridor). v3/v4 EXCLUDED.")
    v6_survey = load_v6_survey(combined_grid)
    v5_survey = load_v5_survey(combined_grid)
    survey = pd.concat([v6_survey, v5_survey], ignore_index=True)
    print(f"      v6: {len(v6_survey)} rows, v5: {len(v5_survey)} rows -> {len(survey)} total geocoded rows.")
    print(f"      {survey['vulnerability'].notna().sum()} rows have a computable vulnerability ratio.")

    print("[3/3] Running experiments (CV bandwidth throughout)...")
    results = []
    results.append(run_experiment(BASE_FEATURE_COLS, survey, combined_grid, "Baseline 4 features"))
    results.append(run_experiment(WITH_DEMOGRAPHY_COLS, survey, combined_grid, "+ population_density_per_km2"))

    print("\n=== SUMMARY ===")
    for r in results:
        if r.get("accuracy_pct") is not None:
            print(f"  {r['label']}: {r['accuracy_pct']}% (n={r['n']}, CI [{r['ci_95_low_pct']}, {r['ci_95_high_pct']}])")
        else:
            print(f"  {r['label']}: could not be computed (n={r.get('n')})")


if __name__ == "__main__":
    run()
