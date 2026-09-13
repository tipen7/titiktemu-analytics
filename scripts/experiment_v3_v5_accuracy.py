"""Read-only accuracy experiment: trains GWR + XGBoost EWS on real
umkm_survey_v3 + umkm_survey_v5 data ONLY (v4 excluded per instruction --
its revenue field is a disclosed mechanical proxy, not an independent
second measurement, unlike v5's re-survey), evaluates the genuine
LOOCV-against-real-survey accuracy, and prints a report.

Deliberately does NOT call write_spatial_grids / write_risk_scores /
write_umkm_businesses / write_dashboard_metrics -- this is purely a local
accuracy experiment. Grid data is read back from spatial_grids (already
persisted by a prior real run) via load_grid_from_db, a read-only SELECT --
nothing in this script writes to the shared Supabase DB the backend/
frontend consume, per the explicit instruction not to wire this up until a
target accuracy is reached.

Run: .venv/Scripts/python.exe -m scripts.experiment_v3_v5_accuracy
"""

import numpy as np
import pandas as pd

from src.config import settings
from src.ingestion.umkm_survey import load_raw_survey, clean_survey, compute_vulnerability_ratio, assign_grid_coords
from src.ingestion.umkm_survey_v5 import load_v5_survey
from src.modeling.gwr import fit_gwr, predict_gwr_surface, loocv_predict
from src.modeling.xgboost_ews import (
    vulnerability_to_ews, train_xgboost_ews, score_matching,
    validate_ews_against_survey, real_survey_ews_cutoffs,
)
from src.persistence.writer import get_engine, load_grid_from_db
from src.preprocessing.districts import STATIONS

SURVEY_V3_PATH = "data/umkm_survey_v3.geojson"

CORRIDOR_GRID_PREFIXES = [
    "cibubur_", "bojongnangka_", "sentul_", "baranangsiang_",
    "tmii_", "depokselatan_", "depokpusat_", "bogorutara_", "blokmselatan_",
]

BASE_FEATURE_COLS = ["dist_to_station", "poi_count", "within_walk_isochrone", "builtup_pct"]


def load_v3_survey(grid: pd.DataFrame) -> pd.DataFrame:
    raw = load_raw_survey(SURVEY_V3_PATH)
    clean = clean_survey(raw)
    clean["vulnerability"] = compute_vulnerability_ratio(clean)
    survey = assign_grid_coords(clean, grid, stations=STATIONS)
    survey = survey.merge(
        grid[["grid_id", "district_name", "kecamatan", "poi_count", "within_walk_isochrone",
              "builtup_pct", "population_density_per_km2"]],
        on="grid_id", how="left",
    )
    return survey


def run_experiment(
    feature_cols: list[str], survey: pd.DataFrame, combined_grid: pd.DataFrame, label: str,
    bw_criterion: str = "AICc",
) -> dict:
    print(f"\n=== {label} (features: {feature_cols}, bw_criterion={bw_criterion}) ===")
    valid_survey = survey.dropna(subset=["vulnerability"] + feature_cols).reset_index(drop=True)
    print(f"    {len(valid_survey)} real survey rows have a computable vulnerability ratio "
          f"AND every feature in this set.")
    if len(valid_survey) < 10:
        print("    SKIPPED: fewer than 10 usable rows, GWR bandwidth selection unreliable.")
        return {"label": label, "n": len(valid_survey), "accuracy_pct": None}

    try:
        fit = fit_gwr(survey, y_col="vulnerability", x_cols=feature_cols, verbose=True, bw_criterion=bw_criterion)
    except ValueError as e:
        print(f"    STOPPED: {e}")
        return {"label": label, "n": len(valid_survey), "accuracy_pct": None}

    if fit["dropped_x_cols"]:
        print(f"    GWR using {fit['used_x_cols']} (excluded {fit['dropped_x_cols']} -- constant across real rows).")

    print("    Running LOOCV against real survey ground truth...")
    loocv_preds = loocv_predict(survey, y_col="vulnerability", x_cols=feature_cols, bw_criterion=bw_criterion)
    ews_validation = validate_ews_against_survey(valid_survey["vulnerability"].values, loocv_preds)
    print(f"    REAL accuracy: {ews_validation['accuracy_pct']}% (n={ews_validation['n']}, "
          f"95% CI [{ews_validation['ci_95_low_pct']}, {ews_validation['ci_95_high_pct']}], "
          f"confidence={ews_validation['confidence_level']})")

    # Surface-fit diagnostic (XGBoost approximating the GWR grid surface) --
    # reported for completeness/comparison with prior runs, NOT the real
    # accuracy metric, same caveat as run_pipeline.py's own framing.
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

    return {"label": label, **ews_validation, "bandwidth": fit["bandwidth"], "n_training_points": fit["n_points"]}


def run_log_transform_experiment(
    feature_cols: list[str], survey: pd.DataFrame, label: str, bw_criterion: str = "CV",
) -> dict:
    """Rent/revenue ratios are heavily right-skewed (0.004-1.01 in this
    data) -- GWR is a LOCAL LINEAR model, so fitting log(vulnerability)
    instead of the raw ratio is a standard variance-stabilizing transform
    for skewed positive ratios, not a data change. Predictions are
    exponentiated back to the original scale before bucketing into EWS
    classes, so this only affects the quality of the fit, not what
    "aman/waspada/bahaya" means."""
    print(f"\n=== {label} (features: {feature_cols}, bw_criterion={bw_criterion}) ===")
    survey = survey.copy()
    survey["log_vulnerability"] = np.log(survey["vulnerability"])

    valid = survey.dropna(subset=["log_vulnerability"] + feature_cols).reset_index(drop=True)
    print(f"    {len(valid)} real survey rows usable.")
    if len(valid) < 10:
        print("    SKIPPED: fewer than 10 usable rows.")
        return {"label": label, "n": len(valid), "accuracy_pct": None}

    n = len(valid)
    log_preds = np.full(n, np.nan)
    for i in range(n):
        train = valid.drop(index=i)
        try:
            fit = fit_gwr(
                train, y_col="log_vulnerability", x_cols=feature_cols, verbose=False, bw_criterion=bw_criterion,
            )
        except ValueError:
            continue
        held_out_coord = [(valid.loc[i, "x"], valid.loc[i, "y"])]
        held_out_X = valid.loc[[i]]
        log_preds[i] = predict_gwr_surface(fit, held_out_coord, held_out_X, clip_non_negative=False)[0]

    linear_preds = np.exp(log_preds)
    ews_validation = validate_ews_against_survey(valid["vulnerability"].values, linear_preds)
    print(f"    REAL accuracy: {ews_validation['accuracy_pct']}% (n={ews_validation['n']}, "
          f"95% CI [{ews_validation['ci_95_low_pct']}, {ews_validation['ci_95_high_pct']}], "
          f"confidence={ews_validation['confidence_level']})")
    return {"label": label, **ews_validation}


def run() -> None:
    engine = get_engine()

    print("[1/3] Reading already-persisted grids (read-only, no writes)...")
    main_grid = load_grid_from_db(engine, "grid_")
    corridor_grids = [load_grid_from_db(engine, prefix) for prefix in CORRIDOR_GRID_PREFIXES]
    combined_grid = pd.concat([main_grid] + corridor_grids, ignore_index=True)
    print(f"      {len(combined_grid)} cells loaded across {1 + len(corridor_grids)} regions.")

    print("[2/3] Loading real survey data: v3 (main study area) + v5 (corridor re-survey). v4 EXCLUDED.")
    v3_survey = load_v3_survey(combined_grid)
    v5_survey = load_v5_survey(combined_grid)
    survey = pd.concat([v3_survey, v5_survey], ignore_index=True)
    print(f"      v3: {len(v3_survey)} rows, v5: {len(v5_survey)} rows -> {len(survey)} total geocoded rows.")
    print(f"      {survey['vulnerability'].notna().sum()} rows have a computable vulnerability ratio "
          f"(both rent and revenue present).")

    print("[3/3] Running experiments...")
    results = []
    results.append(run_log_transform_experiment(BASE_FEATURE_COLS, survey, "log(vulnerability), CV bandwidth"))

    print("\n=== SUMMARY ===")
    for r in results:
        if r.get("accuracy_pct") is not None:
            print(f"  {r['label']}: {r['accuracy_pct']}% (n={r['n']}, CI [{r['ci_95_low_pct']}, {r['ci_95_high_pct']}])")
        else:
            print(f"  {r['label']}: could not be computed (n={r.get('n')})")


if __name__ == "__main__":
    run()
