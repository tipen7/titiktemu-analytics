"""Read-only accuracy experiment: measures whether real, structured
self-report submissions (via the new Self-Tracker intake form ->
POST /api/umkm-self-reports) would move genuine LOOCV-against-real-survey
accuracy, using a handful of MOCK self-report rows constructed in-memory.

IMPORTANT: the mock rows here are for testing the ingestion/accuracy path
ONLY -- they are never written to the real `umkm_self_reports` table or
any other database. Real self-reports (once an operator review flow marks
some 'exported') would be read via read_self_reports(); this script
substitutes a hand-built DataFrame with the same shape instead, so the
experiment can run without needing any real submissions to exist yet.

Mock rows are placed at real coordinates within already-ingested grid
regions (Blok M/Dukuh Atas and Cibubur), with realistic values matching
the pattern actually observed in real v3 data: independent tenants with
high rent-burden ratios, franchise/chain tenants with low ones (e.g. real
v3 rows showed 0.005-0.03 for chains like Auntie Anne's/Koi vs 0.5-1.0 for
independent stalls on the same street).

Run: .venv/Scripts/python.exe -m scripts.experiment_self_reports_accuracy
"""

import pandas as pd

from src.ingestion.umkm_survey import load_raw_survey, clean_survey, compute_vulnerability_ratio, assign_grid_coords
from src.ingestion.umkm_survey_v5 import load_v5_survey
from src.ingestion.umkm_self_reports import load_self_reports_survey
from src.modeling.gwr import fit_gwr, loocv_predict
from src.modeling.xgboost_ews import validate_ews_against_survey
from src.persistence.writer import get_engine, load_grid_from_db
from src.preprocessing.districts import STATIONS

SURVEY_V3_PATH = "data/umkm_survey_v3.geojson"
CORRIDOR_GRID_PREFIXES = [
    "cibubur_", "bojongnangka_", "sentul_", "baranangsiang_",
    "tmii_", "depokselatan_", "depokpusat_", "bogorutara_", "blokmselatan_",
]
FEATURE_COLS = ["dist_to_station", "poi_count", "within_walk_isochrone", "builtup_pct"]

# Real anchor coordinates (verified against already-ingested grid
# centroids), small realistic offsets applied per row so they land in
# different real cells rather than all stacking in one.
BLOK_M = (-6.243976, 106.799126)
DUKUH_ATAS = (-6.221333, 106.810343)
CIBUBUR = (-6.381364, 106.888638)

MOCK_SELF_REPORTS = [
    # (name, category, tenant_type, anchor, offset, rent_amount, rent_period, revenue_per_month)
    ("Warung Bu Siti", "kuliner", "umkm_tetap", BLOK_M, (0.0008, 0.0006), 15_000_000, "bulan", 20_000_000),
    ("McDougal's Express", "kuliner", "franchise_tetap", BLOK_M, (-0.0009, 0.0011), 5_000_000, "bulan", 200_000_000),
    ("Toko Kelontong Pak Budi", "dagang-retail", "umkm_tetap", DUKUH_ATAS, (0.0007, -0.0009), 8_000_000, "bulan", 15_000_000),
    ("Kopi Kenangan Corner", "kuliner", "franchise_tetap", DUKUH_ATAS, (-0.0006, 0.0008), 10_000_000, "bulan", 150_000_000),
    ("Bengkel Motor Jaya", "jasa", "umkm_tetap", BLOK_M, (0.0012, -0.0007), 6_000_000, "bulan", 12_000_000),
    ("Alfamart Express", "dagang-retail", "franchise_tetap", CIBUBUR, (0.0009, 0.0010), 8_000_000, "bulan", 180_000_000),
    ("Warteg Bahagia", "kuliner", "umkm_tetap", CIBUBUR, (-0.0011, -0.0008), 4_000_000, "bulan", 10_000_000),
    ("Indomaret Point", "dagang-retail", "franchise_tetap", CIBUBUR, (0.0006, -0.0012), 12_000_000, "bulan", 220_000_000),
]


def build_mock_self_reports_df() -> pd.DataFrame:
    rows = []
    for i, (name, category, tenant_type, (alat, alng), (dlat, dlng), rent, period, revenue) in enumerate(MOCK_SELF_REPORTS):
        rows.append({
            "id": f"mock-{i}",
            "business_name": name,
            "category": category,
            "tenant_type": tenant_type,
            "latitude": alat + dlat,
            "longitude": alng + dlng,
            "tenant_area_m2": 12.0,
            "rent_price_amount": float(rent),
            "rent_period_unit": period,
            "revenue_per_month_idr": float(revenue),
            "txn_high_idr": None,
            "txn_normal_idr": None,
            "txn_low_idr": None,
            "transaction_per_buyer_idr": 30_000.0,
            "rent_trend_pct": None,
            "status": "exported",
        })
    return pd.DataFrame(rows)


def load_v3_survey(grid: pd.DataFrame) -> pd.DataFrame:
    raw = load_raw_survey(SURVEY_V3_PATH)
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
    valid = survey.dropna(subset=["vulnerability"] + FEATURE_COLS).reset_index(drop=True)
    print(f"    {len(valid)} usable rows.")
    fit = fit_gwr(survey, y_col="vulnerability", x_cols=FEATURE_COLS, verbose=False, bw_criterion="CV")
    loocv_preds = loocv_predict(survey, y_col="vulnerability", x_cols=FEATURE_COLS, bw_criterion="CV")
    result = validate_ews_against_survey(valid["vulnerability"].values, loocv_preds)
    print(f"    REAL accuracy: {result['accuracy_pct']}% (n={result['n']}, "
          f"95% CI [{result['ci_95_low_pct']}, {result['ci_95_high_pct']}], bw={fit['bandwidth']})")
    return {"label": label, **result}


def run() -> None:
    engine = get_engine()
    print("[1/3] Reading already-persisted grids (read-only)...")
    main_grid = load_grid_from_db(engine, "grid_")
    corridor_grids = [load_grid_from_db(engine, p) for p in CORRIDOR_GRID_PREFIXES]
    combined_grid = pd.concat([main_grid] + corridor_grids, ignore_index=True)

    print("[2/3] Loading real v3+v5 survey (current production baseline) + 8 MOCK self-reports...")
    v3 = load_v3_survey(combined_grid)
    v5 = load_v5_survey(combined_grid)
    baseline_survey = pd.concat([v3, v5], ignore_index=True)

    mock_raw = build_mock_self_reports_df()
    mock_survey = load_self_reports_survey(combined_grid, mock_raw)
    print(f"    {len(mock_survey)}/{len(mock_raw)} mock self-reports geocoded to a real grid cell.")
    print(mock_survey[["id", "tenant_type", "category", "vulnerability", "grid_id"]].to_string())

    with_mock_survey = pd.concat([baseline_survey, mock_survey], ignore_index=True)

    print("[3/3] Comparing real accuracy, CV bandwidth, before/after mock self-reports...")
    baseline_result = run_variant(baseline_survey, "Baseline (v3+v5 only, current production)")
    with_mock_result = run_variant(with_mock_survey, "+ 8 mock self-reports")

    print("\n=== SUMMARY ===")
    print(f"  Baseline:        {baseline_result['accuracy_pct']}% (n={baseline_result['n']})")
    print(f"  + self-reports:  {with_mock_result['accuracy_pct']}% (n={with_mock_result['n']})")


if __name__ == "__main__":
    run()
