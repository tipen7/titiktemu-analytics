"""Loader for data/umkm_survey_v5.geojson -- the refined replacement for
umkm_survey_v4.geojson's LRT Cibubur-Bogor + other-Jabodetabek corridor
rows. NOT a superset/subset relationship with v4: this is a re-survey (280
rows down to 149 -- the survey owner's own account is that the rest were
duplicates, since-closed businesses, or otherwise unreliable and were
dropped rather than kept), and it fixes v4's core data-quality problem
rather than inheriting it.

Provenance, verified this session, not assumed:
- v4's `revenue_per_month` was an exact `transaction_per_day * 30` for
  every row (a disclosed proxy, not an independent second measurement --
  see umkm_survey_v4.py's docstring). Checked here: v5 shows ZERO exact
  matches to that formula across the 23 rows where both fields are
  independently parseable, confirming revenue is a genuinely separate
  survey response this time, not a mechanical derivation of
  transaction_per_day. Per the survey owner, both rent and revenue are now
  directly reported -- so v5 rows get data_confidence based on field
  completeness (same convention as v3's real rows), NOT discounted the way
  v4's derived-revenue rows were.
- `jarak_ke_stasiun_meter` reproduced the exact same fabrication bug found
  in v4 (recomputing real haversine distance from each row's own lat/lon
  showed 137/149 rows off by >150m from the stated value, median ~891m,
  some of them literally the same business names as v4's bad rows) --
  fixed by scripts/fix_survey_v5_station_distances.py using the same
  verified reference-point list as v4's fix, since every v5 row's real
  coordinates fall inside one of the same 9 already-verified reference
  areas (checked: 0 unmatched rows, no new reference points needed).
- Financial fields are raw survey text ("6 juta", "50 ribu", "0%"), unlike
  v4's already-numeric strings -- reuses umkm_survey.py's v3-style text
  parsers (parse_currency, parse_period_months, normalize_tenant_type)
  rather than v4's simpler float() casts.
- `rent_price_annual (per tahun)` is already annual (per its own column
  label) -- parsed with parse_currency directly, no period-string ("X
  juta/Y hari") resolution needed, unlike v3's rent_price_annual. Missing
  for 44/149 rows (left null, not imputed -- vulnerability is only
  computed where both rent and revenue exist on the same row, same rule
  umkm_survey.py's compute_vulnerability_ratio uses).
"""

import json

import geopandas as gpd
import pandas as pd

from src.ingestion.umkm_survey import normalize_tenant_type, parse_currency, parse_period_months

SURVEY_V5_PATH = "data/umkm_survey_v5.geojson"


def load_v5_survey(extra_grid: gpd.GeoDataFrame, path: str = SURVEY_V5_PATH) -> pd.DataFrame:
    """Returns rows spatially joined to `extra_grid` (the same combined
    corridor grid umkm_survey_v4.load_v4_survey joins against) for the
    same FEATURE_COLS the main survey uses: poi_count,
    within_walk_isochrone, builtup_pct."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for feature in data["features"]:
        p = feature["properties"]
        revenue = parse_currency(p.get("revenue_per_month"))
        rent = parse_currency(p.get("rent_price_annual (per tahun)"))
        txn_buyer = parse_currency(p.get("transaction_per_buyer"))
        tenant_type = normalize_tenant_type(p.get("tenant_type"))
        period_months = parse_period_months(p.get("period"))

        fields_present = sum(
            v is not None for v in [revenue, rent, txn_buyer, tenant_type, period_months]
        )
        data_confidence = fields_present / 5.0

        rows.append({
            "id": p["id"],
            "name": p.get("name"),
            "latitude": float(p["latitude"]),
            "longitude": float(p["longitude"]),
            "tenant_type": tenant_type,
            "period_months": period_months,
            "transaction_per_buyer_idr": txn_buyer,
            "dist_to_station": float(p["jarak_ke_stasiun_meter"]),
            "nearest_station": p.get("nearest_station_verified"),
            "revenue_per_month_idr": revenue,
            "rent_price_annual_idr": rent,
            "vulnerability": (rent / (revenue * 12)) if (rent is not None and revenue) else None,
            "revenue_is_estimated": False,
            "data_confidence": data_confidence,
        })

    survey = pd.DataFrame(rows)
    points = gpd.GeoDataFrame(
        survey, geometry=gpd.points_from_xy(survey["longitude"], survey["latitude"]), crs="EPSG:4326",
    ).to_crs(extra_grid.crs)

    joined = gpd.sjoin(
        points,
        extra_grid[["grid_id", "district_name", "kecamatan", "poi_count", "within_walk_isochrone", "builtup_pct", "geometry"]],
        how="left", predicate="within",
    )
    joined = joined[~joined.index.duplicated(keep="first")]

    survey["grid_id"] = joined["grid_id"].values
    survey["district_name"] = joined["district_name"].values
    survey["kecamatan"] = joined["kecamatan"].values
    survey["poi_count"] = joined["poi_count"].values
    survey["within_walk_isochrone"] = joined["within_walk_isochrone"].values
    survey["builtup_pct"] = joined["builtup_pct"].values
    survey["x"] = points.geometry.x.values
    survey["y"] = points.geometry.y.values

    dropped = survey["grid_id"].isna().sum()
    if dropped:
        print(f"WARNING: {dropped} v5 survey point(s) fall outside the combined grid -- excluded from GWR training.")
    return survey.dropna(subset=["grid_id"]).reset_index(drop=True)
