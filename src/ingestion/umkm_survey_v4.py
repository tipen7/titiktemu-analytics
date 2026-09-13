"""Loader for data/umkm_survey_v4.geojson -- a SEPARATE format/provenance
from the main umkm_survey_v3.geojson (see umkm_survey.py), so kept in its
own module rather than forced into clean_survey()'s v3-specific
text-parsing pipeline (v4's financial fields are already plain numeric
strings, not "13 juta/bulan"-style text).

Provenance, verified this session, not assumed:
- `jarak_ke_stasiun_meter` was found to not correspond to real geometry at
  all (one case was smaller than the geometrically-possible minimum
  distance) and was recomputed from each row's own real coordinates to
  whichever of several verified real reference points is nearest -- see
  scripts/fix_survey_v4_station_distances.py (the LRT Cibubur-Bogor
  corridor: Ciracas, Harjamukti, Bojong Nangka, Sentul, Baranangsiang) and
  scripts/fix_survey_v4_remaining_distances.py (the rest of the file's real
  Jabodetabek clusters: TMII, Depok-selatan, Depok-pusat, Bogor-utara,
  Blok M-selatan). ALL 280 rows now carry `nearest_station_verified`, set
  by one of those two passes -- this loader doesn't care which pass set it.
- `revenue_per_month` was found to be an EXACT, zero-variance
  `transaction_per_day * 30` for every row (confirmed mathematically, not
  a real independent figure) -- the surveyor's own explanation was that
  transaction_per_day is the real, directly-observed/reported quantity,
  and monthly revenue was never asked directly (a sensitive question) but
  approximated as day-rate x 30. That's a legitimate, disclosed proxy
  methodology, not fabrication -- but it means revenue here is NOT an
  independently-measured second quantity, and every row is flagged
  `revenue_is_estimated=True` accordingly (data_confidence discounted, same
  pattern as v3's `rent_price_annual_estimated` / `is_mock_rent` handling)
  rather than treated as equivalent to a directly-reported figure.
- `rent_price_annual` is reported as real, already an annual rupiah
  figure (no "X juta/Y hari" period string to parse, unlike v3).
- No column needed median/mode/mean imputation: the only universally-null
  field across all 280 rows is `rent_expiry_date` ("none" for every row),
  which isn't used anywhere in the model.
"""

import json

import geopandas as gpd
import pandas as pd

SURVEY_V4_PATH = "data/umkm_survey_v4.geojson"

# Real rent + a real-transaction-volume-derived revenue proxy is not the
# same confidence as v3's fully-real rows (both rent AND revenue directly
# computable from independently reported fields) -- discounted, not zeroed,
# since transaction_per_day and rent are both taken as directly real per
# the surveyor's account.
REVENUE_ESTIMATED_CONFIDENCE = 0.7


def load_v4_survey(extra_grid: gpd.GeoDataFrame, path: str = SURVEY_V4_PATH) -> pd.DataFrame:
    """Returns every row carrying `nearest_station_verified` (as of this
    module's docstring, all 280 -- see there for which correction pass set
    it), spatially joined to `extra_grid` -- the UNION of every non-main-
    study-area grid built by scripts/ingest_cibubur_bogor_corridor.py and
    scripts/ingest_other_jabodetabek_areas.py -- for the same FEATURE_COLS
    the main survey uses: poi_count, within_walk_isochrone, builtup_pct."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for feature in data["features"]:
        p = feature["properties"]
        if "nearest_station_verified" not in p:
            continue
        revenue = float(p["revenue_per_month"])
        rent = float(p["rent_price_annual"])
        rows.append({
            "id": p["id"],
            "name": p["name"],
            "latitude": float(p["latitude"]),
            "longitude": float(p["longitude"]),
            "tenant_type": p.get("tenant_type"),
            "transaction_per_buyer_idr": float(p["transaction_per_buyer"]) if p.get("transaction_per_buyer") else None,
            "dist_to_station": float(p["jarak_ke_stasiun_meter"]),
            "nearest_station": p["nearest_station_verified"],
            "revenue_per_month_idr": revenue,
            "rent_price_annual_idr": rent,
            "vulnerability": rent / (revenue * 12),
            "revenue_is_estimated": True,
            "data_confidence": REVENUE_ESTIMATED_CONFIDENCE,
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
    joined = joined[~joined.index.duplicated(keep="first")]  # a point exactly on a shared cell edge can double-match

    survey["grid_id"] = joined["grid_id"].values
    survey["district_name"] = joined["district_name"].values
    survey["kecamatan"] = joined["kecamatan"].values
    survey["poi_count"] = joined["poi_count"].values
    survey["within_walk_isochrone"] = joined["within_walk_isochrone"].values
    survey["builtup_pct"] = joined["builtup_pct"].values
    survey["x"] = points.geometry.x.values
    survey["y"] = points.geometry.y.values
    return survey
