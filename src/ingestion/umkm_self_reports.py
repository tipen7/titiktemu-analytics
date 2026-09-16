"""Ingests real UMKM self-report submissions from titiktemu-backend's
`umkm_self_reports` table (see titiktemu-backend's
supabase/migrations/005_umkm_self_reports.sql / 007_umkm_self_report_category.sql
and the frontend's app/modules/umkm-self-tracker/self-tracker-form.tsx).

Unlike umkm_survey.py's v3/v6 loaders, there is NO free-text parsing here
at all: the self-report form collects rent_price_amount + rent_period_unit
as separate structured fields (not "13 juta/bulan" text), tenant_type as a
strict enum matching TENANT_TYPE_MAP's normalized output exactly, and
tenant_area_m2 as a plain number (not "34 m x 5 m"). That's the entire
point of building a structured intake form instead of relying on manual
survey transcription -- see titiktemu-analytics' TODO.md for the
data-quality problems this replaces (tenant_type mislabeling, coordinate
corruption, inconsistent rent formats).

Only 'exported' status rows are ingested by default -- an operator must
have reviewed a submission and marked it exported before it feeds the
model, the same trust boundary the batch pipeline already applies to
every other real data source (nothing here trusts unreviewed user input
directly). No operator review UI exists yet (see TODO.md); until it does,
`status` must be moved to 'exported' by hand (e.g. a direct SQL update)
for a submission to be included.
"""

import geopandas as gpd
import pandas as pd
from sqlalchemy import text

from src.ingestion.umkm_survey import PERIOD_TO_PER_YEAR, assign_grid_coords
from src.preprocessing.districts import STATIONS

SELF_REPORT_COLUMNS = [
    "id", "business_name", "category", "tenant_type", "latitude", "longitude",
    "tenant_area_m2", "rent_price_amount", "rent_period_unit",
    "revenue_per_month_idr", "txn_high_idr", "txn_normal_idr", "txn_low_idr",
    "transaction_per_buyer_idr", "rent_trend_pct", "status",
]


def read_self_reports(engine, statuses: tuple[str, ...] = ("exported",)) -> pd.DataFrame:
    """Reads real self-report rows from the shared database. Only rows
    whose `status` is in `statuses` are returned -- see module docstring
    for why 'pending'/'reviewed' rows are excluded by default."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text(f"""
                SELECT {", ".join(SELF_REPORT_COLUMNS)}
                FROM umkm_self_reports
                WHERE status = ANY(:statuses)
            """),
            conn, params={"statuses": list(statuses)},
        )
    return df


def clean_self_reports(raw: pd.DataFrame) -> pd.DataFrame:
    """Pure transformation, no DB/network access -- takes a DataFrame
    shaped like read_self_reports()'s output (or an equivalent
    hand-built/mock one for testing) and returns the same modeling-ready
    shape umkm_survey.py's clean_survey() produces: id, name, latitude,
    longitude, tenant_type, category, revenue_per_month_idr,
    rent_price_annual_idr, transaction_per_buyer_idr, vulnerability,
    data_confidence.

    rent_price_annual_idr is computed directly (amount * periods/year),
    not text-parsed -- rent_period_unit is already one of
    umkm_survey.PERIOD_TO_PER_YEAR's exact keys ('hari'/'bulan'/'tahun'),
    guaranteed by the form's own fixed dropdown, not free text.
    """
    df = raw.copy()

    def resolve_rent_annual(row):
        if pd.isna(row.get("rent_price_amount")) or pd.isna(row.get("rent_period_unit")):
            return None
        per_year = PERIOD_TO_PER_YEAR.get(row["rent_period_unit"])
        if per_year is None:
            return None
        return float(row["rent_price_amount"]) * per_year

    df["rent_price_annual_idr"] = df.apply(resolve_rent_annual, axis=1)

    annual_revenue = df["revenue_per_month_idr"] * 12
    df["vulnerability"] = (df["rent_price_annual_idr"] / annual_revenue).where(
        df["rent_price_annual_idr"].notna() & df["revenue_per_month_idr"].notna()
    )

    # Real completeness signal (not fabricated): counts how many of the
    # fields that actually matter for modeling/display are present on
    # this specific submission -- mirrors umkm_survey.py's
    # data_completeness_score philosophy for a differently-shaped input.
    completeness_cols = ["tenant_type", "category", "rent_price_annual_idr", "revenue_per_month_idr"]
    df["data_confidence"] = df[completeness_cols].notna().sum(axis=1) / len(completeness_cols)

    df["name"] = df["business_name"]
    keep_cols = [
        "id", "name", "latitude", "longitude", "tenant_type", "category",
        "revenue_per_month_idr", "rent_price_annual_idr",
        "transaction_per_buyer_idr", "vulnerability", "data_confidence",
    ]
    return df[keep_cols]


def load_self_reports_survey(grid: gpd.GeoDataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Cleans + geocodes real self-report rows onto `grid`, same shape
    run_pipeline.py's other survey loaders produce. `raw` is whatever
    read_self_reports() returned (or a mock DataFrame with the same
    columns, for testing without touching the database)."""
    if raw.empty:
        return clean_self_reports(raw).assign(grid_id=pd.Series(dtype=object))

    clean = clean_self_reports(raw)
    geocoded = assign_grid_coords(clean, grid, stations=STATIONS)
    geocoded = geocoded.merge(
        grid[["grid_id", "district_name", "kecamatan", "poi_count", "within_walk_isochrone", "builtup_pct"]],
        on="grid_id", how="left",
    )
    return geocoded
