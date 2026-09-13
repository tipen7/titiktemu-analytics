"""Loads and cleans the UMKM survey CSV into a GWR-ready DataFrame.

Column names match the real uploaded survey file exactly:
id, latitude, longitude, name, description (produk),
tenant_type (umkm/franchise-tetap/seasonal), period (tahun/bulan/minggu),
transaction_per_day (ribu/juta/milliar), tenant_area (meters squared),
target_market (...), revenue_per_month, transaction_per_buyer, rent_trend,
rent_price_annual, rent_expiry_date

Per our earlier data-processing discussion, this module owns exactly the
cleaning rules we agreed on: currency/percent/period parsing, tenant_type
normalization, transaction_per_day splitting, rent formula resolution, and
a data_completeness_score -- nothing here silently invents values that
aren't derivable from what the row actually has.
"""

import re
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


TENANT_TYPE_MAP = {
    "umkm-tetap": "umkm_tetap",
    "umkm-seasonal": "umkm_seasonal",
    "franchise-tetap": "franchise_tetap",
    "franchise-seasonal": "franchise_seasonal",
}

UNIT_MULTIPLIERS = {"ribu": 1_000, "juta": 1_000_000, "jt": 1_000_000, "milliar": 1_000_000_000, "m": 1_000_000}
PERIOD_TO_MONTHS = {"tahun": 12.0, "bulan": 1.0, "minggu": 12.0 / 52.0}

RELEVANT_MODELING_COLS = [
    "tenant_type", "period_months", "rent_price_annual_idr",
    "revenue_per_month_idr", "txn_normal",
]


def fix_corrupted_coordinate(raw, integer_digits: int) -> float | None:
    """Fixes a specific, observed corruption pattern in the Sept 2026 survey
    export: decimal points were stripped from lat/long values, then the
    resulting integer got re-formatted with thousands-separator periods
    (e.g. real value -6.202065 -> stripped to 6202065 -> grouped as
    "6.202.065"). Reversing it: strip all '.'/',' separators, then reinsert
    the decimal point at a fixed position based on the known integer-digit
    count for this study area (1 digit for latitude near -6.x, 3 digits for
    longitude near 106.x).

    A second, simpler corruption also appears in the same file: Indonesian
    locale comma-as-decimal (e.g. "-6,202313") with no thousands grouping --
    detected and handled by a straight comma->period swap instead.

    NOTE: this is a targeted fix for THIS export's specific corruption, not
    a general-purpose coordinate parser. Verified against the actual file:
    reconstructs values consistent with neighboring rows' real coordinates
    for every row checked, including the worst-mangled ones (row 17, row 42).
    """
    if pd.isna(raw):
        return None
    s = str(raw).strip().strip('"')
    neg = s.startswith("-")
    s = s.lstrip("-")

    if "," in s and s.count(".") == 0:
        val = float(s.replace(",", "."))
    else:
        digits = re.sub(r"[.,]", "", s)
        if len(digits) <= integer_digits:
            return None
        val = float(digits[:integer_digits] + "." + digits[integer_digits:])

    return -val if neg else val


def load_raw_survey(csv_path: str) -> pd.DataFrame:
    """Reads .csv (the format used so far) or .geojson (in case a Geo MAPID
    Editor export is used instead -- per the earlier QGIS/Geo MAPID
    discussion). GeoJSON path: expects Point features with the same
    attribute names as the CSV columns; coordinates come from the
    geometry itself, not attribute lat/long fields."""
    path = str(csv_path)
    if path.lower().endswith((".geojson", ".json")):
        return _load_raw_survey_geojson(path)

    # "--" is this export's null sentinel, not pandas' default -- without
    # this, missing rent_price_annual cells silently pass .isna() checks
    # as if they had a real value (caught this by testing against the
    # actual file, not assumption).
    df = pd.read_csv(path, na_values=["--", "-", ""])
    # Drop the embedded column-guide row (id == "*unique ID") if present --
    # confirmed present in the originally uploaded file at row 1.
    df = df[pd.to_numeric(df["id"], errors="coerce").notna()].copy()
    df["id"] = df["id"].astype(int)

    # Fix the corrupted lat/long export -- see fix_corrupted_coordinate()
    # docstring. Applied unconditionally: verified harmless on
    # already-well-formed values, and every row in this export needed it.
    df["latitude"] = df["latitude"].apply(lambda v: fix_corrupted_coordinate(v, integer_digits=1))
    df["longitude"] = df["longitude"].apply(lambda v: fix_corrupted_coordinate(v, integer_digits=3))
    return df


def _load_raw_survey_geojson(path: str) -> pd.DataFrame:
    """GeoJSON reader for Geo MAPID Editor exports. NOT verified against a
    real MAPID export (none available) -- built and tested only against a
    synthetic GeoJSON matching the expected shape. Confirm column names
    match once a real export exists; MAPID's actual property names may
    differ from what's assumed here (id, name, and the same attribute
    columns as the CSV)."""
    gdf = gpd.read_file(path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df = df.replace(["--", "-", ""], np.nan)
    df["longitude"] = gdf.geometry.x
    df["latitude"] = gdf.geometry.y
    if "id" in df.columns:
        df = df[pd.to_numeric(df["id"], errors="coerce").notna()].copy()
        df["id"] = df["id"].astype(int)
    return df


PERIOD_TO_PER_YEAR = {"hari": 365.0, "bulan": 12.0, "tahun": 1.0}


def parse_rent_with_period(value) -> float | None:
    """Handles THIS export's rent format: '13 juta/bulan', '160 ribu/hari',
    '2 juta/5 hari', '100 juta/tahun', or a bare '50 juta' (no period ->
    treated as already-annual, matching the plain-value convention from
    the original survey file). Different format from resolve_rent_annual's
    '20% x revenue/day' formula handling -- that format doesn't appear in
    this file at all; kept as a fallback for forward-compatibility only."""
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if "/" not in s:
        return parse_currency(s)

    amount_part, period_part = s.split("/", 1)
    amount = parse_currency(amount_part.strip())
    if amount is None:
        return None

    period_part = period_part.strip()
    match = re.match(r"^(\d+)?\s*(hari|bulan|tahun)$", period_part)
    if not match:
        return None
    n, unit = match.groups()
    n = float(n) if n else 1.0
    per_year = PERIOD_TO_PER_YEAR[unit] / n
    return amount * per_year


def parse_currency(value) -> float | None:
    """'13jt' -> 13_000_000.0, '150juta' -> 150_000_000.0, '800 ribu' -> 800_000.0.
    Handles decimals ('1.5 juta') and missing space before the unit ('25ribu')."""
    if pd.isna(value):
        return None
    s = str(value).strip().lower().replace(",", ".")
    match = re.match(r"^([\d.]+)\s*(ribu|juta|jt|milliar)?$", s)
    if not match:
        return None
    number, unit = match.groups()
    try:
        number = float(number)
    except ValueError:
        return None
    multiplier = UNIT_MULTIPLIERS.get(unit, 1) if unit else 1
    return number * multiplier


def parse_percent(value) -> float | None:
    if pd.isna(value):
        return None
    s = str(value).strip().replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_period_months(value) -> float | None:
    """'1 tahun' -> 12.0, '9 bulan' -> 9.0, '1.5 tahun' -> 18.0."""
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    match = re.match(r"^([\d.]+)\s*(tahun|bulan|minggu)$", s)
    if not match:
        return None
    number, unit = match.groups()
    return float(number) * PERIOD_TO_MONTHS[unit]


def split_transaction_per_day(value) -> dict[str, float | None]:
    """Single value -> txn_normal only. 'A;B;C' -> high/normal/low, per the
    survey guide's 'high_traffic: normal: low_traffic' convention."""
    if pd.isna(value):
        return {"txn_high": None, "txn_normal": None, "txn_low": None}
    parts = [p.strip() for p in str(value).split(";")]
    if len(parts) == 1:
        return {"txn_high": None, "txn_normal": parse_currency(parts[0]), "txn_low": None}
    if len(parts) == 3:
        return {
            "txn_high": parse_currency(parts[0]),
            "txn_normal": parse_currency(parts[1]),
            "txn_low": parse_currency(parts[2]),
        }
    # Unexpected shape -- don't guess, surface it as missing rather than misassign.
    return {"txn_high": None, "txn_normal": None, "txn_low": None}


def normalize_tenant_type(value) -> str | None:
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    normalized = TENANT_TYPE_MAP.get(key)
    if normalized is None:
        print(f"WARNING: unrecognized tenant_type '{value}' -- left as null, not guessed.")
    return normalized


def resolve_rent_annual(rent_price_annual, revenue_per_month_idr) -> tuple[float | None, bool]:
    """Returns (value, is_estimated).
    - Period-annotated values ('13 juta/bulan', '160 ribu/hari') -> parsed
      and annualized via parse_rent_with_period(). This is THIS export's
      actual format, verified against the real file.
    - The older '20% x revenue/day' formula (from the original file, not
      present in this export) only resolves if revenue_per_month is present
      on the SAME row -- kept for forward-compatibility, never borrowed
      from another row.
    - Bare absolute values ('13jt') parse directly, treated as already-annual.
    Returns (None, False) if nothing applies."""
    if pd.isna(rent_price_annual):
        return None, False
    s = str(rent_price_annual).strip().lower()

    if "x revenue/day" in s or "x revenue" in s:
        pct_match = re.match(r"^([\d.]+)%", s)
        if not pct_match or revenue_per_month_idr is None or pd.isna(revenue_per_month_idr):
            return None, False
        pct = float(pct_match.group(1)) / 100.0
        daily_revenue = revenue_per_month_idr / 30.0
        annual = pct * daily_revenue * 365.0
        return annual, True

    if "/" in s:
        return parse_rent_with_period(s), False

    return parse_currency(rent_price_annual), False


def clean_survey(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["tenant_type"] = df["tenant_type (umkm/franchise-tetap/seasonal)"].apply(normalize_tenant_type)
    df["period_months"] = df["period (tahun/bulan/minggu)"].apply(parse_period_months)
    df["revenue_per_month_idr"] = df["revenue_per_month"].apply(parse_currency)
    df["transaction_per_buyer_idr"] = df["transaction_per_buyer"].apply(parse_currency)
    df["rent_trend_pct"] = df["rent_trend"].apply(parse_percent)

    txn_split = df["transaction_per_day (ribu/juta/milliar)"].apply(split_transaction_per_day).apply(pd.Series)
    df = pd.concat([df, txn_split], axis=1)

    resolved = df.apply(
        lambda r: resolve_rent_annual(r["rent_price_annual"], r["revenue_per_month_idr"]), axis=1
    )
    df["rent_price_annual_idr"] = resolved.apply(lambda t: t[0])
    df["rent_price_annual_estimated"] = resolved.apply(lambda t: t[1])

    df["data_completeness_score"] = df[RELEVANT_MODELING_COLS].notna().sum(axis=1) / len(RELEVANT_MODELING_COLS)

    # Unify with mock-data flags (scripts/generate_mock_survey.py's
    # is_mock_* columns) into ONE consistent confidence signal, instead of
    # two overlapping ones a dashboard would have to reconcile separately.
    # Gracefully no-ops on real (non-mock) input where these columns don't exist.
    # NOTE: only is_mock_rent and is_mock_full_row affect this -- they touch
    # RELEVANT_MODELING_COLS. is_mock_new_fields only covers auxiliary
    # columns (tanggal_survey etc.) that aren't used in GWR/XGBoost at all,
    # so it must NOT zero out confidence for rows whose actual modeling
    # data is real (caught this: it was zeroing confidence for every one of
    # rows 1-41, including rows with fully real rent/revenue data).
    modeling_relevant_mock_cols = [c for c in ["is_mock_rent", "is_mock_full_row"] if c in raw.columns]
    if modeling_relevant_mock_cols:
        is_full_mock = raw.get("is_mock_full_row", pd.Series(False, index=raw.index)).values
        is_rent_mock = raw.get("is_mock_rent", pd.Series(False, index=raw.index)).values
        df["data_confidence"] = df["data_completeness_score"]
        df.loc[is_full_mock, "data_confidence"] = 0.0
        df.loc[is_rent_mock & ~is_full_mock, "data_confidence"] = df["data_completeness_score"] * 0.5
    else:
        df["data_confidence"] = df["data_completeness_score"]

    keep_cols = [
        "id", "name", "latitude", "longitude", "tenant_type", "period_months",
        "txn_high", "txn_normal", "txn_low", "revenue_per_month_idr",
        "transaction_per_buyer_idr", "rent_trend_pct", "rent_price_annual_idr",
        "rent_price_annual_estimated", "data_completeness_score", "data_confidence",
    ]
    return df[keep_cols]


def _impute_series(s: pd.Series) -> pd.Series:
    """Picks median (numeric, skew-robust default), mean (numeric, only
    when explicitly requested as more appropriate), or mode (categorical) --
    per your instruction to 'pick scale that fits best for the field'.
    Numeric financial fields (rent, revenue, transaction values) use MEDIAN
    by default: these are right-skewed by nature (a few large values pull
    the mean up), so median is the more representative fill. Categorical
    fields use MODE."""
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(s.median())
    mode = s.mode()
    return s.fillna(mode.iloc[0] if not mode.empty else s)


def impute_missing(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """TODO 1e: fills missing values per-column using the rule in
    _impute_series. Operates on the CLEANED dataframe (after clean_survey),
    not the raw one -- imputing on already-parsed numeric columns, not on
    raw strings like '13jt'.

    IMPORTANT (per Job 1 above): with only 3/41 rows having real
    rent_price_annual/revenue_per_month values, imputing those two columns
    means ~93% of the resulting column is a repeated median of 3 points,
    not real data. This function does it because you explicitly asked for
    it with that tradeoff understood -- but the caller (run_pipeline.py)
    tags the output with `is_imputed` per-cell so nothing downstream can
    accidentally treat imputed values as equivalent to real survey data
    without knowing which is which.
    """
    df = df.copy()
    cols = columns or [
        "period_months", "txn_high", "txn_normal", "txn_low",
        "revenue_per_month_idr", "transaction_per_buyer_idr",
        "rent_trend_pct", "rent_price_annual_idr", "tenant_type",
    ]
    imputed_mask = pd.DataFrame(False, index=df.index, columns=cols)
    for col in cols:
        if col not in df.columns:
            continue
        imputed_mask[col] = df[col].isna()
        df[col] = _impute_series(df[col])

    df["n_fields_imputed"] = imputed_mask.sum(axis=1)
    df["data_completeness_score"] = 1.0  # honest only in the sense of "complete after imputation"
    return df


def compute_vulnerability_ratio(df: pd.DataFrame) -> pd.Series:
    """The GWR y-variable: annual rent burden vs. annual turnover. Only
    computable where BOTH rent_price_annual_idr and revenue_per_month_idr
    exist -- per our earlier discussion, this will be a small subset of
    rows, and that's the honest state of the data, not something to paper over."""
    annual_revenue = df["revenue_per_month_idr"] * 12
    ratio = df["rent_price_annual_idr"] / annual_revenue
    return ratio.where(df["rent_price_annual_idr"].notna() & df["revenue_per_month_idr"].notna())


def assign_grid_coords(df: pd.DataFrame, grid: gpd.GeoDataFrame, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Geocodes each UMKM point (WGS84 lat/long) into the grid's projected
    CRS, spatial-joins to grid_id, and adds x/y columns -- the shape
    src/modeling/gwr.py's fit_gwr() expects.

    Also computes `dist_to_station` PER POINT (not borrowed from the
    enclosing grid cell's centroid value) -- verified this matters: joining
    the grid cell's dist_to_station instead caused GWR's local design
    matrix to go singular whenever multiple survey points shared a grid
    cell (identical values -> collinearity within a kernel neighborhood).
    Each point's own precise distance avoids that entirely."""
    geocoded = df.dropna(subset=["latitude", "longitude"]).copy()
    points = gpd.GeoDataFrame(
        geocoded,
        geometry=[Point(xy) for xy in zip(geocoded["longitude"], geocoded["latitude"])],
        crs="EPSG:4326",
    ).to_crs(grid.crs)

    points["x"] = points.geometry.x
    points["y"] = points.geometry.y

    if stations is not None:
        stations_gdf = gpd.GeoDataFrame(
            stations, geometry=gpd.points_from_xy(stations["lon"], stations["lat"]), crs="EPSG:4326"
        ).to_crs(grid.crs)
        sx, sy = stations_gdf.geometry.x.values, stations_gdf.geometry.y.values
        dist_matrix = np.sqrt(
            (points["x"].values[:, None] - sx[None, :]) ** 2 + (points["y"].values[:, None] - sy[None, :]) ** 2
        )
        points["dist_to_station"] = dist_matrix.min(axis=1)

    joined = gpd.sjoin(points, grid[["grid_id", "geometry"]], how="left", predicate="within")
    # A point exactly on a shared cell edge/vertex, or inside TWO regions
    # whose bboxes happen to overlap (verified real case: blokmselatan's
    # bbox overlapped the main study area's by ~0.8km x 2.2km before that
    # was fixed), can match more than one grid_id -- keep one match per
    # survey point rather than silently duplicating that row's weight in
    # GWR training.
    joined = joined[~joined.index.duplicated(keep="first")]
    dropped = joined["grid_id"].isna().sum()
    if dropped:
        print(f"WARNING: {dropped} survey point(s) fall outside the study grid -- excluded from GWR training.")
    return joined.dropna(subset=["grid_id"]).drop(columns=["index_right"])
