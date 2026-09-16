"""Writes pipeline results to the shared Supabase Postgres+PostGIS instance.
Table shapes match the PRD's 'Database & Spatial Storage Layer' section --
spatial_grids, gentrification_risk_scores. Schema ownership lives in
titiktemu-backend's supabase/migrations/ (see our earlier repo-topology
decision) -- this module only READS that contract, it doesn't define it.

Geometry is written in the internal projected CRS (EPSG:32748) per the PRD;
ST_Transform to EPSG:4326 happens on read, in the backend, not here."""

import geopandas as gpd
import pandas as pd
from shapely import wkt
from sqlalchemy import create_engine, text
from src.config import settings


def get_engine():
    return create_engine(settings.database_url)


def load_grid_from_db(engine, grid_id_prefix: str) -> gpd.GeoDataFrame:
    """Reads back an already-persisted grid (e.g. the Cibubur-Bogor
    corridor's real structural data, see
    scripts/ingest_cibubur_bogor_corridor.py) instead of re-running live
    OSM/WorldCover ingestion for a region that's already been fetched.
    grid_id_prefix filters to one region's cells (e.g. 'cibubur_')."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT grid_id, ST_AsText(geom) AS wkt, poi_count, dist_to_station_m AS dist_to_station,
                       ndbi_mean, district_name, kecamatan, within_walk_isochrone, builtup_pct,
                       population, population_density_per_km2
                FROM spatial_grids WHERE grid_id LIKE :pattern
            """),
            conn, params={"pattern": f"{grid_id_prefix}%"},
        )
    df["geometry"] = df["wkt"].apply(wkt.loads)
    grid = gpd.GeoDataFrame(df.drop(columns=["wkt"]), geometry="geometry", crs=settings.projected_crs)
    grid["x"] = grid.geometry.centroid.x
    grid["y"] = grid.geometry.centroid.y
    return grid


def ensure_schema(engine) -> None:
    """Creates the tables this pipeline writes to, IF they don't already
    exist. In a real deployment the backend's Supabase migrations are the
    source of truth -- this is a convenience for local pipeline development
    and testing, not a replacement for those migrations."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS spatial_grids (
                grid_id             TEXT PRIMARY KEY,
                geom                GEOMETRY(Polygon, 32748),
                poi_count           INTEGER,
                dist_to_station_m   NUMERIC,
                ndbi_mean           NUMERIC,
                district_name       TEXT
            );
        """))
        # Idempotent -- older DBs created before district tagging existed
        # won't have this column; add it without erroring on re-runs.
        conn.execute(text(
            "ALTER TABLE spatial_grids ADD COLUMN IF NOT EXISTS district_name TEXT;"
        ))
        # Same idempotent pattern -- added once real OSM isochrone + real
        # WorldCover built-up % were wired in as modeling features.
        # builtup_pct is a distinct column from ndbi_mean (0-100% vs a
        # -1..1 NDBI value) even though add_builtup_pct()'s docstring
        # treats them as conceptually interchangeable -- reusing ndbi_mean
        # for a differently-scaled value would be misleading to anyone
        # querying this table directly.
        conn.execute(text(
            "ALTER TABLE spatial_grids ADD COLUMN IF NOT EXISTS within_walk_isochrone SMALLINT;"
        ))
        conn.execute(text(
            "ALTER TABLE spatial_grids ADD COLUMN IF NOT EXISTS builtup_pct NUMERIC;"
        ))
        # Real government kecamatan (BPS/Dukcapil boundaries), distinct from
        # district_name (TOD-station catchment) -- see
        # src/preprocessing/districts.py tag_grid_with_kecamatan() docstring.
        conn.execute(text(
            "ALTER TABLE spatial_grids ADD COLUMN IF NOT EXISTS kecamatan TEXT;"
        ))
        # Real BPS/Dukcapil population figures, kecamatan-level (join key is
        # `kecamatan` above) -- see scripts/ingest_cibubur_bogor_corridor.py.
        # Only populated for corridors whose kecamatan match
        # data/demography/demography.csv's coverage (currently: parts of
        # Jakarta Timur, Bogor, Depok) -- NULL elsewhere, not backfilled.
        conn.execute(text(
            "ALTER TABLE spatial_grids ADD COLUMN IF NOT EXISTS population INTEGER;"
        ))
        conn.execute(text(
            "ALTER TABLE spatial_grids ADD COLUMN IF NOT EXISTS population_density_per_km2 NUMERIC;"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_spatial_grids_geom ON spatial_grids USING GIST (geom);"
        ))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gentrification_risk_scores (
                grid_id             TEXT PRIMARY KEY REFERENCES spatial_grids(grid_id),
                vulnerability_index NUMERIC,
                ews_code            SMALLINT,
                matching_score      NUMERIC,
                computed_at         TIMESTAMPTZ DEFAULT now()
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS policy_recommendations (
                grid_id              TEXT PRIMARY KEY REFERENCES spatial_grids(grid_id),
                narrative            TEXT,
                recommendation_type  TEXT,
                ai_generated         BOOLEAN DEFAULT true,
                requires_human_review BOOLEAN DEFAULT true,
                generated_at         TIMESTAMPTZ DEFAULT now()
            );
        """))
        # Batch-precomputed reallocation candidates -- per the batch-vs-live
        # architecture decision: the backend's 'live' reallocation lookup is
        # just `SELECT ... WHERE origin_grid_id = :tenant_grid_id ORDER BY
        # rank`, no Python computation at request time.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reallocation_candidates (
                origin_grid_id       TEXT REFERENCES spatial_grids(grid_id),
                rank                 SMALLINT,
                recommended_grid_id  TEXT REFERENCES spatial_grids(grid_id),
                recommended_district TEXT,
                distance_m           NUMERIC,
                matching_score       NUMERIC,
                crossed_district     BOOLEAN,
                search_radius_used_m NUMERIC,
                expansions_needed    SMALLINT,
                computed_at          TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (origin_grid_id, rank)
            );
        """))
        # Individual real UMKM business records -- from the same survey rows
        # GWR trains on (umkm_survey_v3.geojson / umkm_survey_v4.geojson),
        # NOT a separate/fabricated dataset. Added because the frontend's
        # Discovery Map favorites list and UMKM Self-Tracker table both need
        # per-business records (name, category, location) that no existing
        # table provides -- spatial_grids/gentrification_risk_scores are
        # grid-cell aggregates, never individual businesses. Deliberately
        # does NOT expose raw rent/revenue (private, and for v4 rows,
        # methodology-caveated -- see umkm_survey_v4.py) -- only a single
        # reference transaction value, real fields, and the grid's own
        # already-public risk status.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS umkm_businesses (
                id                          TEXT PRIMARY KEY,
                name                        TEXT,
                category                    TEXT,
                grid_id                     TEXT REFERENCES spatial_grids(grid_id),
                district_name               TEXT,
                kecamatan                   TEXT,
                latitude                    NUMERIC,
                longitude                   NUMERIC,
                dist_to_station_m           NUMERIC,
                reference_price_per_txn_idr NUMERIC,
                data_confidence             NUMERIC,
                source                      TEXT,
                updated_at                  TIMESTAMPTZ DEFAULT now()
            );
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_umkm_businesses_grid_id ON umkm_businesses (grid_id);"
        ))
        # GeoJSON view -- per the PRD's 'no tile server, serve compressed
        # GeoJSON directly from PostGIS' decision. The backend's map-layer
        # endpoint can just `SELECT feature FROM spatial_grids_geojson`
        # instead of hand-writing ST_AsGeoJSON/ST_Transform per query.
        conn.execute(text("""
            CREATE OR REPLACE VIEW spatial_grids_geojson AS
            SELECT
                g.grid_id,
                json_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(ST_Transform(g.geom, 4326))::json,
                    'properties', json_build_object(
                        'grid_id', g.grid_id,
                        'district_name', g.district_name,
                        'kecamatan', g.kecamatan,
                        'poi_count', g.poi_count,
                        'ews_code', r.ews_code,
                        'vulnerability_index', r.vulnerability_index,
                        'matching_score', r.matching_score
                    )
                ) AS feature
            FROM spatial_grids g
            LEFT JOIN gentrification_risk_scores r ON r.grid_id = g.grid_id;
        """))


def write_spatial_grids(engine, grid: gpd.GeoDataFrame) -> None:
    cols = ["grid_id", "geometry", "poi_count", "dist_to_station"]
    optional_cols = ["ndbi_mean", "district_name", "within_walk_isochrone", "builtup_pct", "kecamatan",
                      "population", "population_density_per_km2"]
    for col in optional_cols:
        if col in grid.columns:
            cols.append(col)
    payload = grid[cols].copy()
    payload = payload.rename(columns={"dist_to_station": "dist_to_station_m"})
    for col in optional_cols:
        if col not in payload.columns:
            payload[col] = None

    with engine.begin() as conn:
        for _, row in payload.iterrows():
            conn.execute(text("""
                INSERT INTO spatial_grids
                    (grid_id, geom, poi_count, dist_to_station_m, ndbi_mean, district_name,
                     within_walk_isochrone, builtup_pct, kecamatan, population, population_density_per_km2)
                VALUES
                    (:grid_id, ST_GeomFromText(:wkt, 32748), :poi_count, :dist_to_station_m, :ndbi_mean,
                     :district_name, :within_walk_isochrone, :builtup_pct, :kecamatan, :population,
                     :population_density_per_km2)
                ON CONFLICT (grid_id) DO UPDATE SET
                    geom = EXCLUDED.geom, poi_count = EXCLUDED.poi_count,
                    dist_to_station_m = EXCLUDED.dist_to_station_m, ndbi_mean = EXCLUDED.ndbi_mean,
                    district_name = EXCLUDED.district_name,
                    within_walk_isochrone = EXCLUDED.within_walk_isochrone,
                    builtup_pct = EXCLUDED.builtup_pct,
                    kecamatan = EXCLUDED.kecamatan,
                    population = EXCLUDED.population,
                    population_density_per_km2 = EXCLUDED.population_density_per_km2
            """), {
                "grid_id": row["grid_id"], "wkt": row["geometry"].wkt,
                "poi_count": int(row["poi_count"]) if pd.notna(row["poi_count"]) else None,
                "dist_to_station_m": float(row["dist_to_station_m"]) if pd.notna(row["dist_to_station_m"]) else None,
                "ndbi_mean": float(row["ndbi_mean"]) if pd.notna(row["ndbi_mean"]) else None,
                "district_name": row["district_name"] if pd.notna(row["district_name"]) else None,
                "within_walk_isochrone": int(row["within_walk_isochrone"]) if pd.notna(row["within_walk_isochrone"]) else None,
                "builtup_pct": float(row["builtup_pct"]) if pd.notna(row["builtup_pct"]) else None,
                "kecamatan": row["kecamatan"] if pd.notna(row["kecamatan"]) else None,
                "population": int(row["population"]) if pd.notna(row["population"]) else None,
                "population_density_per_km2": float(row["population_density_per_km2"]) if pd.notna(row["population_density_per_km2"]) else None,
            })


def write_risk_scores(engine, grid: pd.DataFrame) -> None:
    with engine.begin() as conn:
        for _, row in grid.iterrows():
            conn.execute(text("""
                INSERT INTO gentrification_risk_scores (grid_id, vulnerability_index, ews_code, matching_score)
                VALUES (:grid_id, :vulnerability_index, :ews_code, :matching_score)
                ON CONFLICT (grid_id) DO UPDATE SET
                    vulnerability_index = EXCLUDED.vulnerability_index,
                    ews_code = EXCLUDED.ews_code, matching_score = EXCLUDED.matching_score,
                    computed_at = now()
            """), {
                "grid_id": row["grid_id"],
                "vulnerability_index": float(row["vulnerability_index"]),
                "ews_code": int(row["ews_code"]),
                "matching_score": float(row["matching_score"]),
            })


def write_narratives(engine, narratives: list[dict]) -> None:
    """narratives: list of {"grid_id", "narrative", "recommendation_type", ...}
    as produced by src/narrative/gemini_client.generate_narrative(). This was
    a real gap -- run_pipeline.py was generating these and discarding them
    without ever calling a writer, caught by checking policy_recommendations
    row count after a full pipeline run (0 rows despite 260 narratives logged)."""
    with engine.begin() as conn:
        for n in narratives:
            conn.execute(text("""
                INSERT INTO policy_recommendations (grid_id, narrative, recommendation_type)
                VALUES (:grid_id, :narrative, :recommendation_type)
                ON CONFLICT (grid_id) DO UPDATE SET
                    narrative = EXCLUDED.narrative,
                    recommendation_type = EXCLUDED.recommendation_type,
                    generated_at = now()
            """), {
                "grid_id": n["grid_id"],
                "narrative": n["narrative"],
                "recommendation_type": n["recommendation_type"],
            })


def write_reallocation_candidates(engine, candidates: pd.DataFrame) -> None:
    """Writes the batch-precomputed reallocation table -- see
    src/modeling/zones.precompute_reallocations() for the architecture
    decision (batch precompute + SQL lookup, not a live Python service).

    Full replace every run, not a per-origin_grid_id delete: a cell that
    was "bahaya" in a PAST run but isn't anymore (EWS cutoffs shift as
    survey data changes) previously kept its stale candidates forever,
    since a delete scoped to only the NEW batch's origin_grid_ids never
    touches origin_grid_ids absent from the new batch. Verified real:
    997 distinct origins had rows in this table but only 248 were
    actually still "bahaya" -- 749 stale origins, 75% of the table.
    Deletes unconditionally (even when `candidates` is empty -- "zero
    danger cells this run" is real new information, not a reason to
    leave a fully stale table untouched)."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM reallocation_candidates"))
        if candidates.empty:
            return
        for _, row in candidates.iterrows():
            conn.execute(text("""
                INSERT INTO reallocation_candidates
                    (origin_grid_id, rank, recommended_grid_id, recommended_district,
                     distance_m, matching_score, crossed_district, search_radius_used_m, expansions_needed)
                VALUES
                    (:origin_grid_id, :rank, :recommended_grid_id, :recommended_district,
                     :distance_m, :matching_score, :crossed_district, :search_radius_used_m, :expansions_needed)
            """), row.to_dict())


def delete_umkm_businesses_by_source(engine, source: str) -> None:
    """Removes every umkm_businesses row tagged with `source` -- used to
    retire v4's records when switching to v5 (see run_pipeline.py): v5 is a
    re-survey of the same real areas with 131 fewer rows (since-closed/
    duplicate/unreliable businesses dropped, not renamed), so a plain
    upsert on the new v5 rows would leave those retired v4 rows behind
    forever instead of removing them."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM umkm_businesses WHERE source = :source"), {"source": source})


def write_umkm_businesses(engine, survey: pd.DataFrame, source: str) -> None:
    """Persists individual real UMKM business records from a survey
    DataFrame (v3's `load_umkm_survey()` or v4's `load_v4_survey()` output
    in run_pipeline.py) -- see ensure_schema()'s umkm_businesses docstring
    for why this table exists and what it deliberately omits (no raw
    rent/revenue). `source` records provenance ("v3" or "v4") since the two
    files have different data-quality caveats (see umkm_survey_v4.py).

    Rows missing a `grid_id` (fell outside every known grid -- see
    assign_grid_coords's own "excluded from GWR training" warning) are
    skipped here too; a business record with no grid to join against for
    risk status isn't useful to display."""
    if survey.empty:
        return
    rows = survey.dropna(subset=["grid_id"])
    with engine.begin() as conn:
        for _, row in rows.iterrows():
            conn.execute(text("""
                INSERT INTO umkm_businesses
                    (id, name, category, grid_id, district_name, kecamatan,
                     latitude, longitude, dist_to_station_m, reference_price_per_txn_idr,
                     data_confidence, source, updated_at)
                VALUES
                    (:id, :name, :category, :grid_id, :district_name, :kecamatan,
                     :latitude, :longitude, :dist_to_station_m, :reference_price_per_txn_idr,
                     :data_confidence, :source, now())
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name, category = EXCLUDED.category,
                    grid_id = EXCLUDED.grid_id, district_name = EXCLUDED.district_name,
                    kecamatan = EXCLUDED.kecamatan, latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude, dist_to_station_m = EXCLUDED.dist_to_station_m,
                    reference_price_per_txn_idr = EXCLUDED.reference_price_per_txn_idr,
                    data_confidence = EXCLUDED.data_confidence, source = EXCLUDED.source,
                    updated_at = now()
            """), {
                "id": str(row["id"]),
                "name": row.get("name"),
                "category": row.get("tenant_type"),
                "grid_id": row["grid_id"],
                "district_name": row.get("district_name") if pd.notna(row.get("district_name")) else None,
                "kecamatan": row.get("kecamatan") if pd.notna(row.get("kecamatan")) else None,
                "latitude": float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
                "longitude": float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
                "dist_to_station_m": float(row["dist_to_station"]) if pd.notna(row.get("dist_to_station")) else None,
                "reference_price_per_txn_idr": float(row["transaction_per_buyer_idr"]) if pd.notna(row.get("transaction_per_buyer_idr")) else None,
                "data_confidence": float(row["data_confidence"]) if pd.notna(row.get("data_confidence")) else None,
                "source": source,
            })
