"""Writes pipeline results to the shared Supabase Postgres+PostGIS instance.
Table shapes match the PRD's 'Database & Spatial Storage Layer' section --
spatial_grids, gentrification_risk_scores. Schema ownership lives in
titiktemu-backend's supabase/migrations/ (see our earlier repo-topology
decision) -- this module only READS that contract, it doesn't define it.

Geometry is written in the internal projected CRS (EPSG:32748) per the PRD;
ST_Transform to EPSG:4326 happens on read, in the backend, not here."""

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import settings


def get_engine():
    return create_engine(settings.database_url)


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
    if "ndbi_mean" in grid.columns:
        cols.append("ndbi_mean")
    if "district_name" in grid.columns:
        cols.append("district_name")
    payload = grid[cols].copy()
    payload = payload.rename(columns={"dist_to_station": "dist_to_station_m"})
    if "ndbi_mean" not in payload.columns:
        payload["ndbi_mean"] = None
    if "district_name" not in payload.columns:
        payload["district_name"] = None

    with engine.begin() as conn:
        for _, row in payload.iterrows():
            conn.execute(text("""
                INSERT INTO spatial_grids (grid_id, geom, poi_count, dist_to_station_m, ndbi_mean, district_name)
                VALUES (:grid_id, ST_GeomFromText(:wkt, 32748), :poi_count, :dist_to_station_m, :ndbi_mean, :district_name)
                ON CONFLICT (grid_id) DO UPDATE SET
                    geom = EXCLUDED.geom, poi_count = EXCLUDED.poi_count,
                    dist_to_station_m = EXCLUDED.dist_to_station_m, ndbi_mean = EXCLUDED.ndbi_mean,
                    district_name = EXCLUDED.district_name
            """), {
                "grid_id": row["grid_id"], "wkt": row["geometry"].wkt,
                "poi_count": int(row["poi_count"]) if pd.notna(row["poi_count"]) else None,
                "dist_to_station_m": float(row["dist_to_station_m"]) if pd.notna(row["dist_to_station_m"]) else None,
                "ndbi_mean": float(row["ndbi_mean"]) if pd.notna(row["ndbi_mean"]) else None,
                "district_name": row["district_name"] if pd.notna(row["district_name"]) else None,
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
    Clears and rewrites per origin_grid_id each run -- candidates should
    fully reflect the latest scoring, not accumulate stale ranks."""
    if candidates.empty:
        return
    with engine.begin() as conn:
        origin_ids = tuple(candidates["origin_grid_id"].unique())
        conn.execute(
            text("DELETE FROM reallocation_candidates WHERE origin_grid_id = ANY(:ids)"),
            {"ids": list(origin_ids)},
        )
        for _, row in candidates.iterrows():
            conn.execute(text("""
                INSERT INTO reallocation_candidates
                    (origin_grid_id, rank, recommended_grid_id, recommended_district,
                     distance_m, matching_score, crossed_district, search_radius_used_m, expansions_needed)
                VALUES
                    (:origin_grid_id, :rank, :recommended_grid_id, :recommended_district,
                     :distance_m, :matching_score, :crossed_district, :search_radius_used_m, :expansions_needed)
            """), row.to_dict())
