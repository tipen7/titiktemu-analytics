"""Contract tests: guard the schema this repo writes to against silent
drift, since the backend reads these tables/views directly (see TODO.md
item 3). Requires a real Postgres connection (DATABASE_URL) -- skips
cleanly if unavailable rather than failing the whole suite in an
environment without one."""

import pytest
from sqlalchemy import inspect, text
from src.persistence.writer import get_engine, ensure_schema

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def engine():
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("No live Postgres connection available for contract tests")
    ensure_schema(eng)
    return eng


EXPECTED_TABLES = {
    "spatial_grids": {"grid_id", "geom", "poi_count", "dist_to_station_m", "ndbi_mean", "district_name",
                       "within_walk_isochrone", "builtup_pct", "kecamatan", "population",
                       "population_density_per_km2"},
    "gentrification_risk_scores": {"grid_id", "vulnerability_index", "ews_code", "matching_score", "computed_at"},
    "policy_recommendations": {"grid_id", "narrative", "recommendation_type", "ai_generated", "requires_human_review", "generated_at"},
    "reallocation_candidates": {"origin_grid_id", "rank", "recommended_grid_id", "recommended_district",
                                 "distance_m", "matching_score", "crossed_district",
                                 "search_radius_used_m", "expansions_needed", "computed_at"},
}


@pytest.mark.parametrize("table_name,expected_cols", EXPECTED_TABLES.items())
def test_table_has_expected_columns(engine, table_name, expected_cols):
    inspector = inspect(engine)
    actual_cols = {c["name"] for c in inspector.get_columns(table_name)}
    missing = expected_cols - actual_cols
    assert not missing, f"{table_name} is missing expected columns: {missing}"


def test_spatial_grids_geojson_view_exists(engine):
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT 1 FROM information_schema.views WHERE table_name = 'spatial_grids_geojson'"
        )).fetchone()
    assert result is not None


# Indonesia's approximate lon/lat envelope -- generous on purpose, just
# tight enough to catch an accidental coordinate-order swap (this study
# area's cells should read as ~106 E, ~-6 N; a swap would put them at
# ~106 N, ~-6 E, well outside this box).
INDONESIA_LON_RANGE = (94.0, 142.0)
INDONESIA_LAT_RANGE = (-12.0, 7.0)


def test_spatial_grids_geojson_features_are_well_formed(engine):
    """Only test_spatial_grids_geojson_view_exists above checks the view
    exists -- this checks what it actually returns is valid, correctly-
    ordered GeoJSON, since ST_AsGeoJSON/ST_Transform mistakes (wrong SRID,
    swapped axis order) would otherwise only surface once a map consumer
    renders garbage."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT feature FROM spatial_grids_geojson LIMIT 25")).fetchall()

    assert rows, "spatial_grids_geojson returned no rows -- run the pipeline at least once first"

    for (feature,) in rows:
        assert feature["type"] == "Feature"

        geometry = feature["geometry"]
        assert geometry["type"] == "Polygon"
        rings = geometry["coordinates"]
        assert rings and len(rings[0]) >= 4  # closed ring: >=3 distinct points + repeated first
        for lon, lat in rings[0]:
            assert INDONESIA_LON_RANGE[0] <= lon <= INDONESIA_LON_RANGE[1], (
                f"longitude {lon} outside Indonesia's range -- possible lat/lon axis swap"
            )
            assert INDONESIA_LAT_RANGE[0] <= lat <= INDONESIA_LAT_RANGE[1], (
                f"latitude {lat} outside Indonesia's range -- possible lat/lon axis swap"
            )

        properties = feature["properties"]
        assert {"grid_id", "district_name", "poi_count"}.issubset(properties.keys())
