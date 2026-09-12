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
    "spatial_grids": {"grid_id", "geom", "poi_count", "dist_to_station_m", "ndbi_mean", "district_name"},
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
