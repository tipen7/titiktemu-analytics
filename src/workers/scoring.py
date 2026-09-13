"""RQ job for on-demand, single-point scoring -- e.g. the backend enqueues
this the moment a UMKM registers a location, so the Self Discovery Tracker
has an immediate answer instead of waiting for the next nightly batch run.

Per the batch-vs-live architecture decision (CONTEXT.md Sec 4): this job does
NOT re-run GWR/XGBoost for the new point. It looks the point up against the
grid the LAST batch run (run_pipeline.py) already scored and persisted to
Postgres, reusing src/modeling/zones.py's find_zone_for_location() and
find_reallocation() rather than duplicating that logic. If no batch run has
completed yet, it raises rather than fabricating a score -- the caller
should treat that as "try again after the next scheduled run", same
philosophy as the rest of this repo's "don't guess" data handling."""

import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from src.persistence.writer import get_engine
from src.preprocessing.grid import build_study_grid
from src.modeling.zones import find_zone_for_location, find_reallocation


def load_latest_scored_grid(engine) -> gpd.GeoDataFrame:
    """Rebuilds the grid geometry (deterministic from config -- identical
    bbox/cell-size/CRS to what the batch run built) and joins the latest
    persisted scores from Postgres. Cheap: no GWR/XGBoost recomputation,
    just geometry construction + a SQL read + a merge."""
    grid = build_study_grid()
    with engine.connect() as conn:
        scores = pd.read_sql(
            text("""
                SELECT g.grid_id, g.district_name, r.ews_code, r.vulnerability_index, r.matching_score
                FROM spatial_grids g
                JOIN gentrification_risk_scores r ON r.grid_id = g.grid_id
            """),
            conn,
        )
    return grid.merge(scores, on="grid_id", how="inner")


def score_point_against_grid(latitude: float, longitude: float, scored_grid: gpd.GeoDataFrame) -> dict:
    """The pure scoring logic, decoupled from the Postgres read above so it
    can be unit-tested directly against a fixture grid (see
    tests/test_scoring.py) without a live database or RQ/Redis."""
    zone = find_zone_for_location(latitude, longitude, scored_grid)
    if zone is None:
        return {"status": "outside_study_area", "latitude": latitude, "longitude": longitude}

    result = {"status": "scored", **zone}
    if zone["ews_code"] == 2:
        result["reallocation"] = find_reallocation(latitude, longitude, scored_grid)
    return result


def score_new_point(latitude: float, longitude: float) -> dict:
    """The actual RQ job entrypoint -- e.g. enqueued as
    `queue.enqueue(score_new_point, lat, lon)` from the backend.

    Raises ValueError if the batch pipeline hasn't scored a grid yet
    (nothing in gentrification_risk_scores) -- an empty/fabricated result
    would be worse than a clear "not ready yet" error for the caller."""
    engine = get_engine()
    scored_grid = load_latest_scored_grid(engine)
    if scored_grid.empty:
        raise ValueError(
            "No scored grid available yet -- run_pipeline.py hasn't completed a "
            "batch run. On-demand scoring depends on the batch run's output; it "
            "does not compute its own GWR/XGBoost fit."
        )
    return score_point_against_grid(latitude, longitude, scored_grid)
