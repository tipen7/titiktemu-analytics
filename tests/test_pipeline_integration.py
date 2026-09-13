"""Integration test: runs the actual run_pipeline.run() orchestrator
end-to-end against a live Postgres connection.

Isolation: everything runs inside ONE held-open connection's transaction,
using a SAVEPOINT per write_*() call (mirroring each engine.begin() the
orchestrator makes), and the outer transaction is rolled back at teardown
-- nothing this test writes is ever committed.

This is NOT the first design tried here. An earlier version isolated
writes via a per-test Postgres SCHEMA, set through a `search_path`
connect_args option. That silently failed: this project's DATABASE_URL
goes through Supabase's transaction-mode pooler (port 6543), which does
not guarantee a client's session-level settings (search_path included)
survive across the separate pooled connections run_pipeline.write_*()
opens via repeated engine.begin() calls -- so writes landed in `public`,
the real schema the backend reads, and corrupted real pipeline output.
(Confirmed by a manual check afterwards, and confirmed as the mechanism
by reproducing the schema-isolation failure directly.)

The SAVEPOINT approach avoids that failure mode structurally: it holds a
single physical connection for the whole test and never commits the outer
transaction, so there's no second pooled connection for isolation to leak
across, and no dependency on search_path being honored at all -- verified
directly against this same pooler (open a transaction, INSERT+CREATE TABLE
inside a savepoint, confirm a second connection sees neither before nor
after rollback) before relying on it here.

Requires a live DATABASE_URL -- skips cleanly (same pattern as
test_schema_contract.py) when unavailable."""

import json
from contextlib import contextmanager

import numpy as np
import pytest
from sqlalchemy import create_engine, text

import run_pipeline
from src.config import settings


pytestmark = pytest.mark.contract


def _write_synthetic_survey_geojson(path, west, south, east, north, n=20, seed=7):
    """A small, fully-synthetic UMKM survey -- structurally identical to a
    real Geo MAPID export (same property names umkm_survey.py expects),
    but with fabricated points/values so this test never depends on real,
    private survey data. Inset well within the bbox so reprojection-
    clipping at the study area's edges can't drop points below
    fit_gwr()'s 10-row minimum."""
    rng = np.random.default_rng(seed)
    inset_lon = 0.15 * (east - west)
    inset_lat = 0.15 * (north - south)

    features = []
    for i in range(n):
        lon = rng.uniform(west + inset_lon, east - inset_lon)
        lat = rng.uniform(south + inset_lat, north - inset_lat)
        revenue_juta = rng.integers(10, 60)
        rent_juta = rng.integers(5, 40)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat), 0]},
            "properties": {
                "id": str(i + 1),
                "name": f"Synthetic UMKM {i + 1}",
                "tenant_type (umkm/franchise-tetap/seasonal)": "umkm-tetap",
                "period (tahun/bulan/minggu)": "1 tahun",
                "transaction_per_day (ribu/juta/milliar)": "500 ribu",
                "revenue_per_month": f"{revenue_juta} juta",
                "transaction_per_buyer": "20 ribu",
                "rent_trend": "5%",
                "rent_price_annual": f"{rent_juta} juta/tahun",
            },
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


class _SavepointEngine:
    """Duck-types just enough of SQLAlchemy's Engine interface for
    src/persistence/writer.py and dashboard_metrics.py's usage (every
    write goes through `with engine.begin() as conn: ...`) -- but each
    "begin" opens a SAVEPOINT on one shared, already-open connection
    instead of a new pooled connection with its own top-level transaction.
    """

    def __init__(self, connection):
        self._connection = connection

    def begin(self):
        return self._begin_ctx()

    @contextmanager
    def _begin_ctx(self):
        nested = self._connection.begin_nested()
        try:
            yield self._connection
        except BaseException:
            nested.rollback()
            raise
        else:
            nested.commit()

    def connect(self):
        return self._connect_ctx()

    @contextmanager
    def _connect_ctx(self):
        yield self._connection


@pytest.fixture
def rollback_engine():
    try:
        base_engine = create_engine(settings.database_url)
        connection = base_engine.connect()
        # Connection.begin() must come before any execute() -- SQLAlchemy
        # 2.0 auto-begins a transaction on first execute(), and a second
        # explicit begin() on top of that raises InvalidRequestError.
        outer_txn = connection.begin()
        connection.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("No live Postgres connection available for pipeline integration test")
    try:
        yield _SavepointEngine(connection)
    finally:
        outer_txn.rollback()
        connection.close()
        base_engine.dispose()


def test_run_pipeline_end_to_end(tmp_path, monkeypatch, rollback_engine):
    survey_path = tmp_path / "synthetic_survey.geojson"
    _write_synthetic_survey_geojson(
        str(survey_path),
        settings.study_area_bbox_west, settings.study_area_bbox_south,
        settings.study_area_bbox_east, settings.study_area_bbox_north,
    )

    monkeypatch.setattr(run_pipeline, "SURVEY_CSV_PATH", str(survey_path))
    monkeypatch.setattr(run_pipeline, "get_engine", lambda: rollback_engine)
    # Force the narrative stub path (no network, deterministic, fast) --
    # Gemini's live behavior is exercised separately, not by this test.
    monkeypatch.setattr(settings, "gemini_api_key", "")

    conn = rollback_engine._connection
    run_pipeline.ensure_schema(rollback_engine)
    # dashboard_summary is created lazily inside write_dashboard_metrics()
    # itself, not ensure_schema() -- mirror that DDL here (idempotent) so
    # the baseline read below works whether or not a prior run has already
    # created it.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS dashboard_summary (
            id           SERIAL PRIMARY KEY,
            metrics      JSONB,
            computed_at  TIMESTAMPTZ DEFAULT now()
        );
    """))
    dashboard_count_before = conn.execute(text("SELECT COUNT(*) FROM dashboard_summary")).scalar()

    run_pipeline.run()

    grid_count = conn.execute(text("SELECT COUNT(*) FROM spatial_grids")).scalar()
    risk_count = conn.execute(text("SELECT COUNT(*) FROM gentrification_risk_scores")).scalar()
    dashboard_count_after = conn.execute(text("SELECT COUNT(*) FROM dashboard_summary")).scalar()
    # Every currently-flagged (waspada/bahaya) cell should have a matching
    # narrative row -- checked by existence, not by a raw total-row-count
    # comparison, since policy_recommendations can carry narrative rows
    # from a PRIOR run's different flagged set (write_narratives only
    # upserts currently-flagged cells; it never prunes ones that are no
    # longer flagged) -- a real, separate staleness characteristic of that
    # table, not something this test should trip over.
    unnarrated_flagged = conn.execute(text("""
        SELECT COUNT(*) FROM gentrification_risk_scores r
        WHERE r.ews_code > 0
          AND NOT EXISTS (SELECT 1 FROM policy_recommendations p WHERE p.grid_id = r.grid_id)
    """)).scalar()

    assert grid_count > 0
    assert risk_count == grid_count
    assert unnarrated_flagged == 0
    assert dashboard_count_after == dashboard_count_before + 1
