# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Offline batch pipeline for a TOD (transit-oriented-development) commercial gentrification
early-warning system: GWR (spatial regression) + XGBoost (EWS classification/matching) + LLM
(Gemini) narrative synthesis, writing scored results to a shared Supabase Postgres/PostGIS
instance that `titiktemu-backend` reads from.

**This is a batch script, not a hosted service.** It runs on a schedule (GitHub Actions cron,
see `.github/workflows/run_pipeline.yml`) or manually, and exits. There is no live HTTP API in
this repo. Live, per-request features described in the PRD (e.g. the UMKM zone map, the
reallocation button) are served by the backend directly querying tables/views this pipeline
writes — no Python runs at request time. Keep this boundary in mind before adding anything
that looks like a live endpoint here.

`worker.py` (RQ worker) exists as a separate, currently-stub process for *future* live-enqueued
jobs (e.g. on-demand scoring of a single newly-registered point) — distinct from the scheduled
full-grid batch run. Its job implementation (`src/workers/scoring.py`) is an intentional
`NotImplementedError` stub; don't "complete" it without checking with the user first, since the
job contract hasn't been decided yet.

## Commands

```bash
# Setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in DATABASE_URL at minimum

# Run the full pipeline
.venv/bin/python run_pipeline.py

# Run the RQ worker (separate process, not part of the batch run)
.venv/bin/python worker.py

# Tests
pytest                              # unit tests; contract tests auto-skip without a live DB
pytest tests/test_gwr.py            # single file
pytest tests/test_zones.py::test_find_reallocation_expands_radius   # single test
pytest -m contract                  # schema-contract tests only (needs DATABASE_URL)
pytest -m "not contract"            # everything except live-DB tests
```

Docker: `Dockerfile` defaults `CMD` to `python run_pipeline.py` (the scheduled batch job the
GitHub Actions workflow invokes). Override with `docker run <image> python worker.py` to run
the worker process instead.

## Pipeline architecture

`run_pipeline.py` is the single entry point and orchestrates five stages, matching the PRD's
"Technology Architecture" section:

1. **Grid construction** (`src/preprocessing/grid.py`) — builds a 250m grid over the study area
   bbox in the projected CRS (`EPSG:32748`, UTM 48S) so cell size is a real metric distance.
   All internal spatial math stays in this projected CRS; conversion to `EPSG:4326` only
   happens when serving GeoJSON to a client (`ST_Transform` on read, done in Postgres/the
   backend, not here).
2. **District tagging** (`src/preprocessing/districts.py`) — assigns each grid cell to its
   nearest station (registry in `STATIONS`) and recomputes an authoritative `dist_to_station`.
   Currently only two stations exist (Dukuh Atas, Blok M), both central Jakarta ~3km apart —
   cross-district generalization is *mechanically wired up* but **not statistically validated**;
   don't present district-normalized results as validated across regions until survey data from
   outside central Jakarta exists.
3. **Ingestion** (`src/ingestion/`) — `osm.py` (POI density), `sentinel2.py` (NDBI from
   Sentinel-2), `mapid.py` (MAPID API), `umkm_survey.py` (the UMKM survey CSV/GeoJSON, the one
   ingestion path that's fully live). The first three are stubbed/unverified against real
   network access in this environment; `run_pipeline.py` currently fakes `poi_count` with a
   seeded RNG as a flagged placeholder rather than pretending it's real data — don't remove that
   flag/print without replacing it with real ingestion.
4. **Modeling**:
   - `src/modeling/gwr.py` — Geographically Weighted Regression, Adaptive Bisquare kernel,
     fit only on rows with a real `vulnerability` value (dropna'd). Two deliberate, documented
     workarounds live here — don't "clean them up" without understanding why first:
     - `_jitter_coincident_covariates()`: tiny deterministic jitter to break exact ties in
       grid-joined covariates (survey points cluster into very few grid cells → near-zero
       variance columns → singular design matrix in local kernel neighborhoods).
     - `predict_gwr_surface()` does NOT use `mgwr`'s own `GWR.predict()` for scoring the full
       grid — that method has a confirmed upstream bug (pysal/mgwr#50) that throws when
       predicting at more locations than were trained on. Instead it IDW-interpolates the
       fitted local coefficients onto new grid points and applies them directly (same technique
       GIS tools use for GWR coefficient surfaces).
     - GWR bandwidth currently comes out near-global (~98% of n) on the current clustered
       survey data — a data/sample-size symptom, not a bug; the code prints a warning rather
       than failing.
   - `src/modeling/xgboost_ews.py` — turns the vulnerability surface into EWS codes
     (0/1/2 = aman/waspada/bahaya) and trains an XGBoost model. Framing note: XGBoost is
     *approximating* the GWR surface for the matching score, not an independently validated
     classifier — keep that framing in any dashboard/report copy.
   - `src/modeling/zones.py` — `find_zone_for_location()` (point-in-polygon lookup) and
     `find_reallocation()` / `precompute_reallocations()` (nearest-safe-zone search with
     expanding radius). **Architecture decision: batch-precomputed, not a live service** —
     `precompute_reallocations()` runs once per pipeline run for every danger cell, so the
     backend's "live" reallocation lookup is a single `SELECT ... WHERE origin_grid_id = ...`
     with no Python at request time. Preserve this pattern; don't reintroduce a live compute
     path here without an explicit decision to do so.
5. **Narrative synthesis** (`src/narrative/gemini_client.py`) — calls Gemini per flagged
   (waspada/bahaya) grid cell, strict JSON-only response contract; raises rather than silently
   persisting a malformed response. Returns a stub narrative (no API call) when
   `GEMINI_API_KEY` is unset — this is the expected/default local-dev behavior, not a failure.
6. **Persistence** (`src/persistence/writer.py`, `src/persistence/dashboard_metrics.py`) —
   writes to `spatial_grids`, `gentrification_risk_scores`, `policy_recommendations`,
   `reallocation_candidates`, plus a `spatial_grids_geojson` view (PostGIS `ST_AsGeoJSON`, no
   tile server, per the PRD). **Schema ownership lives in `titiktemu-backend`'s Supabase
   migrations** — `ensure_schema()` here is a local-dev/testing convenience (idempotent
   `CREATE TABLE IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS`), not the source of truth. Changing
   a table shape here without checking the backend's migrations and `tests/test_schema_contract.py`
   risks silently breaking the backend.

## Survey data cleaning (`src/ingestion/umkm_survey.py`)

This module owns all UMKM survey CSV cleaning and is the most idiosyncratic part of the repo —
most of its logic exists to reverse *specific, verified* corruption in the real survey export,
not general-purpose parsing:

- `fix_corrupted_coordinate()` reverses a specific Sept 2026 export bug where decimal points
  were stripped from lat/long then re-grouped with thousands separators (plus a separate
  comma-as-decimal locale issue). It is applied unconditionally to every row — verified
  harmless on already-well-formed values.
- `"--"` is this export's null sentinel (not pandas' default) — `load_raw_survey()` passes it
  explicitly to `na_values`.
- `parse_rent_with_period()` / `parse_currency()` / `resolve_rent_annual()` handle this export's
  actual rent format (`"13 juta/bulan"`, bare `"50 juta"`); the older `"20% x revenue/day"`
  formula format is kept only for forward-compatibility and doesn't appear in current data.
- `compute_vulnerability_ratio()` (the GWR y-variable, rent burden vs. turnover) is only
  computable where both rent and revenue exist on the same row — currently a small subset of
  rows. This is the honest state of the data, not something to backfill by guessing.
- `impute_missing()` median/mode-imputes remaining gaps but the caller is expected to track
  `n_fields_imputed` / `data_completeness_score` per row so imputed values are never silently
  treated as equivalent to real survey data downstream.
- `assign_grid_coords()` computes `dist_to_station` **per survey point**, not borrowed from the
  enclosing grid cell — verified that grid-cell-borrowed distances caused GWR's local design
  matrix to go singular when multiple points shared a cell.

Real, non-mock, non-imputed survey data is still the main blocker for real (not
mock/imputed) pipeline results — see `TODO.md` §1 for the current state of every data source.

## Testing

- `tests/conftest.py` provides `grid` and `scored_grid` fixtures (`scored_grid` fabricates
  structured-but-fake EWS/vulnerability/matching scores so zone/dashboard-metric logic can be
  tested without a real GWR/XGBoost run).
- `tests/test_schema_contract.py` is marked `contract` and requires a live `DATABASE_URL` — it
  skips cleanly (not a failure) when no DB is reachable. It guards the exact table/column shapes
  the backend depends on (see `EXPECTED_TABLES`); update it whenever you change a written
  table's shape.
- Config (`src/config.py`) is a `pydantic-settings` `BaseSettings` reading from `.env` — tests
  that need specific settings should not rely on the developer's local `.env` being any
  particular value.

## Cross-repo context

- `DATABASE_URL` connects to the **same** Supabase Postgres instance `titiktemu-backend` uses,
  via a direct connection (not the PgBouncer pooler) — this pipeline runs longer transactions
  than the backend's request-scoped queries.
- There is currently no formalized interface between this repo's pure-Python functions
  (`find_zone_for_location`, `find_reallocation`) and the backend's live per-request needs
  beyond the batch-precompute-then-SQL-lookup pattern described above — see `TODO.md` §3 before
  changing that boundary.
