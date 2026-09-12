# titiktemu-analytics — Remaining TODOs

Consolidated as of this session. Organized by category, not just chronological order.
Each item lists the file(s) it touches and why it's still open (not done vs. can't
verify vs. needs a decision).

---

## 1. Data — blocking real (non-mock) results

- [ ] **Complete, non-null survey CSV** — current pipeline runs end-to-end only against
      mock/imputed data (rows 42-100 fully mocked, 6 rent values median-imputed). Real
      results require the real file. Files: `data/umkm_survey.csv` (not yet provided).
- [ ] **Spatial analysis in QGIS + Geo MAPID Editor** — deferred per this message. Two
      sections already drafted in conversation history (Geo MAPID for collaborative
      coordinate collection, QGIS for visual QA of pipeline output) — revisit when ready.
      No files affected yet.
- [ ] **Live MAPID API access** (Properti Go, Struk Go, Menu Go) — still pending per
      earliest discussion in this project. Files: `src/ingestion/mapid.py` (stub only,
      request/response shapes unverified against real docs).
- [ ] **Live OSM ingestion** — `osmnx` calls are written against the real documented API
      and confirmed installable, but never executed against the real Overpass API (not
      reachable from this sandbox). Files: `src/ingestion/osm.py`.
- [ ] **Real Sentinel-2 imagery** — NDBI computation verified against a synthetic raster
      only; no real Copernicus download wired up. Files: `src/ingestion/sentinel2.py`.
- [ ] **UMKM Self-Tracker submissions as an ongoing data source** — the PRD describes
      UMKM users self-reporting data over time, but nothing in this repo ingests that
      (only the one-time survey CSV). Needs a new ingestion path once the backend's
      self-tracker submissions table exists. Files: new, likely
      `src/ingestion/self_tracker.py`.

## 2. Modeling — quality/validity concerns, not correctness bugs

- [ ] **Cross-district generalization is UNVALIDATED** — `districts.py`'s mechanism
      exists and works mechanically, but Dukuh Atas + Blok M are both central Jakarta
      (~3km apart). The Jakarta-vs-Bogor-style regional variation this is meant to
      capture needs survey data from outside central Jakarta before it can be validated,
      not just built. Files: `src/preprocessing/districts.py`, `src/modeling/gwr.py`.
- [ ] **GWR bandwidth is near-global (98/100 points)** on current data — meaning the
      model isn't really fitting *local* variation yet. Likely an artifact of how
      clustered/sparse the current data is; revisit once real, more spatially spread
      data exists. Files: `src/modeling/gwr.py`.
- [ ] **Covariate jitter is a workaround, not a permanent design** — `_jitter_coincident_covariates()`
      exists because real survey points cluster into very few grid cells, making
      grid-joined features (like `poi_count`) collinear. Once real point-level POI
      density (small-radius buffer count per point, not per grid cell) is available from
      live OSM data, this workaround should be revisited/removed. Files: `src/modeling/gwr.py`.
- [ ] **XGBoost is approximating the GWR surface, not independently validated** — keep
      this framing in any dashboard/report copy per earlier discussion; don't let it get
      presented as a separately-validated classifier. No file changes needed, just a
      documentation/communication discipline item.

## 3. Integration — analytics repo's outputs aren't reachable yet

- [ ] **No live interface for `find_zone_for_location()` / `find_reallocation()`** — this
      is the biggest open architectural question. These are pure Python functions in a
      *batch* pipeline repo, but the UMKM Self Discovery Tracker's zone map and the
      reallocation button are *live, per-request* user actions. Needs a decision: does
      the backend (`titiktemu-backend`) import/call analytics code directly, does
      analytics need a thin query-serving layer after all, or does the batch pipeline
      pre-materialize a lookup table the backend queries with plain SQL (point-in-polygon
      via PostGIS `ST_Contains`, no Python needed at request time)? Recommend the last
      option for architectural consistency with the rest of this repo — but needs your
      confirmation before building it. Files: `src/modeling/zones.py` (logic exists),
      `titiktemu-backend` (consuming side, not yet built).
- [ ] **No compressed GeoJSON export step** — the updated PRD specifies serving spatial
      data as compressed GeoJSON directly from PostGIS (no tile server). Nothing in this
      repo currently exports that; it's implicitly the backend's job via
      `ST_AsGeoJSON`, but worth confirming that's still the plan now that `zones.py` and
      `dashboard_metrics.py` exist as intermediate consumers. Files: none yet — backend
      responsibility, cross-repo.
- [ ] **No formal contract test** between what this repo writes to Postgres
      (`gentrification_risk_scores`, `policy_recommendations`, `dashboard_summary` JSON
      shape) and what the backend/frontend actually expect to read. Risk: a schema
      change here silently breaks the backend. Files: new, e.g. `tests/test_schema_contract.py`.

## 4. Infra / Ops

- [ ] **No scheduling mechanism for `run_pipeline.py`** — open question since early in
      this repo's life (GitHub Actions cron vs. manual vs. thin trigger endpoint), never
      answered. Files: new, e.g. `.github/workflows/run_pipeline.yml`.
- [ ] **No RQ worker process actually configured** — `rq` is installed and importable
      (per the original dependency decision to keep the job queue in Python), but no
      worker entrypoint or Dockerfile `CMD` runs it. Files: `Dockerfile`,
      `src/workers/scoring.py` (still `NotImplementedError` by design).
- [ ] **GeoJSON import path for Geo MAPID exports** — `load_raw_survey()` only reads CSV.
      If the team exports from Geo MAPID as GeoJSON instead, this needs a matching
      reader. Files: `src/ingestion/umkm_survey.py`.

## 5. Testing / maintainability

- [ ] **No automated test suite** — every verification this session (and prior sessions)
      was done via one-off scripts run manually, not committed as repeatable tests.
      Nothing currently catches a regression automatically (e.g., the
      `compute_vulnerability_ratio` orphaned-signature bug from earlier this session
      would have been caught instantly by a test importing the module). Files: new,
      `tests/` directory — currently exists as an empty package
      (`tests/__init__.py`) with no actual test files.
- [ ] **Mock-data flags (`is_mock_rent`, `is_mock_full_row`) and the real
      `data_completeness_score`/`n_fields_imputed` system aren't unified** — two separate
      "how much of this row is real" signals exist (one from `generate_mock_survey.py`,
      one from `impute_missing()`). Should be reconciled so the dashboard can show one
      consistent "data confidence" figure instead of two overlapping ones. Files:
      `scripts/generate_mock_survey.py`, `src/ingestion/umkm_survey.py`.

---

## Already done this session (for reference, not action)

- [x] Coordinate corruption fix, verified against real data (`fix_corrupted_coordinate`)
- [x] New rent-with-period format parser, verified (`parse_rent_with_period`)
- [x] `"--"` null-sentinel handling fixed
- [x] Mock data generation script, run and verified
- [x] Full pipeline wired and run end-to-end against real local PostGIS for the first time
- [x] Point-level `dist_to_station` fix (was grid-cell-borrowed, caused GWR issues)
- [x] `policy_recommendations` write path (was silently discarding generated narratives)
