# CONTEXT.md — titiktemu-analytics

Read this before touching code. It's the accumulated context from many sessions of
design, building, and debugging — the "why" behind decisions that aren't obvious from
the code alone. Where something was verified vs. assumed, that distinction is called
out explicitly; don't upgrade an assumption to a fact without re-checking it yourself.

---"C:\Users\LENOVO\titiktemu-frontend-latest"

## 1. What TitikTemu is

A WebGIS platform mitigating commercial gentrification around Jakarta's TOD (Transit
Oriented Development) zones — the pattern where rising rent near a transit station
displaces the small/local businesses (UMKM) that made the area viable in the first
place. Two user-facing features this repo's output feeds:

1. **UMKM Self Discovery Tracker** — a UMKM (small business) user sees their own
   location's gentrification risk on a red/yellow/green map, and a "View Reallocation"
   button if they're in a danger zone.
2. **Smart Tenant Matching Engine** — recommends where a high-risk tenant could
   relocate to, and separately, matches available commercial space to suitable tenants
   for operators/landlords.

## 2. Sample vs. population — the single most important framing to hold onto

The **survey sample** is ~100 UMKM points around two stations: **Dukuh Atas and Blok M**
— both central Jakarta, ~3km apart. The **population** this is meant to generalize to
is *all* TOD zones in Jabodetabek (Jakarta-Bogor-Depok-Tangerang-Bekasi). This gap
matters constantly:

- The district-tagging mechanism (`src/preprocessing/districts.py`) exists so the model
  *can* account for regional variation (e.g., Jakarta vs. Bogor rent regimes differ) —
  but with only two central-Jakarta stations in the data, that mechanism is **built and
  mechanically working, not validated**. Don't let a future report or dashboard imply
  cross-region generalization has been proven; it hasn't.
- GWR's fitted bandwidth is currently near-global (98/100 points) — the model isn't
  really capturing *local* spatial variation yet, most likely because the real UMKM
  points are extremely tightly clustered (100 real points collapsed into only 5 of 390
  grid cells in one actual run). This is a data characteristic, not a code bug — it'll
  likely improve once survey coverage widens.

## 3. Data — what's real, what's corrupted, what's mock, and why that distinction is load-bearing

The survey CSV has gone through several real-data-quality problems, each discovered by
actually running code against the file, not by inspection:

- **Coordinate corruption**: the raw export strips decimal points from lat/long, then
  re-applies thousands-separator formatting to the resulting integer (e.g. real value
  `-6.202065` becomes `-6.202.065`, or worse, `-62.017.784.033.739.400`). Fixed by
  `fix_corrupted_coordinate()` in `src/ingestion/umkm_survey.py` — reverses this by
  stripping all separators and reinserting a decimal point at a fixed digit position (1
  digit for Jakarta latitude, 3 for longitude). Verified against every row in the actual
  file, including the worst-mangled ones. A second, separate corruption (Indonesian
  comma-as-decimal, e.g. `-6,202313`) is also handled by the same function.
- **Rent format**: the actual file uses period-annotated values like `"13 juta/bulan"`,
  `"160 ribu/hari"`, `"2 juta/5 hari"` — parsed and annualized by
  `parse_rent_with_period()`. An OLDER, different format (`"20% x revenue/day"`) exists
  in `resolve_rent_annual()` as a fallback for forward-compatibility, but does **not**
  appear in the actual survey file — don't assume it's the primary path.
- **`"--"` is this file's null sentinel**, not pandas' default. `load_raw_survey()`
  handles this via explicit `na_values`. If a future CSV uses a different null marker,
  this needs updating.
- **Mock data**: `scripts/generate_mock_survey.py` is a **one-off, temporary** script
  that fills gaps in the survey for pipeline testing — deliberately kept OUT of `src/`
  so it never accidentally runs against real complete data. Real survey data (68 named
  rows as of the last processed file, `umkm_survey_v3_filtered.csv`, delivered to the
  user directly, not committed to this repo) supersedes the mock file once available.
- **`data_confidence`** (in `clean_survey()`'s output) is the unified signal for "how
  much of this row is real vs. imputed/mocked" — there used to be two separate,
  overlapping systems for this; they were unified into one column. Two zero-real-value
  columns (`kondisi_bangunan`, `moda_terhubung`) cannot be median/mode-imputed at all —
  there's nothing to derive a central tendency from — and are left null rather than
  fabricated.

## 4. Architecture decisions (made deliberately, don't re-litigate without new information)

- **Batch pipeline, not a live service.** `run_pipeline.py` is a scheduled/manually-run
  script (GitHub Actions cron, `.github/workflows/run_pipeline.yml`, nightly at 01:00
  WIB + manual trigger), not a hosted API. This was a deliberate simplification once the
  PRD shifted from a live FastAPI microservice design (an earlier, now-superseded
  `titiktemu-ai-service` repo) to an offline analytical pipeline.
- **Zone lookup and reallocation are batch-precomputed, not live-computed.** This was an
  explicit architecture decision (the user asked "you decide, batch or live"). Reasoning:
  scores only change once per pipeline run anyway, so a live Python computation on every
  page-load adds latency for zero freshness benefit. `precompute_reallocations()`
  computes top-3 nearest-safe-zone candidates for every danger-zone grid cell once per
  run, written to `reallocation_candidates`. The backend's "live" lookup is meant to be a
  **plain SQL query** (`SELECT ... WHERE origin_grid_id = ? ORDER BY rank`) — no Python
  service needed at request time. `find_zone_for_location()`/`find_reallocation()` in
  `src/modeling/zones.py` still exist for offline analysis/precomputation use, not as a
  live-callable interface.
- **No tile server.** Per the PRD, spatial data is served as compressed GeoJSON directly
  from PostGIS, not via MVT tiles. `spatial_grids_geojson` (a SQL view created in
  `ensure_schema()`) does the `ST_Transform`/`ST_AsGeoJSON` work once so the backend can
  `SELECT feature FROM spatial_grids_geojson` directly.
- **Job queue lives in Python (RQ), not Node.** Chosen because the actual queued work
  (GWR/XGBoost/Gemini calls) is Python-native; putting the queue in Node would mean
  cross-language duplication of scientific-computing logic. `worker.py` is the RQ worker
  entrypoint — **no real jobs are wired into it yet**; it's a working process shell, not
  a functioning async pipeline. `src/workers/scoring.py`'s `run_scoring_batch()` is a
  deliberate `NotImplementedError` stub.
- **Repo topology**: `titiktemu-frontend` (Next.js), `titiktemu-backend` (Node/Express,
  Supabase Auth), `titiktemu-analytics` (this repo, Python batch pipeline). An earlier
  `titiktemu-ai-service` (FastAPI, live endpoints) is **architecturally superseded** by
  this repo once the PRD moved to the offline-pipeline design — don't resurrect its
  patterns (live `/ai/narrative` endpoint, Redis narrative cache) without confirming
  that's actually wanted again.
- **Auth**: Supabase-issued JWTs, verified via the simple `supabase.auth.getUser(token)`
  round-trip (not local JWT verification) — chosen deliberately for hackathon speed over
  the lower-latency local-verification alternative.



## 6. Repo map (what each piece does)

```
src/
  config.py              # pydantic-settings: DATABASE_URL, study area bbox, grid size, API keys
  preprocessing/
    grid.py               # 250x250m grid builder, EPSG:32748 (UTM Zone 48S)
    districts.py           # nearest-station tagging; STATIONS registry (currently 2: Dukuh Atas, Blok M)
    isochrone.py            # walk-time service area (networkx-based); verified against synthetic graph only
  ingestion/
    umkm_survey.py          # THE critical module -- coordinate fix, rent parsing, cleaning, imputation, geocoding
    self_tracker.py          # ongoing UMKM self-reported data (not live-tested, no backend endpoint exists yet)
    osm.py                    # OSM POI + walk network -- correct API usage, network-blocked in dev sandbox
    sentinel2.py                # NDBI zonal stats -- verified against synthetic raster only
    mapid.py                      # MAPID API stub -- request/response shapes UNVERIFIED, no real docs available
  modeling/
    gwr.py                  # GWR fit + full-grid scoring via IDW-interpolated coefficients (NOT mgwr's own
                             # .predict() -- that has a confirmed unresolved bug, see gwr.py docstring)
    xgboost_ews.py            # EWS classification (0/1/2) + matching score, approximates the GWR surface
    zones.py                    # zone color mapping, point lookup, reallocation search + precomputation
  interpolation/
    idw.py                  # general IDW, used for heatmap AND internally by gwr.py
  narrative/
    gemini_client.py         # Gemini call with fixed prompt template + disclaimer flags; untested live
                              # (generativelanguage.googleapis.com not reachable from dev sandbox)
  persistence/
    writer.py                # schema (spatial_grids, gentrification_risk_scores, policy_recommendations,
                              # reallocation_candidates) + GeoJSON view + all write functions
    dashboard_metrics.py      # operator dashboard aggregates

run_pipeline.py             # orchestrator -- the actual entry point, run via cron or manually
worker.py                    # RQ worker entrypoint -- shell only, no real jobs wired in
scripts/generate_mock_survey.py  # TEMPORARY, testing only -- never import from src/
tests/                       # 26 passing tests as of last verified run -- run before trusting any refactor
```



