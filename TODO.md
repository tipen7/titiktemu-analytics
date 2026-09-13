# titiktemu-analytics — Remaining TODOs

Reflects the real state as of the v3+v5 pipeline run (CV-bandwidth GWR, 64.7% real
accuracy). Organized by category. Items from earlier sessions that are now genuinely
done have been removed rather than kept as stale checkmarks — see git history for that
trail if needed.

---

## 1. Model accuracy — currently 64.7% (n=119), target 80-90% not yet reached

Real, LOOCV-against-real-survey accuracy (see `src/modeling/xgboost_ews.validate_ews_against_survey`),
trained on `umkm_survey_v3.geojson` + `umkm_survey_v5.geojson` only (`v4` retired — see
`data/README.md`). CV-selected GWR bandwidth (vs. the old AICc default) got this from
57.1% to 64.7% — a genuine methodology improvement, not a data change. A flexible
non-spatial classifier (RandomForest, 5-fold CV) on the same 4 features caps at the same
~64%, which means the ceiling right now is the FEATURE SET, not the modeling approach.
Confirmed NOT to help (see `scripts/experiment_chain_density*.py`, kept as a record of a
negative result): an OSM chain/franchise-density feature — tested both as a raw count
and as a ratio, both scored below the 4-feature baseline.

What's left, in order of expected value:

- [ ] **Main study area demography** — `data/demography/demography.csv` has zero
      coverage for Dukuh Atas/Blok M's own kecamatan (Setiabudi, Tanah Abang, Menteng,
      Kebayoran Baru, Mampang Prapatan, Pal Merah), so `population_density_per_km2` is
      null for all 390 main-grid cells. User has this data and will provide it.
- [ ] **Complete the 98 partial survey rows** — 217 real rows exist (68 v3 + 149 v5),
      but only 119 have BOTH rent and revenue on the same row (GWR's y-variable needs
      both). Cheapest lever: go back to already-surveyed businesses missing one field,
      not new site visits. Worst gaps: main grid (33/68 missing), Cibubur (19/48),
      Sentul (11/27).
- [ ] **More real survey coverage in thin areas** — Depok Pusat (2 usable rows), Bogor
      Utara (4), TMII (4) are too sparse for GWR to say anything reliable locally.
- [ ] **A genuinely new predictive feature** — something that actually tracks
      tenant-specific economics (independent stall vs. established chain), not just
      location-structural signal. Chain POI density didn't pan out; the real
      differentiator observed in v3 (small stalls at vulnerability ~0.5-1.0 vs. chains
      at ~0.005-0.03 on the same street) isn't captured by any current grid-cell-level
      feature.

## 2. Narrative generation — Gemini quota is the binding constraint

Last full run: 3/2032 flagged cells got a real narrative before hitting the API quota
(`GeminiQuotaExceededError`, handled — stops immediately instead of retrying each of the
remaining ~2000 cells individually). `policy_recommendations` is correspondingly sparse.
Not a code bug — either raise the quota/billing tier, or add a resumable
already-narrated-cells skip so re-runs top up instead of restarting from zero.

## 3. Data provenance still worth tightening

- [ ] **`umkm_survey_v4.geojson`'s relationship to v5 isn't formally reconciled** — v5 is
      a re-survey of the same real areas with 131 fewer rows (since-closed/
      duplicate/unreliable businesses dropped). No row-level mapping exists between which
      v4 business became which v5 business (or was dropped). Not blocking anything
      today since v4 is fully excluded from training, but would matter if v4 is ever
      revisited.
- [ ] **Mock-data flags (`is_mock_rent`, `is_mock_full_row`) vs. `data_completeness_score`/
      `n_fields_imputed`** — two overlapping "how real is this row" signals still exist
      for the original v3 pipeline path (`umkm_survey.py`). Should be reconciled into
      one consistent confidence figure.

## 4. Infra / Ops

- [ ] **No RQ worker actually implemented** — `worker.py` runs the process, but
      `src/workers/scoring.py` is an intentional `NotImplementedError` stub (the
      on-demand single-point scoring job contract hasn't been decided). Don't
      "complete" it without checking first, per that file's own docstring.
- [ ] **MAPID API (Properti Go/Struk Go/Menu Go) still stubbed** — real access
      confirmed reachable in an earlier session, but the actual dataset returned covers
      Kota Tangerang/Bogor/Jakarta Timur/Depok, none of which overlap the main study
      area's bbox. Revisit if scope expands there.
- [x] GitHub Actions scheduled run (`.github/workflows/run_pipeline.yml`) exists.
- [x] Real OSM POI + walk-isochrone ingestion is live (confirmed reachable from this
      environment as of the v3+v5 run — the "not reachable from this sandbox" caveat in
      `src/ingestion/osm.py`'s docstring is stale, left in place as historical context
      for why the code was originally written untested against a live endpoint).
