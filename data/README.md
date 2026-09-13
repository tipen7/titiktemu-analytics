# data/ — what's tracked vs. what you need to supply

## Tracked in git

- `umkm_survey_v3.geojson` — real UMKM survey, main study area (Dukuh Atas/Blok M), 68 rows.
- `umkm_survey_v4.geojson` — retired. Its `revenue_per_month` is a disclosed mechanical
  proxy (`transaction_per_day * 30`), not an independent measurement -- kept for
  historical reference only. **Not used for model training** (see
  `src/ingestion/umkm_survey_v5.py`'s docstring). Use v5 instead.
- `umkm_survey_v5.geojson` — real UMKM survey, LRT Cibubur-Bogor corridor + other
  Jabodetabek clusters, 149 rows. Re-survey of the same areas v4 covered, with
  independently-measured rent and revenue. This is what the pipeline actually trains on
  alongside v3.
- `demography/demography.csv` — real BPS/Dukcapil kecamatan-level population data.
  Covers Ciracas/Pasar Rebo/Cipayung (Jakarta Timur), several Bogor kecamatan, and
  Cimanggis/Tapos (Depok) — **zero coverage for the main Dukuh Atas/Blok M study area's
  kecamatan** (Setiabudi, Tanah Abang, Menteng, Kebayoran Baru, Mampang Prapatan, Pal
  Merah). Extending this file to cover those six is the main lever left for closing the
  model-accuracy gap (see `CONTEXT.md`).

## NOT tracked (gitignored — supply these locally)

- `administration-district/district-administrative.{shp,dbf,prj,shx,sbn,sbx,cpg,shp.xml}`
  — a real kecamatan boundary shapefile (BPS/Dukcapil "BATAS KECAMATAN" administrative
  boundaries), ~94MB. Used by `src/preprocessing/districts.py`'s
  `tag_grid_with_kecamatan()`. All 8 files must share the exact stem
  `district-administrative` — a shapefile's `.shp`/`.dbf`/`.shx`/etc. are one dataset
  split across matching-named files, not independent pieces.
- `worldcover/ESA_WorldCover_10m_2021_v200_S09E105_Map.tif` — ESA WorldCover 10m land
  cover raster (tile S09E105, covering Jakarta/Bogor/Depok), ~36MB. Used by
  `src/ingestion/sentinel2.py`'s `add_builtup_pct()`. Free, no registration/API key
  required — see https://esa-worldcover.org. (`..._InputQuality.tif`, if you also
  download it, isn't read by any code here — safe to omit.)

Both are excluded from git because they're large (~140MB combined) third-party datasets
this project doesn't own or redistribute, not because they're optional — the pipeline
will fail at the district-tagging / built-up-% ingestion steps without them.
