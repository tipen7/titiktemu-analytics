"""Extends REAL structural grid data -- OSM POI, OSM walk isochrone,
WorldCover built-up %, real kecamatan tagging, real BPS/Dukcapil
demography -- to the LRT Cibubur-Bogor corridor's real station clusters.

Built as SEVERAL small, targeted grids (one per real cluster) rather than
one giant grid spanning the whole ~32km corridor: the clusters are
geographically discontinuous (Ciracas/Harjamukti, then a ~15-25km gap, then
Bojong Nangka/Cikeas Udik, Sentul, Baranangsiang), so a single contiguous
250m grid across the whole span would be >7,000 cells, almost all of it
empty countryside between clusters that no survey point or station is
near. Each area below is sized to its own real survey points' extent
(verified from data/umkm_survey_v4.geojson) plus a margin, same pattern as
the original Ciracas+Harjamukti box.

Real reference coordinates, all independently verified (Wikipedia /
OpenStreetMap Nominatim), not guessed -- see
scripts/fix_survey_v4_station_distances.py for the same trail:
  - Ciracas/Harjamukti: real, operating LRT stations.
  - Bojong Nangka, Sentul, Baranangsiang: real places along the PLANNED,
    not-yet-built Bogor extension -- used as station-area proxies since
    the real station siting isn't finalized.

This now DOES feed run_pipeline.py's GWR fit -- see
src/ingestion/umkm_survey_v4.py for the real (disclosed-estimate) survey
data this grid is joined against. Run standalone first, before
run_pipeline.py, whenever this corridor's structural data needs
(re)building: .venv/Scripts/python.exe -m scripts.ingest_cibubur_bogor_corridor
"""

import pandas as pd
from src.preprocessing.grid import build_study_grid
from src.preprocessing.districts import tag_grid_with_district, tag_grid_with_kecamatan, CIBUBUR_BOGOR_STATIONS, CIBUBUR_BOGOR_BBOX
from src.preprocessing.isochrone import tag_grid_within_isochrone
from src.ingestion.osm import fetch_poi_counts, fetch_walk_network
from src.ingestion.sentinel2 import add_builtup_pct
from src.ingestion.demography import join_demography
from src.persistence.writer import get_engine, ensure_schema, write_spatial_grids

WORLDCOVER_RASTER_PATH = "data/worldcover/ESA_WorldCover_10m_2021_v200_S09E105_Map.tif"

# Each entry: (id_prefix, bbox_wgs84 (west, south, east, north), stations DataFrame)
AREAS = [
    ("cibubur", CIBUBUR_BOGOR_BBOX, CIBUBUR_BOGOR_STATIONS),
    (
        "bojongnangka",
        (106.885, -6.4500, 106.9450, -6.4000),
        pd.DataFrame([{"station_id": "bojong_nangka", "district_name": "Bojong Nangka", "lat": -6.4299962, "lon": 106.9024879}]),
    ),
    (
        "sentul",
        (106.8450, -6.5450, 106.8650, -6.5150),
        pd.DataFrame([{"station_id": "sentul", "district_name": "Sentul", "lat": -6.535861, "lon": 106.856778}]),
    ),
    (
        "baranangsiang",
        (106.8000, -6.6150, 106.8200, -6.5950),
        pd.DataFrame([{"station_id": "baranangsiang", "district_name": "Baranangsiang", "lat": -6.6098861, "lon": 106.8148447}]),
    ),
]


def ingest_area(id_prefix: str, bbox: tuple[float, float, float, float], stations: pd.DataFrame, engine) -> None:
    print(f"\n=== {id_prefix} ===")
    print(f"[1/4] Building grid (bbox={bbox})...")
    grid = build_study_grid(bbox_wgs84=bbox, id_prefix=id_prefix)
    grid["x"] = grid.geometry.centroid.x
    grid["y"] = grid.geometry.centroid.y
    grid = tag_grid_with_district(grid, stations=stations)
    grid = tag_grid_with_kecamatan(grid)
    print(f"      {len(grid)} cells built, tagged across {grid['district_name'].nunique()} station(s) "
          f"and {grid['kecamatan'].nunique()} real kecamatan.")

    print("[2/4] Ingestion (real OSM POI, real OSM walk isochrone, real WorldCover built-up %)...")
    grid = fetch_poi_counts(grid, bbox_wgs84=bbox)
    print(f"      {int(grid['poi_count'].sum())} real OSM commercial POIs found "
          f"(mean {grid['poi_count'].mean():.1f}/cell).")

    walk_graph = fetch_walk_network(bbox)
    grid = tag_grid_within_isochrone(grid, walk_graph, stations, max_distance_m=800.0)
    print(f"      {int(grid['within_walk_isochrone'].sum())}/{len(grid)} cells within an "
          f"800m/~10-min walk isochrone of a station ({len(walk_graph.nodes)} network nodes).")

    grid = add_builtup_pct(grid, WORLDCOVER_RASTER_PATH)
    print(f"      WorldCover built-up %% joined -- mean {grid['builtup_pct'].mean():.1f}%% built-up per cell.")

    print("[3/4] Joining real BPS/Dukcapil demography...")
    grid = join_demography(grid)
    matched = grid["population_density_per_km2"].notna().sum()
    print(f"      {matched}/{len(grid)} cells matched to real kecamatan-level population data.")

    print("[4/4] Persisting structural context data...")
    write_spatial_grids(engine, grid)
    print(f"      Done. {len(grid)} real, structural grid cells persisted (grid_id prefix '{id_prefix}_').")


def run() -> None:
    engine = get_engine()
    ensure_schema(engine)
    for id_prefix, bbox, stations in AREAS:
        ingest_area(id_prefix, bbox, stations, engine)


if __name__ == "__main__":
    run()
