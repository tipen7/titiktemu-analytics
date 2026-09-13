"""TODO 1a / 1f: District/station tagging per grid cell.

Per the updated PRD context: the SAMPLE is UMKM around Dukuh Atas and Blok M
specifically, but the POPULATION is all TOD zones in Jabodetabek. Tagging
each grid cell with its nearest station/district is what makes district-level
covariates (rent regime, regional cost index) possible later -- and it's
what the reallocation search (1d) uses to know when it has crossed into a
'different district' vs just a different grid cell of the same one.

NOTE (honesty, not hedging): with only Dukuh Atas + Blok M in the current
sample -- both central Jakarta, ~3km apart -- this mechanism can be built
and wired up, but cross-region generalization (Jakarta vs Bogor rent
regimes) cannot be VALIDATED until UMKM data from stations outside central
Jakarta exists. Don't present district-normalized scores as validated
across regions until then.
"""

import geopandas as gpd
import numpy as np
import pandas as pd


# Station registry -- extend this as more TOD districts get added to the
# population beyond the current Dukuh Atas/Blok M sample.
STATIONS = pd.DataFrame([
    {"station_id": "dukuh_atas", "district_name": "Dukuh Atas", "lat": -6.1988, "lon": 106.8230},
    {"station_id": "blok_m", "district_name": "Blok M", "lat": -6.2440, "lon": 106.7995},
])

# LRT Cibubur-Bogor corridor -- real, currently-operating/real-landmark
# stations only (Harjamukti/Ciracas are real LRT stations; Sentul/Bojong
# Nangka/Baranangsiang are real places along the PLANNED, not-yet-built
# Bogor extension, used as station-area proxies -- see
# scripts/ingest_cibubur_bogor_corridor.py's docstring for the verification
# trail on each coordinate). Kept as a separate registry from STATIONS
# (not merged in) since these are a distinct geographic cluster used for a
# second, far-away grid -- see run_pipeline.py's combined-grid handling.
CIBUBUR_BOGOR_STATIONS = pd.DataFrame([
    {"station_id": "ciracas", "district_name": "Ciracas", "lat": -6.3237, "lon": 106.8867},
    {"station_id": "harjamukti", "district_name": "Harjamukti", "lat": -6.373988, "lon": 106.895623},
])

# (west, south, east, north), WGS84 -- both stations plus a ~2-3.5km margin.
CIBUBUR_BOGOR_BBOX = (106.8650, -6.4050, 106.9150, -6.2950)


def tag_grid_with_district(grid: gpd.GeoDataFrame, stations: pd.DataFrame = STATIONS) -> gpd.GeoDataFrame:
    """Adds `station_id`, `district_name`, and (re-derived, authoritative)
    `dist_to_station` columns -- each cell assigned to its NEAREST station,
    distance computed in the grid's own projected CRS for accuracy."""
    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs="EPSG:4326",
    ).to_crs(grid.crs)

    grid = grid.copy()
    cx, cy = grid.geometry.centroid.x.values, grid.geometry.centroid.y.values
    sx, sy = stations_gdf.geometry.x.values, stations_gdf.geometry.y.values

    # (n_cells, n_stations) distance matrix -- fine at this scale (hundreds
    # of cells x a handful of stations), no need for a spatial index yet.
    dist_matrix = np.sqrt((cx[:, None] - sx[None, :]) ** 2 + (cy[:, None] - sy[None, :]) ** 2)
    nearest_idx = dist_matrix.argmin(axis=1)

    grid["station_id"] = stations_gdf["station_id"].values[nearest_idx]
    grid["district_name"] = stations_gdf["district_name"].values[nearest_idx]
    grid["dist_to_station"] = dist_matrix[np.arange(len(grid)), nearest_idx]
    return grid


ADMIN_DISTRICT_SHP_PATH = "data/administration-district/district-administrative.shp"


def tag_grid_with_kecamatan(grid: gpd.GeoDataFrame, shp_path: str = ADMIN_DISTRICT_SHP_PATH) -> gpd.GeoDataFrame:
    """Adds a `kecamatan` column via a real point-in-polygon join against
    BPS/Dukcapil administrative boundaries (national kecamatan-level file --
    verified it covers the current study area's 6 kecamatan: Setiabudi,
    Tanah Abang, Menteng [Dukuh Atas side], Kebayoran Baru, Mampang
    Prapatan, Pal Merah [Blok M side]).

    NOTE (honesty, not hedging): this is DISTINCT from `district_name`
    above ("TOD catchment nearest station") -- `kecamatan` is the real
    government administrative unit. It exists as grid metadata only; it is
    NOT joined to data/demography/demography.csv as a modeling feature,
    because that file's coverage (Ciracas, Pasar Rebo, Cipayung in Jakarta
    Timur; several Bogor/Depok kecamatan) has ZERO overlap with this study
    area's kecamatan -- joining it would silently produce an all-null
    column, not a real feature. Same reasoning `run_pipeline.py` already
    applies to deferring MAPID (real API, real data, wrong geography for
    the current bbox). Revisit once the study area actually expands to
    Jakarta Timur/Bogor/Depok stations.

    The shapefile as delivered had its .shp renamed without renaming its
    .shx/.dbf/.prj sidecars (verified: loading failed until the sidecars
    were copied to match `district-administrative.*`) -- fixed in the repo
    data itself, not worked around here.
    """
    admin = gpd.read_file(shp_path)[["KECAMATAN", "geometry"]].to_crs(grid.crs)
    joined = gpd.sjoin(
        grid.copy(), admin, how="left", predicate="intersects",
    ).drop(columns=["index_right"])
    # A grid cell can straddle >1 kecamatan boundary at a 250m resolution --
    # sjoin then returns duplicate rows; keep one match per grid cell
    # (first is fine, this is metadata, not a modeling input).
    joined = joined[~joined.index.duplicated(keep="first")]
    grid = grid.copy()
    grid["kecamatan"] = joined["KECAMATAN"].str.title()
    return grid
