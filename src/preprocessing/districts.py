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
