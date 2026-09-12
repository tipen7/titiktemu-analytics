"""NDBI (Normalized Difference Built-up Index) extraction from Sentinel-2 imagery,
via Zonal Statistics per PRD Section 'GIS & Spatial Analytics Pipeline'.

NDBI = (SWIR - NIR) / (SWIR + NIR)
     = (Band 11 - Band 8) / (Band 11 + Band 8)  for Sentinel-2 L2A

NOTE: fetching real Sentinel-2 scenes requires the Copernicus Data Space API,
which is not reachable from this sandbox's network allowlist. This module
takes local raster paths as input -- wiring up the actual Copernicus download
step is a separate, untested-here piece (see README)."""

import geopandas as gpd
import numpy as np
import rasterio
from rasterstats import zonal_stats


def compute_ndbi_raster(nir_band_path: str, swir_band_path: str, output_path: str) -> None:
    """Computes a single NDBI raster from two co-registered Sentinel-2 bands."""
    with rasterio.open(nir_band_path) as nir_src, rasterio.open(swir_band_path) as swir_src:
        nir = nir_src.read(1).astype("float32")
        swir = swir_src.read(1).astype("float32")
        profile = nir_src.profile

        denom = swir + nir
        ndbi = np.where(denom == 0, 0, (swir - nir) / denom)

        profile.update(dtype="float32", count=1)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(ndbi, 1)


def add_ndbi_mean(grid: gpd.GeoDataFrame, ndbi_raster_path: str) -> gpd.GeoDataFrame:
    """Adds an `ndbi_mean` column to the grid via zonal statistics.
    `grid` must already be in the same CRS as the raster."""
    stats = zonal_stats(grid.geometry, ndbi_raster_path, stats=["mean"], nodata=np.nan)
    grid = grid.copy()
    grid["ndbi_mean"] = [s["mean"] for s in stats]
    return grid
