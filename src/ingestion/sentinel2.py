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
    `grid` must already be in the same CRS as the raster.

    Deliberately does NOT pass a `nodata` override to zonal_stats -- doing
    so (this used to hardcode `nodata=np.nan`) replaces the raster's own
    declared nodata value, so real masked/no-data pixels (which use a real
    sentinel like 0, not literal NaN) stopped being excluded from the mean
    and silently came back as a real-looking value instead of null. Found
    by testing an all-nodata raster and an out-of-extent grid cell -- both
    returned 0.0 (a fake 'not built up' reading) instead of a missing
    value until this override was removed."""
    stats = zonal_stats(grid.geometry, ndbi_raster_path, stats=["mean"])
    grid = grid.copy()
    grid["ndbi_mean"] = [s["mean"] for s in stats]
    return grid


# ESA WorldCover class code for "Built-up" -- see
# https://esa-worldcover.org (10m, derived from Sentinel-1+2, no
# registration/API key required -- the simpler alternative to raw NDBI
# band math for a hackathon timeline).
WORLDCOVER_BUILTUP_CLASS = 50


def add_builtup_pct(grid: gpd.GeoDataFrame, worldcover_raster_path: str) -> gpd.GeoDataFrame:
    """Adds a `builtup_pct` column (0-100) -- the share of each grid cell's
    area classified as 'Built-up' in the ESA WorldCover raster. Alternative
    to add_ndbi_mean() -- use whichever data source is actually available;
    both feed the same conceptual role (built-up density per grid cell) in
    the modeling pipeline, so downstream code should treat them as
    interchangeable inputs, not both required.

    ESA WorldCover is delivered in EPSG:4326 -- `grid` must be reprojected
    to match before calling this (unlike add_ndbi_mean, which assumes the
    raster is already in the grid's CRS)."""
    grid_wgs84 = grid.to_crs("EPSG:4326")
    stats = zonal_stats(
        grid_wgs84.geometry, worldcover_raster_path,
        categorical=True, nodata=0,
    )
    pct = []
    for cell_counts in stats:
        total = sum(cell_counts.values())
        builtup = cell_counts.get(WORLDCOVER_BUILTUP_CLASS, 0)
        pct.append(100.0 * builtup / total if total else None)
    grid = grid.copy()
    grid["builtup_pct"] = pct
    return grid