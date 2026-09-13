import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from src.ingestion.sentinel2 import add_ndbi_mean, compute_ndbi_raster

CRS = "EPSG:32748"


def _write_raster(path, array, transform, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1], count=1,
        dtype=array.dtype, crs=CRS, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def test_add_ndbi_mean_all_nodata_raster_returns_null_not_zero(tmp_path):
    """A fully cloud-masked/nodata scene should surface as a missing value
    per cell -- silently reporting 0 would look like a real "not built up"
    reading instead of "no data here"."""
    transform = from_origin(0, 100, 10, 10)
    array = np.full((10, 10), -9999, dtype="float32")
    raster_path = tmp_path / "all_nodata.tif"
    _write_raster(str(raster_path), array, transform, nodata=-9999)

    grid = gpd.GeoDataFrame({"grid_id": ["g1"]}, geometry=[box(0, 0, 50, 50)], crs=CRS)
    result = add_ndbi_mean(grid, str(raster_path))

    assert result["ndbi_mean"].isna().all()


def test_add_ndbi_mean_cell_outside_raster_extent_returns_null(tmp_path):
    """A grid cell entirely outside the raster's footprint (e.g. a study
    area edge cell not covered by a particular Sentinel-2 tile) must not
    crash and must not silently borrow a neighboring cell's value. Raster
    declares nodata=0, matching how real Sentinel-2 L2A products mark
    missing pixels (boundless reads pad out-of-extent area with the same
    declared nodata value)."""
    transform = from_origin(0, 100, 10, 10)
    array = np.full((10, 10), 0.3, dtype="float32")
    raster_path = tmp_path / "small.tif"
    _write_raster(str(raster_path), array, transform, nodata=0)

    grid = gpd.GeoDataFrame(
        {"grid_id": ["inside", "far_outside"]},
        geometry=[box(0, 0, 50, 50), box(10_000, 10_000, 10_050, 10_050)],
        crs=CRS,
    )
    result = add_ndbi_mean(grid, str(raster_path))

    inside = result.loc[result["grid_id"] == "inside", "ndbi_mean"].iloc[0]
    outside = result.loc[result["grid_id"] == "far_outside", "ndbi_mean"].iloc[0]
    assert inside == pytest.approx(0.3)
    assert pd.isna(outside)


def test_compute_ndbi_raster_matches_formula_and_guards_zero_denominator(tmp_path):
    transform = from_origin(0, 10, 1, 1)
    # Pixel 0: nir=100, swir=300 -> (300-100)/(300+100) = 0.5
    # Pixel 1: nir=0,   swir=0   -> denom=0, guarded to 0 (not NaN/inf)
    nir = np.array([[100.0, 0.0]], dtype="float32")
    swir = np.array([[300.0, 0.0]], dtype="float32")
    nir_path, swir_path, out_path = tmp_path / "nir.tif", tmp_path / "swir.tif", tmp_path / "ndbi.tif"
    _write_raster(str(nir_path), nir, transform)
    _write_raster(str(swir_path), swir, transform)

    compute_ndbi_raster(str(nir_path), str(swir_path), str(out_path))

    with rasterio.open(str(out_path)) as src:
        result = src.read(1)
    np.testing.assert_allclose(result, np.array([[0.5, 0.0]]), rtol=1e-6)
