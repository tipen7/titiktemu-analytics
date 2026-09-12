"""Builds the 250x250m analysis grid over the study area, per PRD Section
'GIS & Spatial Analytics Pipeline'. Grid cells are generated in the projected
CRS (EPSG:32748 / UTM Zone 48S) so that cell_size_m is a real metric size,
then geometry is kept in that CRS for all internal spatial math -- only
converted to EPSG:4326 at the point of serving GeoJSON to the client
(matches the PRD's ST_Transform-on-read pattern)."""

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from src.config import settings


def build_study_grid(
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    cell_size_m: float | None = None,
    crs: str | None = None,
) -> gpd.GeoDataFrame:
    """bbox_wgs84 = (west, south, east, north) in EPSG:4326.
    Returns a GeoDataFrame with columns [grid_id, geometry], geometry in `crs`."""
    west, south, east, north = bbox_wgs84 or (
        settings.study_area_bbox_west,
        settings.study_area_bbox_south,
        settings.study_area_bbox_east,
        settings.study_area_bbox_north,
    )
    cell_size = cell_size_m or settings.grid_cell_size_m
    target_crs = crs or settings.projected_crs

    study_area = gpd.GeoDataFrame(
        geometry=[box(west, south, east, north)], crs="EPSG:4326"
    ).to_crs(target_crs)

    minx, miny, maxx, maxy = study_area.total_bounds

    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    cells = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cells.append({
                "grid_id": f"grid_{i:03d}_{j:03d}",
                "geometry": box(x, y, x + cell_size, y + cell_size),
            })

    grid = gpd.GeoDataFrame(cells, crs=target_crs)
    # Keep only cells that actually intersect the study area (the bbox->UTM
    # reprojection produces a non-axis-aligned box, so the raw x/y range
    # over-covers a rotated rectangle).
    grid = grid[grid.intersects(study_area.union_all())].reset_index(drop=True)
    return grid
