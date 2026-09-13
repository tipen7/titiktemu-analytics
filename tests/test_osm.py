from unittest.mock import patch
import geopandas as gpd
import pytest
from shapely.geometry import Point, box
from src.ingestion.osm import fetch_poi_counts, fetch_chain_poi_counts

CRS = "EPSG:32748"
FAKE_BBOX = (106.79, -6.25, 106.83, -6.19)


def test_fetch_poi_counts_aggregates_points_per_cell():
    grid = gpd.GeoDataFrame(
        {"grid_id": ["g1", "g2"]},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
        crs=CRS,
    )
    fake_pois = gpd.GeoDataFrame(geometry=[Point(10, 10), Point(20, 20), Point(150, 50)], crs=CRS)

    with patch("src.ingestion.osm.ox.features_from_bbox", return_value=fake_pois):
        result = fetch_poi_counts(grid, bbox_wgs84=FAKE_BBOX)

    counts = dict(zip(result["grid_id"], result["poi_count"]))
    assert counts["g1"] == 2
    assert counts["g2"] == 1


def test_fetch_poi_counts_defaults_to_zero_when_no_pois_in_cell():
    grid = gpd.GeoDataFrame(
        {"grid_id": ["g1", "g2"]},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
        crs=CRS,
    )
    fake_pois = gpd.GeoDataFrame(geometry=[Point(10, 10)], crs=CRS)

    with patch("src.ingestion.osm.ox.features_from_bbox", return_value=fake_pois):
        result = fetch_poi_counts(grid, bbox_wgs84=FAKE_BBOX)

    counts = dict(zip(result["grid_id"], result["poi_count"]))
    assert counts["g2"] == 0
    assert result["poi_count"].dtype.kind in "iu"  # stays integer, not NaN-coerced float


def test_fetch_chain_poi_counts_requires_poi_count_column():
    grid = gpd.GeoDataFrame({"grid_id": ["g1"]}, geometry=[box(0, 0, 100, 100)], crs=CRS)
    with pytest.raises(ValueError, match="poi_count"):
        fetch_chain_poi_counts(grid, bbox_wgs84=FAKE_BBOX)


def test_fetch_chain_poi_counts_computes_count_and_ratio():
    grid = gpd.GeoDataFrame(
        {"grid_id": ["g1", "g2"], "poi_count": [4, 0]},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
        crs=CRS,
    )
    fake_chains = gpd.GeoDataFrame(geometry=[Point(10, 10), Point(20, 20)], crs=CRS)  # both in g1

    with patch("src.ingestion.osm.ox.features_from_bbox", return_value=fake_chains):
        result = fetch_chain_poi_counts(grid, bbox_wgs84=FAKE_BBOX)

    counts = dict(zip(result["grid_id"], result["chain_poi_count"]))
    ratios = dict(zip(result["grid_id"], result["chain_ratio"]))
    assert counts["g1"] == 2
    assert counts["g2"] == 0
    assert ratios["g1"] == pytest.approx(0.5)  # 2 chains / 4 total POIs
    assert ratios["g2"] == 0.0  # no commercial POIs at all -> 0, not NaN
