import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from src.ingestion.demography import join_demography


def test_join_demography_matches_known_kecamatan(tmp_path):
    csv = tmp_path / "demography.csv"
    csv.write_text(
        "nama_kecamatan,nama_desa,jumlah_penduduk,kepadatan_per_km2\n"
        "Ciracas,Cibubur,100,200\n"
        "Ciracas,Susukan,300,400\n"
    )
    grid = gpd.GeoDataFrame(
        {"kecamatan": ["Ciracas"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326",
    )

    result = join_demography(grid, csv_path=str(csv))

    assert result.loc[0, "population"] == 400  # 100 + 300
    # Population-weighted mean density: (100*200 + 300*400) / 400 = 350
    assert result.loc[0, "population_density_per_km2"] == 350.0


def test_join_demography_leaves_unmatched_kecamatan_null(tmp_path):
    csv = tmp_path / "demography.csv"
    csv.write_text(
        "nama_kecamatan,nama_desa,jumlah_penduduk,kepadatan_per_km2\n"
        "Ciracas,Cibubur,100,200\n"
    )
    grid = gpd.GeoDataFrame(
        {"kecamatan": ["Setiabudi", None], "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)]},
        crs="EPSG:4326",
    )

    result = join_demography(grid, csv_path=str(csv))

    assert result["population"].isna().all()
    assert result["population_density_per_km2"].isna().all()
