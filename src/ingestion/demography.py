"""Real BPS/Dukcapil kecamatan-level population data --
`data/demography/demography.csv` (nama_kecamatan, nama_desa,
jumlah_penduduk, kepadatan_per_km2).

Verified coverage: Ciracas/Pasar Rebo/Cipayung (Jakarta Timur), several
Bogor kecamatan, and Cimanggis/Tapos (Depok) -- ZERO overlap with the main
Dukuh Atas/Blok M study area's kecamatan (Setiabudi, Tanah Abang, Menteng,
Kebayoran Baru, Mampang Prapatan, Pal Merah). That's why this was left
unwired for the main pipeline (see run_pipeline.py's MAPID comment for the
same reasoning applied to a different data source) -- joining it there
would silently produce an all-null column, not a real feature.

It DOES genuinely cover the LRT Cibubur-Bogor corridor (Ciracas + Harjamukti
stations both sit in kecamatan this file has real data for) -- see
scripts/ingest_cibubur_bogor_corridor.py."""

import numpy as np
import pandas as pd
import geopandas as gpd


def join_demography(grid: gpd.GeoDataFrame, csv_path: str = "data/demography/demography.csv") -> gpd.GeoDataFrame:
    """Joins real kecamatan-level population/density onto `grid` via its
    `kecamatan` column (see src/preprocessing/districts.py
    tag_grid_with_kecamatan). The source file is desa-level; there's no
    desa-level boundary shapefile available to join at that finer
    granularity, so this aggregates up to kecamatan (population =
    kecamatan total; density = population-weighted mean across its desa,
    not a plain mean, so denser desa contribute proportionally more).

    Grid cells whose kecamatan isn't in this file (e.g. anywhere in the
    main Dukuh Atas/Blok M study area) get NULL, not a fabricated value --
    matches every other "honest gap" in this repo's data handling."""
    demo = pd.read_csv(csv_path)
    agg = (
        demo.groupby("nama_kecamatan")
        .apply(lambda g: pd.Series({
            "population": int(g["jumlah_penduduk"].sum()),
            "population_density_per_km2": float(np.average(g["kepadatan_per_km2"], weights=g["jumlah_penduduk"])),
        }), include_groups=False)
        .reset_index()
    )

    grid = grid.copy()
    merged = grid.merge(agg, left_on="kecamatan", right_on="nama_kecamatan", how="left")
    grid["population"] = merged["population"].values
    grid["population_density_per_km2"] = merged["population_density_per_km2"].values
    return grid
