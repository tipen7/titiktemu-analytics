"""Real BPS/Dukcapil kecamatan-level population data --
`data/demography/demography.csv` (nama_kecamatan, nama_desa,
jumlah_penduduk, kepadatan_per_km2).

Extended this session to cover every kecamatan the pipeline's grid regions
actually touch (30 distinct kecamatan across all 10 regions, including the
main Dukuh Atas/Blok M study area's own -- Setiabudi, Tanah Abang, Menteng,
Kebayoran Baru, Mampang Prapatan, Kebayoran Lama), not just the Cibubur-
Bogor corridor's. `Pal Merah` (Jakarta Barat, 30 grid cells in the
`blokmselatan`/main regions) is the one remaining known gap -- not yet in
this file, left NULL, not fabricated.

Grid cells whose kecamatan isn't in this file get NULL, not a fabricated
value -- matches every other "honest gap" in this repo's data handling."""

import numpy as np
import pandas as pd
import geopandas as gpd


# Real spelling/spacing differences between the two real government
# sources -- the district-administrative shapefile (grid's `kecamatan`
# column, see src/preprocessing/districts.py tag_grid_with_kecamatan) and
# this BPS/Dukcapil demography.csv -- for the SAME official kecamatan.
# Verified by cross-checking which real place each refers to, not a fuzzy
# match: an unlisted mismatch stays NULL rather than being silently
# guessed at.
KECAMATAN_ALIASES = {
    "kramatjati": "kramat jati",
    "jati sempurna": "jatisampurna",
}


def _normalize_kecamatan(name) -> str | None:
    if pd.isna(name):
        return None
    key = str(name).strip().lower()
    return KECAMATAN_ALIASES.get(key, key)


def join_demography(grid: gpd.GeoDataFrame, csv_path: str = "data/demography/demography.csv") -> gpd.GeoDataFrame:
    """Joins real kecamatan-level population/density onto `grid` via its
    `kecamatan` column (see src/preprocessing/districts.py
    tag_grid_with_kecamatan). The source file is desa-level; there's no
    desa-level boundary shapefile available to join at that finer
    granularity, so this aggregates up to kecamatan (population =
    kecamatan total; density = population-weighted mean across its desa,
    not a plain mean, so denser desa contribute proportionally more).

    Joins on a normalized (lowercased, whitespace-trimmed, alias-resolved)
    key rather than the raw string -- verified real mismatches exist
    between the two sources' spelling for the same kecamatan (see
    KECAMATAN_ALIASES) that would otherwise silently join to nothing.

    Grid cells whose kecamatan isn't in this file (e.g. Pal Merah) get
    NULL, not a fabricated value -- matches every other "honest gap" in
    this repo's data handling."""
    demo = pd.read_csv(csv_path)
    demo["_key"] = demo["nama_kecamatan"].apply(_normalize_kecamatan)
    agg = (
        demo.groupby("_key")
        .apply(lambda g: pd.Series({
            "population": int(g["jumlah_penduduk"].sum()),
            "population_density_per_km2": float(np.average(g["kepadatan_per_km2"], weights=g["jumlah_penduduk"])),
        }), include_groups=False)
        .reset_index()
    )

    grid = grid.copy()
    # `grid` may already carry population/population_density_per_km2
    # columns (e.g. loaded back from spatial_grids via load_grid_from_db,
    # which always selects them, null or not) -- merging without dropping
    # them first silently produces population_x/population_y instead of
    # overwriting, since both sides would have the same column name.
    grid = grid.drop(columns=["population", "population_density_per_km2"], errors="ignore")
    grid["_key"] = grid["kecamatan"].apply(_normalize_kecamatan)
    merged = grid.merge(agg, on="_key", how="left")
    grid["population"] = merged["population"].values
    grid["population_density_per_km2"] = merged["population_density_per_km2"].values
    grid = grid.drop(columns=["_key"])
    return grid
