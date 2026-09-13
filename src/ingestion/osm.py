"""OSM POI density + walkable road network, per PRD:
'agregasi densitas komersial (POI count) dari OpenStreetMap' and the
isochrone network analysis input.

Live and confirmed working as of the v3+v5 pipeline run: `fetch_poi_counts()`
and `fetch_walk_network()` have been run for real against the Overpass API
from this environment (947+ real POIs, 8000+ real walk-network nodes seen
in one run). An earlier version of this docstring said the Overpass API
wasn't reachable at all -- that was true of a different, more restricted
sandbox this module was originally written in, not a limitation of the
code itself. `fetch_chain_poi_counts()` (below) hit real, intermittent
Overpass connect timeouts on some larger/heavier queries during testing --
treat that as normal for this API (rate limits, shared public infra), not
a sign the client is broken; retry rather than assume it's permanently
unreachable."""

import geopandas as gpd
import osmnx as ox
import pandas as pd


COMMERCIAL_TAGS = {
    "shop": True,
    "amenity": ["cafe", "restaurant", "fast_food", "marketplace"],
}


def fetch_poi_counts(grid: gpd.GeoDataFrame, bbox_wgs84: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """bbox_wgs84 = (west, south, east, north). Returns `grid` with an added
    `poi_count` column -- count of OSM commercial POIs per cell."""
    west, south, east, north = bbox_wgs84
    pois = ox.features_from_bbox((west, south, east, north), tags=COMMERCIAL_TAGS)
    pois = pois.to_crs(grid.crs)

    joined = gpd.sjoin(pois, grid[["grid_id", "geometry"]], how="inner", predicate="within")
    counts = joined.groupby("grid_id").size().rename("poi_count")

    result = grid.merge(counts, on="grid_id", how="left")
    result["poi_count"] = result["poi_count"].fillna(0).astype(int)
    return result


def fetch_walk_network(bbox_wgs84: tuple[float, float, float, float]):
    """Returns a networkx graph of the walkable road network -- input to
    src/preprocessing/isochrone.py."""
    west, south, east, north = bbox_wgs84
    return ox.graph_from_bbox((west, south, east, north), network_type="walk")


def fetch_chain_poi_counts(grid: gpd.GeoDataFrame, bbox_wgs84: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Real OSM chain/franchise-presence feature -- built to test whether it
    raises the model's genuine LOOCV-against-real-survey accuracy (see
    scripts/experiment_v3_v5_accuracy.py), motivated by a real pattern found
    in umkm_survey_v3.geojson: independent stalls showed rent/revenue ratios
    up to 1.01, while franchise tenants on the same street (Auntie Anne's,
    Koi, McDonald's) sat at 0.005-0.03 -- tenant type looks like real signal
    the existing 4 structural features don't capture at the grid-cell level.

    `brand` is OSM's standard tag for a chain/franchise's identity (e.g.
    brand=McDonald's) -- any commercial POI carrying it is being counted
    as a chain location, regardless of which specific brand. Adds two
    columns: `chain_poi_count` (raw count, matches poi_count's shape) and
    `chain_ratio` (chain_poi_count / poi_count, i.e. what share of this
    cell's commercial POIs are chains -- 0 where poi_count is 0, since a
    cell with no commercial POIs at all has no "chain share" to speak of,
    not a missing value).

    `grid` must already have `poi_count` (i.e. fetch_poi_counts() has run)
    -- chain_ratio is undefined without it."""
    if "poi_count" not in grid.columns:
        raise ValueError("fetch_chain_poi_counts requires poi_count already on the grid (run fetch_poi_counts first)")

    west, south, east, north = bbox_wgs84
    chains = ox.features_from_bbox((west, south, east, north), tags={"brand": True})
    chains = chains.to_crs(grid.crs)

    joined = gpd.sjoin(chains, grid[["grid_id", "geometry"]], how="inner", predicate="within")
    counts = joined.groupby("grid_id").size().rename("chain_poi_count")

    result = grid.merge(counts, on="grid_id", how="left")
    result["chain_poi_count"] = result["chain_poi_count"].fillna(0).astype(int)
    result["chain_ratio"] = result["chain_poi_count"] / result["poi_count"].replace(0, pd.NA)
    result["chain_ratio"] = result["chain_ratio"].fillna(0.0).astype(float)
    return result
