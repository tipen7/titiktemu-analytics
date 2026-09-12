"""OSM POI density + walkable road network, per PRD:
'agregasi densitas komersial (POI count) dari OpenStreetMap' and the
isochrone network analysis input.

NOTE: osmnx calls the Overpass API over the network, which is not reachable
from this sandbox (not in its allowlist) -- this module is written against
osmnx's real, documented API and installs/imports cleanly, but the live
fetch itself is untested here. Run fetch_poi_counts() and fetch_walk_network()
for real once this repo runs somewhere with normal internet access."""

import geopandas as gpd
import osmnx as ox


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
