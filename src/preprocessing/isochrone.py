"""Isochrone (walk-time service area) network analysis, per PRD:
'perhitungan radius keterjangkauan jalan kaki (Isochrone Network Analysis)
sejauh 0-800 meter atau setara 10 menit berjalan kaki dari simpul transit.'

compute_isochrone() is tested against a synthetic grid graph (networkx);
tag_grid_within_isochrone() is the piece run_pipeline.py actually calls,
against osm.fetch_walk_network()'s real graph, now that live network
access to the Overpass API is available from this environment (verified
directly before wiring this in -- see run_pipeline.py's ingestion stage)."""

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union


def compute_isochrone(graph: nx.Graph, station_node, max_distance_m: float = 800.0):
    """Returns a Shapely polygon approximating the walk-time service area
    from `station_node`, using Dijkstra shortest-path distances over the
    graph's `length` edge attribute (osmnx's standard convention)."""
    distances = nx.single_source_dijkstra_path_length(
        graph, station_node, cutoff=max_distance_m, weight="length"
    )
    reachable_nodes = [
        Point(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in distances.keys()
    ]
    if len(reachable_nodes) < 3:
        # Not enough reachable nodes for a polygon -- return a simple buffer instead.
        return Point(graph.nodes[station_node]["x"], graph.nodes[station_node]["y"]).buffer(
            max_distance_m / 111_000  # rough degrees-to-meters fallback, only hit in degenerate graphs
        )
    return MultiPoint(reachable_nodes).convex_hull


def tag_grid_within_isochrone(
    grid: gpd.GeoDataFrame,
    walk_graph: nx.Graph,
    stations: pd.DataFrame,
    max_distance_m: float = 800.0,
) -> gpd.GeoDataFrame:
    """Adds `within_walk_isochrone` (1/0) -- whether each cell's centroid
    falls inside the 800m/~10-min walk-network isochrone of ANY station,
    per the PRD's isochrone spec. This is a real network-distance
    catchment, not the straight-line `dist_to_station` districts.py
    already computes -- a cell can be close as the crow flies but outside
    the walk isochrone if the street network doesn't connect directly.

    `walk_graph` is unprojected (lat/lon degrees, osmnx's default), same
    as the isochrone polygons compute_isochrone() returns -- both get
    reprojected to `grid`'s CRS here so the containment check is valid."""
    station_nodes = ox.distance.nearest_nodes(walk_graph, stations["lon"].values, stations["lat"].values)
    isochrones = [compute_isochrone(walk_graph, node, max_distance_m) for node in station_nodes]
    isochrone_union = (
        gpd.GeoSeries([unary_union(isochrones)], crs="EPSG:4326").to_crs(grid.crs).iloc[0]
    )

    grid = grid.copy()
    grid["within_walk_isochrone"] = grid.geometry.centroid.within(isochrone_union).astype(int)
    return grid
