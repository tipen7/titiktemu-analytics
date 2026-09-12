"""Isochrone (walk-time service area) network analysis, per PRD:
'perhitungan radius keterjangkauan jalan kaki (Isochrone Network Analysis)
sejauh 0-800 meter atau setara 10 menit berjalan kaki dari simpul transit.'

Tested here against a synthetic grid graph (networkx) rather than a real
OSM extract -- proves the graph-distance + buffer logic works; swap in
osm.fetch_walk_network()'s real graph once network access allows it."""

import networkx as nx
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
