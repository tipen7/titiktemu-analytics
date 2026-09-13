import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point
from src.preprocessing.isochrone import compute_isochrone, tag_grid_within_isochrone


def _lattice_graph(n=5, spacing=100.0):
    """An n x n grid graph, orthogonal edges only (no diagonals) -- shortest
    path distance from (0, 0) to (i, j) is exactly (i + j) * spacing, which
    makes expected reachability trivial to compute for assertions below."""
    g = nx.Graph()
    for i in range(n):
        for j in range(n):
            g.add_node((i, j), x=i * spacing, y=j * spacing)
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                g.add_edge((i, j), (i + 1, j), length=spacing)
            if j + 1 < n:
                g.add_edge((i, j), (i, j + 1), length=spacing)
    return g


def test_compute_isochrone_includes_reachable_excludes_unreachable():
    spacing = 100.0
    g = _lattice_graph(n=5, spacing=spacing)
    iso = compute_isochrone(g, station_node=(0, 0), max_distance_m=2.5 * spacing)

    # (1, 1): graph distance 2*spacing = 200 <= 250 -- reachable.
    reachable_pt = Point(1 * spacing, 1 * spacing)
    # (4, 4): graph distance 8*spacing = 800, far beyond cutoff -- unreachable.
    unreachable_pt = Point(4 * spacing, 4 * spacing)

    assert iso.buffer(1e-6).contains(reachable_pt)
    assert not iso.contains(unreachable_pt)


def test_compute_isochrone_area_grows_with_max_distance():
    g = _lattice_graph(n=6, spacing=100.0)
    small = compute_isochrone(g, station_node=(0, 0), max_distance_m=150.0)
    large = compute_isochrone(g, station_node=(0, 0), max_distance_m=450.0)
    assert large.area > small.area


def test_compute_isochrone_falls_back_to_buffer_for_sparse_reachability():
    g = _lattice_graph(n=5, spacing=100.0)
    # Cutoff smaller than a single edge -- only the station node itself is
    # reachable, so compute_isochrone must take the <3-node buffer fallback
    # rather than trying (and failing) to build a hull from 1 point.
    iso = compute_isochrone(g, station_node=(0, 0), max_distance_m=10.0)

    assert iso.geom_type == "Polygon"
    station_point = Point(g.nodes[(0, 0)]["x"], g.nodes[(0, 0)]["y"])
    assert iso.contains(station_point)


def _lonlat_lattice_graph(n=5, step_deg=0.0009, origin=(106.80, -6.20), edge_length_m=100.0):
    """Same lattice as _lattice_graph, but with real lon/lat-scale node
    coordinates (~100m/step near Jakarta's latitude) -- needed here since
    tag_grid_within_isochrone reprojects assuming EPSG:4326 node coords,
    unlike compute_isochrone's other tests which use arbitrary meter offsets."""
    origin_lon, origin_lat = origin
    # MultiDiGraph with flat integer node ids -- ox.distance.nearest_nodes()
    # converts via convert.graph_to_gdfs() internally, which requires both
    # the graph type AND flat (non-tuple) node ids real osmnx graphs use;
    # tuple node ids like compute_isochrone's other tests use break its
    # internal (u, v, key) MultiIndex construction.
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"

    def node_id(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            g.add_node(node_id(i, j), x=origin_lon + i * step_deg, y=origin_lat + j * step_deg)
    for i in range(n):
        for j in range(n):
            # Both directions -- real osmnx walk networks are reciprocal;
            # a MultiDiGraph with only one-way edges would make Dijkstra
            # unable to walk outward from the station in every direction.
            if i + 1 < n:
                g.add_edge(node_id(i, j), node_id(i + 1, j), length=edge_length_m)
                g.add_edge(node_id(i + 1, j), node_id(i, j), length=edge_length_m)
            if j + 1 < n:
                g.add_edge(node_id(i, j), node_id(i, j + 1), length=edge_length_m)
                g.add_edge(node_id(i, j + 1), node_id(i, j), length=edge_length_m)
    return g


def test_tag_grid_within_isochrone_marks_near_and_far_cells_correctly():
    origin = (106.80, -6.20)
    g = _lonlat_lattice_graph(n=5, origin=origin, edge_length_m=100.0)
    stations = pd.DataFrame([{"lon": origin[0], "lat": origin[1]}])

    step_deg = 0.0009
    # The 250m isochrone from a 4-directional lattice is a diamond (convex
    # hull of the reachable-within-250m nodes) -- 0.5 steps out in each
    # direction is safely inside it, well clear of the hull's edges (unlike
    # a point sitting exactly on a reachable node's own position, which can
    # land right on the hull boundary and fail strict `.within()`).
    near_point = Point(origin[0] + 0.5 * step_deg, origin[1] + 0.5 * step_deg)
    far_point = Point(origin[0] + 4 * step_deg, origin[1] + 4 * step_deg)  # 8 hops = 800m

    grid = gpd.GeoDataFrame(
        {"grid_id": ["near", "far"]},
        geometry=[near_point.buffer(0.0001), far_point.buffer(0.0001)],
        crs="EPSG:4326",
    ).to_crs("EPSG:32748")

    result = tag_grid_within_isochrone(grid, g, stations, max_distance_m=250.0)

    assert result.loc[result["grid_id"] == "near", "within_walk_isochrone"].iloc[0] == 1
    assert result.loc[result["grid_id"] == "far", "within_walk_isochrone"].iloc[0] == 0
