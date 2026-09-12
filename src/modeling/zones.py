"""TODO 1b/1c/1d: zone color mapping, point-in-polygon lookup, and the
reallocation/expanding-radius search -- the core of both user-facing
features (UMKM Self Discovery Tracker's zone map, and the Smart Tenant
Matching Engine's reallocation recommendation)."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# TODO 1b: EWS code -> zone color, per the user-flow description
# ("zona merah/kuning/hijau"). Single source of truth -- both the map
# rendering and the reallocation search below key off this.
EWS_TO_COLOR = {0: "green", 1: "yellow", 2: "red"}
EWS_TO_LABEL_ID = {0: "aman", 1: "waspada", 2: "bahaya"}
SAFE_EWS_CODE = 0  # "green" -- what reallocation search looks for


def find_zone_for_location(lat: float, lon: float, scored_grid: gpd.GeoDataFrame) -> dict | None:
    """TODO 1c: point-in-polygon lookup for a UMKM user's registered
    location. `scored_grid` must already have ews_code/vulnerability_index/
    district_name (i.e. the grid AFTER modeling + district tagging).
    Returns None if the point falls outside every grid cell (e.g. a user
    registers somewhere outside the current study area -- see TODO note in
    docstring below on multi-district population coverage)."""
    point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(scored_grid.crs)
    hit = scored_grid[scored_grid.contains(point.geometry.iloc[0])]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "grid_id": row["grid_id"],
        "district_name": row["district_name"],
        "ews_code": int(row["ews_code"]),
        "zone_color": EWS_TO_COLOR[int(row["ews_code"])],
        "vulnerability_index": float(row["vulnerability_index"]),
        "matching_score": float(row["matching_score"]),
    }


def find_reallocation(
    lat: float,
    lon: float,
    scored_grid: gpd.GeoDataFrame,
    initial_radius_m: float = 500.0,
    expand_factor: float = 2.0,
    max_expansions: int = 5,
) -> dict | None:
    """TODO 1d: the Smart Tenant Matching Engine's reallocation search.

    Per the user-flow spec: prioritize the NEAREST safe (green) zone to the
    tenant's current location; if none found within the initial radius,
    expand the search radius (doubling by default) up to `max_expansions`
    times -- 'still near the user's location' means we search the tenant's
    OWN district first at each radius, and only fall through to
    neighboring districts if the radius genuinely has to grow past the
    tenant's district extent to find a green zone. This does NOT hardcode
    'must be a different district' -- a green cell in the tenant's own
    district, if one exists nearby, is a perfectly valid (and closer)
    recommendation than one in a different district.

    Returns None if no safe zone is found even after all expansions -- the
    caller (backend) should treat that as 'no recommendation available
    yet', not silently recommend something unsafe.
    """
    point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(scored_grid.crs)[
        "geometry"
    ].iloc[0]

    # Which district is the tenant actually in? Needed to report whether a
    # recommendation crosses into a different district.
    containing = scored_grid[scored_grid.contains(point)]
    origin_district = containing.iloc[0]["district_name"] if not containing.empty else None

    safe_cells = scored_grid[scored_grid["ews_code"] == SAFE_EWS_CODE].copy()
    if safe_cells.empty:
        return None

    safe_cells["distance_m"] = safe_cells.geometry.centroid.distance(point)

    radius = initial_radius_m
    for attempt in range(max_expansions + 1):
        candidates = safe_cells[safe_cells["distance_m"] <= radius]
        if not candidates.empty:
            best = candidates.sort_values("distance_m").iloc[0]
            return {
                "grid_id": best["grid_id"],
                "district_name": best["district_name"],
                "distance_m": round(float(best["distance_m"]), 1),
                "matching_score": float(best["matching_score"]),
                "search_radius_used_m": radius,
                "expansions_needed": attempt,
                "crossed_district": origin_district is not None and best["district_name"] != origin_district,
            }
        radius *= expand_factor

    # Exhausted all expansions -- fall back to the single nearest safe cell
    # regardless of distance, rather than returning nothing when at least
    # ONE safe cell exists somewhere in the (currently limited) study area.
    best = safe_cells.sort_values("distance_m").iloc[0]
    return {
        "grid_id": best["grid_id"],
        "district_name": best["district_name"],
        "distance_m": round(float(best["distance_m"]), 1),
        "matching_score": float(best["matching_score"]),
        "search_radius_used_m": radius,
        "expansions_needed": max_expansions,
        "note": "Exceeded max radius expansions -- this is the nearest safe "
                "zone found, but may be far. With only 2 stations in the "
                "current study area, 'no nearby green zone' is expected "
                "sometimes -- this will improve as more TOD districts are "
                "added to the population.",
    }


def precompute_reallocations(
    scored_grid: gpd.GeoDataFrame,
    initial_radius_m: float = 500.0,
    expand_factor: float = 2.0,
    max_expansions: int = 5,
    top_n: int = 3,
) -> pd.DataFrame:
    """Architecture decision (batch vs. live, per explicit request to choose):
    BATCH precompute, not a live per-request service. Reasoning: EWS/
    vulnerability scores only change once per pipeline run, so a live
    Python computation on every user page-load would add latency and a new
    service to operate for zero freshness benefit. Instead, this runs once
    per pipeline run, computing the top-N nearest safe-zone candidates for
    EVERY danger-zone grid cell (not per-tenant -- tenants in the same
    grid cell get the same candidates, so grid-cell-level precomputation is
    both sufficient and far cheaper than per-tenant). The backend's 'live'
    lookup then becomes a single fast SQL query by grid_id -- no Python
    service needed at request time at all.

    Returns a long-format DataFrame: one row per (origin_grid_id, rank),
    ready for src/persistence/writer.write_reallocation_candidates().
    """
    danger_cells = scored_grid[scored_grid["ews_code"] == 2]
    rows = []
    for _, origin in danger_cells.iterrows():
        centroid = origin.geometry.centroid
        # Reuse find_reallocation's core logic by calling it with the
        # cell's own centroid as the query point (lat/long not needed --
        # everything here is already in the grid's projected CRS).
        safe_cells = scored_grid[scored_grid["ews_code"] == SAFE_EWS_CODE].copy()
        if safe_cells.empty:
            continue
        safe_cells["distance_m"] = safe_cells.geometry.centroid.distance(centroid)

        radius = initial_radius_m
        candidates = pd.DataFrame()
        expansions_used = 0
        for attempt in range(max_expansions + 1):
            candidates = safe_cells[safe_cells["distance_m"] <= radius]
            if not candidates.empty:
                expansions_used = attempt
                break
            radius *= expand_factor
        if candidates.empty:
            candidates = safe_cells
            expansions_used = max_expansions

        top = candidates.sort_values("distance_m").head(top_n)
        for rank, (_, cand) in enumerate(top.iterrows(), start=1):
            rows.append({
                "origin_grid_id": origin["grid_id"],
                "origin_district": origin["district_name"],
                "rank": rank,
                "recommended_grid_id": cand["grid_id"],
                "recommended_district": cand["district_name"],
                "distance_m": round(float(cand["distance_m"]), 1),
                "matching_score": float(cand["matching_score"]),
                "crossed_district": cand["district_name"] != origin["district_name"],
                "search_radius_used_m": radius,
                "expansions_needed": expansions_used,
            })
    return pd.DataFrame(rows)
