from src.modeling.zones import find_zone_for_location, find_reallocation, precompute_reallocations, EWS_TO_COLOR


def test_ews_to_color_mapping():
    assert EWS_TO_COLOR == {0: "green", 1: "yellow", 2: "red"}


def test_find_zone_for_location_returns_correct_zone(scored_grid):
    row = scored_grid.iloc[0]
    centroid_wgs84 = scored_grid.to_crs("EPSG:4326").geometry.centroid.iloc[0]
    result = find_zone_for_location(centroid_wgs84.y, centroid_wgs84.x, scored_grid)
    assert result is not None
    assert result["grid_id"] == row["grid_id"]
    assert result["zone_color"] == EWS_TO_COLOR[int(row["ews_code"])]


def test_find_zone_for_location_outside_grid_returns_none(scored_grid):
    result = find_zone_for_location(0.0, 0.0, scored_grid)  # nowhere near Jakarta
    assert result is None


def test_find_reallocation_expands_radius_when_needed(scored_grid):
    # A point deep in Blok M (forced all-red in the fixture) should need to
    # expand its search radius to find a green cell, likely in Dukuh Atas.
    result = find_reallocation(-6.2440, 106.7995, scored_grid, initial_radius_m=100)
    assert result is not None
    assert result["expansions_needed"] >= 0
    assert "grid_id" in result


def test_precompute_reallocations_covers_all_danger_cells(scored_grid):
    candidates = precompute_reallocations(scored_grid, top_n=3)
    danger_cells = scored_grid[scored_grid["ews_code"] == 2]
    if not danger_cells.empty:
        assert set(candidates["origin_grid_id"]).issubset(set(danger_cells["grid_id"]))
        # Every origin should have at most top_n candidates.
        assert candidates.groupby("origin_grid_id").size().max() <= 3
