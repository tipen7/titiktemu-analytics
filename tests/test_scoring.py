from src.workers.scoring import score_point_against_grid


def test_score_point_against_grid_safe_zone_has_no_reallocation(scored_grid):
    safe_row = scored_grid[scored_grid["ews_code"] == 0].iloc[0]
    centroid_wgs84 = scored_grid.to_crs("EPSG:4326").geometry.centroid
    idx = scored_grid.index.get_loc(safe_row.name)
    point = centroid_wgs84.iloc[idx]

    result = score_point_against_grid(point.y, point.x, scored_grid)

    assert result["status"] == "scored"
    assert result["grid_id"] == safe_row["grid_id"]
    assert "reallocation" not in result


def test_score_point_against_grid_danger_zone_includes_reallocation(scored_grid):
    danger_row = scored_grid[scored_grid["ews_code"] == 2].iloc[0]
    centroid_wgs84 = scored_grid.to_crs("EPSG:4326").geometry.centroid
    idx = scored_grid.index.get_loc(danger_row.name)
    point = centroid_wgs84.iloc[idx]

    result = score_point_against_grid(point.y, point.x, scored_grid)

    assert result["status"] == "scored"
    assert result["ews_code"] == 2
    assert "reallocation" in result


def test_score_point_against_grid_outside_study_area(scored_grid):
    result = score_point_against_grid(0.0, 0.0, scored_grid)  # nowhere near Jakarta
    assert result == {"status": "outside_study_area", "latitude": 0.0, "longitude": 0.0}
