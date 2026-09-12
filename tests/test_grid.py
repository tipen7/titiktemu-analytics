def test_grid_cells_are_exactly_250x250m(grid):
    assert all(abs(area - 62500) < 1 for area in grid.geometry.area)


def test_grid_covers_expected_cell_count(grid):
    # Regression guard -- catches accidental bbox/cell-size changes.
    assert 380 <= len(grid) <= 400


def test_grid_district_tagging_covers_both_stations(grid):
    assert set(grid["district_name"].unique()) == {"Dukuh Atas", "Blok M"}
    assert grid["dist_to_station"].min() >= 0
