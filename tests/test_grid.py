def test_grid_cells_are_exactly_250x250m(grid):
    assert all(abs(area - 62500) < 1 for area in grid.geometry.area)


def test_grid_covers_expected_cell_count(grid):
    # Regression guard -- catches accidental bbox/cell-size changes.
    assert 380 <= len(grid) <= 400


def test_grid_district_tagging_covers_both_stations(grid):
    assert set(grid["district_name"].unique()) == {"Dukuh Atas", "Blok M"}
    assert grid["dist_to_station"].min() >= 0


def test_grid_id_prefix_avoids_collisions_across_study_areas():
    # A second corridor built with a different id_prefix (see
    # scripts/ingest_cibubur_bogor_corridor.py) must never produce grid_ids
    # that collide with the main study area's default "grid_" ids when both
    # are persisted to the same spatial_grids table.
    from src.preprocessing.grid import build_study_grid
    main = build_study_grid()
    other = build_study_grid(bbox_wgs84=(106.8650, -6.4050, 106.9150, -6.2950), id_prefix="cibubur")
    assert all(gid.startswith("grid_") for gid in main["grid_id"])
    assert all(gid.startswith("cibubur_") for gid in other["grid_id"])
    assert set(main["grid_id"]).isdisjoint(set(other["grid_id"]))
