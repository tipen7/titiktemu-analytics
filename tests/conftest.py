import numpy as np
import pytest
from src.preprocessing.grid import build_study_grid
from src.preprocessing.districts import tag_grid_with_district


@pytest.fixture
def grid():
    g = build_study_grid()
    g = tag_grid_with_district(g)
    return g


@pytest.fixture
def scored_grid(grid):
    """A grid with fake-but-structured scores, for testing modules that
    consume scored output (zones, dashboard metrics) without needing a
    real GWR/XGBoost run."""
    rng = np.random.default_rng(1)
    grid = grid.copy()
    grid["poi_count"] = rng.integers(5, 60, len(grid))
    grid["ews_code"] = np.where(grid["district_name"] == "Blok M", 2, rng.choice([0, 0, 1], size=len(grid)))
    grid["vulnerability_index"] = rng.uniform(0, 1, len(grid))
    grid["matching_score"] = rng.uniform(0, 100, len(grid))
    return grid
