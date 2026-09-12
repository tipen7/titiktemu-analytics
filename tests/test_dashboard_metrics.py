import json
from src.persistence.dashboard_metrics import compute_dashboard_metrics


def test_dashboard_metrics_are_json_serializable(scored_grid):
    metrics = compute_dashboard_metrics(scored_grid)
    json.dumps(metrics)  # regression guard for the tuple-key MultiIndex bug found earlier


def test_dashboard_metrics_counts_sum_to_total(scored_grid):
    metrics = compute_dashboard_metrics(scored_grid)
    total = metrics["danger_zone_count"] + metrics["moderate_zone_count"] + metrics["safe_zone_count"]
    assert total == metrics["total_grid_cells"]


def test_dashboard_metrics_by_district_present(scored_grid):
    metrics = compute_dashboard_metrics(scored_grid)
    assert set(metrics["by_district"].keys()) == {"Dukuh Atas", "Blok M"}
