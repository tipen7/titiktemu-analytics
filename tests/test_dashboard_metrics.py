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


def test_dashboard_metrics_includes_surface_fit_when_given(scored_grid):
    metrics = compute_dashboard_metrics(scored_grid, ews_test_accuracy=0.987)
    assert metrics["xgboost_surface_fit_pct"] == 98.7


def test_dashboard_metrics_surface_fit_defaults_to_none(scored_grid):
    metrics = compute_dashboard_metrics(scored_grid)
    assert metrics["xgboost_surface_fit_pct"] is None


def test_dashboard_metrics_includes_ews_validation_when_given(scored_grid):
    validation = {
        "n": 35, "matches": 24, "accuracy_pct": 68.6,
        "ci_95_low_pct": 51.9, "ci_95_high_pct": 81.9, "confidence_level": "moderate",
        "exact_match_accuracy_pct": 51.4, "opposite_extreme_error_pct": 8.6,
    }
    metrics = compute_dashboard_metrics(scored_grid, ews_validation=validation)
    assert metrics["ews_validation_accuracy_pct"] == 68.6
    assert metrics["ews_validation_exact_match_accuracy_pct"] == 51.4
    assert metrics["ews_validation_opposite_extreme_error_pct"] == 8.6
    assert metrics["ews_validation_n"] == 35
    assert metrics["confidence_level"] == "moderate"


def test_dashboard_metrics_ews_validation_defaults_to_none(scored_grid):
    metrics = compute_dashboard_metrics(scored_grid)
    assert metrics["ews_validation_accuracy_pct"] is None
    assert metrics["confidence_level"] is None
