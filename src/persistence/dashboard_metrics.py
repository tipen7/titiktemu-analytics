"""TODO 3: Operator dashboard aggregate metrics -- computed once per
pipeline run (not per dashboard page-load, per our earlier 'batch not
live' architecture decision) and written to a summary table the backend
reads directly, same pattern as gentrification_risk_scores."""

import pandas as pd
from sqlalchemy import text
from src.modeling.zones import SAFE_EWS_CODE


def compute_dashboard_metrics(
    scored_grid: pd.DataFrame,
    survey: pd.DataFrame | None = None,
    ews_test_accuracy: float | None = None,
    ews_validation: dict | None = None,
) -> dict:
    """Returns the base set of operator-facing metrics. `survey` is
    optional -- some metrics (tenant-level counts) need the UMKM survey
    joined to grid scores, not just the grid itself.

    Two DIFFERENT accuracy numbers are persisted here, deliberately not
    collapsed into one -- conflating them is exactly what made the old
    single `ews_model_accuracy_pct` (~97-99%) read as an unrealistically
    confident, untrustworthy figure:

    - `xgboost_surface_fit_pct` (from `ews_test_accuracy`, train_xgboost_ews()'s
      held-out test-set accuracy, 0-1) measures how well XGBoost reproduces
      the GWR-fitted vulnerability surface for fast serving. It is near-100%
      by construction (GWR's bandwidth is near-global, so XGBoost is
      approximating an almost-linear function of its own inputs) and says
      NOTHING about real-world predictive skill -- internal/diagnostic only.
    - `ews_validation_*` (from `ews_validation`, see
      `xgboost_ews.validate_ews_against_survey`) is the genuine number: a
      leave-one-out cross-validated check of the pipeline's predicted risk
      zone against REAL, measured UMKM vulnerability at real survey points.
      This is the one that should be shown to a user/operator as "model
      accuracy" -- it comes with its own sample size and 95% CI so a thin
      sample (see CONTEXT.md Sec. 2/3) can't imply false precision.

    `confidence_level` here is derived from `ews_validation`'s SAMPLE SIZE,
    not from the accuracy percentage -- do not let a caller (e.g. the
    backend) re-derive it from the accuracy number instead; that was the
    other half of why the old figure looked untrustworthy (a high number
    was mechanically labeled "high confidence" regardless of how many real
    points backed it)."""
    total_cells = len(scored_grid)
    by_ews = scored_grid["ews_code"].value_counts().to_dict()

    metrics = {
        "total_grid_cells": total_cells,
        "danger_zone_count": int(by_ews.get(2, 0)),
        "moderate_zone_count": int(by_ews.get(1, 0)),
        "safe_zone_count": int(by_ews.get(0, 0)),
        "danger_zone_pct": round(100 * by_ews.get(2, 0) / total_cells, 1) if total_cells else 0,
        "avg_vulnerability_index": round(float(scored_grid["vulnerability_index"].mean()), 3),
        "avg_matching_score": round(float(scored_grid["matching_score"].mean()), 1),
        "xgboost_surface_fit_pct": round(ews_test_accuracy * 100, 1) if ews_test_accuracy is not None else None,
        "ews_validation_accuracy_pct": ews_validation["accuracy_pct"] if ews_validation else None,
        "ews_validation_n": ews_validation["n"] if ews_validation else None,
        "ews_validation_ci_95_low_pct": ews_validation["ci_95_low_pct"] if ews_validation else None,
        "ews_validation_ci_95_high_pct": ews_validation["ci_95_high_pct"] if ews_validation else None,
        "confidence_level": ews_validation["confidence_level"] if ews_validation else None,
        "methodology_note": "xgboost_surface_fit_pct measures XGBoost approximating the "
                             "GWR-fitted vulnerability surface for fast serving, not "
                             "real-world accuracy -- near-100% by construction, internal "
                             "diagnostic only. ews_validation_accuracy_pct is the genuine "
                             "figure: leave-one-out cross-validated against real, measured "
                             "UMKM survey vulnerability (n=ews_validation_n). Do not present "
                             "xgboost_surface_fit_pct as validated model accuracy in dashboard "
                             "copy.",
    }

    if "district_name" in scored_grid.columns:
        by_district = {}
        for district, group in scored_grid.groupby("district_name"):
            by_district[district] = {
                "danger": int((group["ews_code"] == 2).sum()),
                "moderate": int((group["ews_code"] == 1).sum()),
                "safe": int((group["ews_code"] == 0).sum()),
            }
        metrics["by_district"] = by_district

    if survey is not None and "grid_id" in survey.columns:
        tenant_grid = survey.merge(scored_grid[["grid_id", "ews_code"]], on="grid_id", how="left")
        metrics["tenants_needing_reallocation"] = int((tenant_grid["ews_code"] == 2).sum())
        metrics["total_tenants_tracked"] = len(tenant_grid)

    return metrics


def write_dashboard_metrics(engine, metrics: dict) -> None:
    import json
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dashboard_summary (
                id           SERIAL PRIMARY KEY,
                metrics      JSONB,
                computed_at  TIMESTAMPTZ DEFAULT now()
            );
        """))
        conn.execute(
            text("INSERT INTO dashboard_summary (metrics) VALUES (:metrics)"),
            {"metrics": json.dumps(metrics)},
        )
