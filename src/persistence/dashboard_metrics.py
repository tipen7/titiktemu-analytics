"""TODO 3: Operator dashboard aggregate metrics -- computed once per
pipeline run (not per dashboard page-load, per our earlier 'batch not
live' architecture decision) and written to a summary table the backend
reads directly, same pattern as gentrification_risk_scores."""

import pandas as pd
from sqlalchemy import text
from src.modeling.zones import SAFE_EWS_CODE


def compute_dashboard_metrics(scored_grid: pd.DataFrame, survey: pd.DataFrame | None = None) -> dict:
    """Returns the base set of operator-facing metrics. `survey` is
    optional -- some metrics (tenant-level counts) need the UMKM survey
    joined to grid scores, not just the grid itself."""
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
        "methodology_note": "EWS classification is XGBoost approximating the GWR-fitted "
                             "vulnerability surface for fast serving, not an independently "
                             "validated classifier -- do not present it as one in dashboard copy.",
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
