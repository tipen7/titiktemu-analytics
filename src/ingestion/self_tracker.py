"""UMKM Self-Tracker ingestion -- the PRD describes UMKM users self-reporting
data over time via the app (not just the one-time survey CSV). This reads
from a `self_tracker_submissions` table the backend writes to when a UMKM
user submits an update through the app.

NOT LIVE-TESTED against a real backend submission -- built and tested
against a synthetic table matching the expected shape, since no backend
self-tracker endpoint exists yet to submit real data through. Confirm
column names match once that endpoint is built."""

import pandas as pd
from sqlalchemy import text


def ensure_self_tracker_schema(engine) -> None:
    """Convenience for local dev/testing -- in production this table is
    owned by titiktemu-backend's migrations, same pattern as the other
    shared tables (see src/persistence/writer.py's ensure_schema note)."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS self_tracker_submissions (
                id                   SERIAL PRIMARY KEY,
                umkm_user_id         TEXT,
                grid_id              TEXT,
                latitude             NUMERIC,
                longitude            NUMERIC,
                revenue_per_month    NUMERIC,
                rent_price_annual    NUMERIC,
                submitted_at         TIMESTAMPTZ DEFAULT now()
            );
        """))


def load_self_tracker_submissions(engine, since=None) -> pd.DataFrame:
    """Pulls self-tracker submissions for inclusion in the next pipeline
    run. `since` (a timestamp) limits to submissions newer than the last
    pipeline run, so repeated runs don't reprocess the same rows forever --
    caller (run_pipeline.py) is responsible for tracking/passing that
    watermark; this function doesn't persist it itself."""
    query = "SELECT * FROM self_tracker_submissions"
    params = {}
    if since is not None:
        query += " WHERE submitted_at > :since"
        params["since"] = since
    return pd.read_sql(text(query), engine, params=params)


def merge_self_tracker_into_survey(survey: pd.DataFrame, submissions: pd.DataFrame) -> pd.DataFrame:
    """Self-tracker submissions use the SAME column shape as the cleaned
    survey DataFrame's core fields (revenue_per_month_idr, etc. -- renamed
    here to match), so they can be concatenated directly rather than
    needing a separate GWR-input path. Submissions from users NOT in the
    original 41+59 survey get appended as new rows; updates to existing
    UMKM (matched by umkm_user_id, if the survey gains that column later)
    are not yet handled -- straight append only for now."""
    if submissions.empty:
        return survey

    renamed = submissions.rename(columns={
        "revenue_per_month": "revenue_per_month_idr",
        "rent_price_annual": "rent_price_annual_idr",
    })
    renamed["id"] = renamed["umkm_user_id"]
    renamed["data_confidence"] = 1.0  # self-reported, treated as real unless later flagged otherwise

    common_cols = [c for c in survey.columns if c in renamed.columns]
    return pd.concat([survey, renamed[common_cols]], ignore_index=True)
