"""One-off backfill for the new umkm_businesses table (see
src/persistence/writer.py's schema note) against grid data that's ALREADY
persisted from the last full run_pipeline.py run -- no GWR refit or live
OSM/WorldCover ingestion needed, just the survey-loading + join logic
run_pipeline.py already does, against grids loaded back from the DB.

Run standalone: .venv/Scripts/python.exe -m scripts.backfill_umkm_businesses

Future full pipeline runs write this table automatically (see
run_pipeline.py's persistence step) -- this script exists only because
re-running the full pipeline just to populate one new table would cost
another 30-60 minutes of LOOCV for no modeling benefit.
"""

import pandas as pd
from src.preprocessing.grid import build_study_grid
from src.preprocessing.districts import tag_grid_with_district, tag_grid_with_kecamatan
from src.ingestion.umkm_survey_v4 import load_v4_survey
from src.persistence.writer import get_engine, ensure_schema, load_grid_from_db, write_umkm_businesses
import run_pipeline as rp


def run() -> None:
    engine = get_engine()
    ensure_schema(engine)

    print("Loading main study area grid from DB (already ingested)...")
    main_grid = load_grid_from_db(engine, "grid_")
    if main_grid.empty:
        # The main grid is written with plain "grid_XXX_YYY" ids -- if a
        # fresh DB has never had run_pipeline.py's main stage run against
        # it, fall back to rebuilding the grid+tagging (cheap, no network
        # calls) so this script still works standalone.
        print("  (not found in DB -- rebuilding grid + district/kecamatan tagging locally)")
        main_grid = build_study_grid()
        main_grid["x"] = main_grid.geometry.centroid.x
        main_grid["y"] = main_grid.geometry.centroid.y
        main_grid = tag_grid_with_district(main_grid)
        main_grid = tag_grid_with_kecamatan(main_grid)

    print("Loading extra region grids from DB (already ingested)...")
    corridor_grids = [load_grid_from_db(engine, prefix) for prefix in rp.CORRIDOR_GRID_PREFIXES]
    combined_grid = pd.concat([main_grid] + corridor_grids, ignore_index=True)
    print(f"  Combined grid: {len(combined_grid)} cells across {combined_grid['grid_id'].map(rp.region_of).nunique()} regions.")

    print("Loading real survey data (v3 + v4) and joining to grid...")
    main_survey = rp.load_umkm_survey(combined_grid)
    v4_survey = load_v4_survey(combined_grid)
    print(f"  v3: {len(main_survey)} rows, v4: {len(v4_survey)} rows.")

    print("Writing umkm_businesses...")
    write_umkm_businesses(engine, main_survey, source="v3")
    write_umkm_businesses(engine, v4_survey, source="v4")
    print(f"Done. {len(main_survey.dropna(subset=['grid_id']))} + "
          f"{len(v4_survey.dropna(subset=['grid_id']))} real UMKM business records persisted.")


if __name__ == "__main__":
    run()
