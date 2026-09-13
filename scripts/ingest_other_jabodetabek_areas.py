"""Extends REAL structural grid data to the remaining ~101 real-but-out-of-
corridor rows in data/umkm_survey_v4.geojson (the ones NOT in the LRT
Cibubur-Bogor corridor -- see scripts/fix_survey_v4_station_distances.py
and scripts/ingest_cibubur_bogor_corridor.py for that separate batch).

These rows cluster into 5 real, distinct Jabodetabek areas (verified by
their own real GPS coordinates, not by the file's free-text area labels,
some of which were just "Indonesia"). Reference points, verified via
Wikipedia / OpenStreetMap Nominatim, not guessed:
  - TMII: real, operating LRT station (-6.292874, 106.880535)
  - Bojong Pondok Terong (Depok-selatan cluster): -6.4393081, 106.8020916
  - Kemiri Muka (Depok-pusat cluster): -6.3755262, 106.8369117 (a school --
    the nearest real, verifiable point Nominatim returned for this
    kelurahan; same "real place-level proxy" approach already used for
    Bojong Nangka/Sentul/Baranangsiang)
  - Cimahpar (Bogor-utara cluster): -6.5845680, 106.8271658
  - Blok M (blok-m-selatan cluster): reuses the existing, already-real
    Blok M station -- this cluster (Melawai/Pulo/Gandaria
    Utara/Gelora) sits just south of the main study area.

Run standalone: .venv/Scripts/python.exe -m scripts.ingest_other_jabodetabek_areas
"""

import pandas as pd
from scripts.ingest_cibubur_bogor_corridor import ingest_area
from src.persistence.writer import get_engine, ensure_schema

AREAS = [
    (
        "tmii",
        (106.870, -6.300, 106.890, -6.280),
        pd.DataFrame([{"station_id": "tmii", "district_name": "TMII", "lat": -6.292874, "lon": 106.880535}]),
    ),
    (
        "depokselatan",
        (106.790, -6.460, 106.815, -6.430),
        pd.DataFrame([{"station_id": "bojong_pondok_terong", "district_name": "Depok Selatan", "lat": -6.4393081, "lon": 106.8020916}]),
    ),
    (
        "depokpusat",
        (106.810, -6.400, 106.840, -6.370),
        pd.DataFrame([{"station_id": "kemiri_muka", "district_name": "Depok Pusat", "lat": -6.3755262, "lon": 106.8369117}]),
    ),
    (
        "bogorutara",
        (106.820, -6.600, 106.850, -6.560),
        pd.DataFrame([{"station_id": "cimahpar", "district_name": "Bogor Utara", "lat": -6.5845680, "lon": 106.8271658}]),
    ),
    (
        # North bound is -6.246, not -6.225 -- the original box overlapped
        # the main study area's bbox (south bound -6.245) by ~0.8km x 2.2km,
        # verified by it silently duplicating 27 of the 68 real v3 survey
        # rows (they fell inside both grids' cells at once). Real
        # boundaries between adjacent regions must not overlap.
        "blokmselatan",
        (106.785, -6.260, 106.805, -6.246),
        pd.DataFrame([{"station_id": "blok_m_selatan", "district_name": "Blok M Selatan", "lat": -6.2440, "lon": 106.7995}]),
    ),
]


def run() -> None:
    engine = get_engine()
    ensure_schema(engine)
    for id_prefix, bbox, stations in AREAS:
        ingest_area(id_prefix, bbox, stations, engine)


if __name__ == "__main__":
    run()
