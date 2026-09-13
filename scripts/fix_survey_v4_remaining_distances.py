"""Second pass of the same correction as
scripts/fix_survey_v4_station_distances.py, extended to the 166 rows that
script deliberately left untouched (outside the LRT Cibubur-Bogor corridor
scope). Those 166 rows turned out to cluster into 5 more real Jabodetabek
areas (verified by their own real GPS coordinates, not the file's
sometimes-missing/ambiguous free-text area labels -- some rows'
`description` was just "Indonesia") -- see
scripts/ingest_other_jabodetabek_areas.py for the same verification trail.

Uses nearest-of-ALL-known-real-reference-points (not label matching) for
the same reason as the first pass: more robust than trusting free text.
"""

import json
import math

from scripts.fix_survey_v4_station_distances import CORRIDOR_STATIONS, haversine_m

SURVEY_PATH = "data/umkm_survey_v4.geojson"

OTHER_STATIONS = {
    "tmii": (-6.292874, 106.880535),
    "bojong_pondok_terong": (-6.4393081, 106.8020916),
    "kemiri_muka": (-6.3755262, 106.8369117),
    "cimahpar": (-6.5845680, 106.8271658),
    "blok_m_selatan": (-6.2440, 106.7995),
    # Main study area stations -- some of these 166 rows sit close enough
    # to Dukuh Atas/Blok M themselves (e.g. the Gelora/Senayan cluster) that
    # they may be nearer to these than to the new area-specific references.
    "dukuh_atas": (-6.1988, 106.8230),
    "blok_m": (-6.2440, 106.7995),
}

ALL_STATIONS = {**CORRIDOR_STATIONS, **OTHER_STATIONS}


def nearest_station_distance_m(lat: float, lon: float) -> tuple[str, float]:
    dists = {name: haversine_m(lat, lon, slat, slon) for name, (slat, slon) in ALL_STATIONS.items()}
    nearest = min(dists, key=dists.get)
    return nearest, dists[nearest]


def run() -> None:
    with open(SURVEY_PATH, encoding="utf-8") as f:
        data = json.load(f)

    fixed = 0
    for feature in data["features"]:
        props = feature["properties"]
        if "nearest_station_verified" in props:
            continue  # already fixed in the first pass (corridor rows)

        lat, lon = float(props["latitude"]), float(props["longitude"])
        nearest, dist_m = nearest_station_distance_m(lat, lon)
        props["jarak_ke_stasiun_meter"] = round(dist_m, 1)
        props["nearest_station_verified"] = nearest
        fixed += 1

    with open(SURVEY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Fixed {fixed} additional rows (real haversine distance to nearest verified station).")


if __name__ == "__main__":
    run()
