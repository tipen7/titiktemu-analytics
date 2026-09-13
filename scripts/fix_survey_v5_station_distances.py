"""One-off correction for data/umkm_survey_v5.geojson's `jarak_ke_stasiun_meter`
field -- the same fabrication pattern found and fixed in v4 (see
scripts/fix_survey_v4_station_distances.py / fix_survey_v4_remaining_distances.py)
recurs in this newer, otherwise-independently-verified export: recomputing
real haversine distance from each row's own stated lat/lon to the nearest
real reference point showed 137/149 rows off by >150m from the stated
value (median diff ~891m, max ~4.5km) -- including the exact same
"Hanamasa Cibubur" / "Pizza Hut Delivery" rows already caught in v4.
Verified this is independent of the rent/revenue correction the survey
owner already did for this file: rent and revenue in v5 are NOT the v4
mechanical `transaction_per_day * 30` relationship (checked: 0/23
comparable rows match that formula), so only this one field needed fixing
here.

Every one of v5's 149 rows falls inside a bbox already covered by the 9
real reference areas verified for v4 (checked: 0 unmatched rows) --
no new reference points needed, just reuse of the same verified list.
"""

import json
import math

SURVEY_PATH = "data/umkm_survey_v5.geojson"

STATIONS = {
    "ciracas": (-6.3237, 106.8867),
    "harjamukti": (-6.373988, 106.895623),
    "bojong_nangka": (-6.4299962, 106.9024879),
    "sentul": (-6.535861, 106.856778),
    "baranangsiang": (-6.6098861, 106.8148447),
    "tmii": (-6.292874, 106.880535),
    "bojong_pondok_terong": (-6.4393081, 106.8020916),
    "kemiri_muka": (-6.3755262, 106.8369117),
    "cimahpar": (-6.5845680, 106.8271658),
    "blok_m_selatan": (-6.2440, 106.7995),
    "dukuh_atas": (-6.1988, 106.8230),
    "blok_m": (-6.2440, 106.7995),
}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_station_distance_m(lat: float, lon: float) -> tuple[str, float]:
    dists = {name: haversine_m(lat, lon, slat, slon) for name, (slat, slon) in STATIONS.items()}
    nearest = min(dists, key=dists.get)
    return nearest, dists[nearest]


def run() -> None:
    with open(SURVEY_PATH, encoding="utf-8") as f:
        data = json.load(f)

    n_fixed = 0
    diffs = []
    for feature in data["features"]:
        props = feature["properties"]
        lat, lon = float(props["latitude"]), float(props["longitude"])
        nearest, dist_m = nearest_station_distance_m(lat, lon)
        old = float(props["jarak_ke_stasiun_meter"])
        diffs.append(abs(old - dist_m))
        props["jarak_ke_stasiun_meter"] = round(dist_m, 1)
        props["nearest_station_verified"] = nearest
        n_fixed += 1

    with open(SURVEY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    diffs.sort()
    median = diffs[len(diffs) // 2]
    print(f"Fixed {n_fixed} rows. Median correction: {median:.0f}m, max: {max(diffs):.0f}m.")


if __name__ == "__main__":
    run()
