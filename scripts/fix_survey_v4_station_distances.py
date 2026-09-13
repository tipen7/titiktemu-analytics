"""One-off correction for data/umkm_survey_v4.geojson's `jarak_ke_stasiun_meter`
field.

Found by direct verification (not assumed): recomputing the real haversine
distance from each feature's own stated lat/lon to the real, Wikipedia-
verified Harjamukti station coordinate showed differences of hundreds of
meters against the file's stated values -- in one case (Kantin Sehat
Cibubur Point) the stated distance (57m) was smaller than the geometrically
IMPOSSIBLE minimum (straight-line distance was already 452m), proving the
field wasn't derived from the point's own coordinates.

SCOPE: this file spans ~28 distinct real Jabodetabek areas, far beyond the
LRT Cibubur-Bogor corridor this conversation has been about. This script
only recomputes rows whose `description` names one of the 6 real, verified
corridor reference points below -- everything else is left untouched
pending an explicit decision on how to handle it (see the "unscoped" rows
printed at the end).

Reference coordinates, all independently verified this session (Wikipedia /
OpenStreetMap Nominatim), not guessed:
  - Ciracas LRT station:      -6.3237,    106.8867    (Wikipedia)
  - Harjamukti LRT station:   -6.373988,  106.895623  (Wikipedia)
  - Bojong Nangka (village centroid -- the planned station itself isn't
    engineered/sited yet, see scripts/ingest_cibubur_bogor_corridor.py's
    docstring): -6.4299962, 106.9024879  (OSM Nominatim)
  - Sirkuit Internasional Sentul: -6.535861, 106.856778  (Wikipedia) --
    proxy for the "Sentul" planned station area
  - Baranangsiang:            -6.6098861, 106.8148447  (OSM Nominatim)

`jarak_ke_stasiun_meter` is set to the haversine distance to whichever of
these 5 points is geographically nearest -- not forced to match the row's
own area label -- since "distance to nearest station" is the actual
quantity this field represents, and nearest-of-all is more robust than
trusting free-text area labels.
"""

import json
import math

SURVEY_PATH = "data/umkm_survey_v4.geojson"

CORRIDOR_STATIONS = {
    "ciracas": (-6.3237, 106.8867),
    "harjamukti": (-6.373988, 106.895623),
    "bojong_nangka": (-6.4299962, 106.9024879),
    "sentul": (-6.535861, 106.856778),
    "baranangsiang": (-6.6098861, 106.8148447),
}

# Which free-text area labels (last comma-separated segment of `description`)
# count as "in the Cibubur-Bogor corridor" for this correction pass.
CORRIDOR_AREA_LABELS = {"Ciracas", "Harjamukti", "Bojong Nangka", "Cikeas Udik", "Sentul", "Baranangsiang"}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_station_distance_m(lat: float, lon: float) -> tuple[str, float]:
    dists = {name: haversine_m(lat, lon, slat, slon) for name, (slat, slon) in CORRIDOR_STATIONS.items()}
    nearest = min(dists, key=dists.get)
    return nearest, dists[nearest]


def run() -> None:
    with open(SURVEY_PATH, encoding="utf-8") as f:
        data = json.load(f)

    fixed, unscoped = 0, []
    for feature in data["features"]:
        props = feature["properties"]
        area_label = props.get("description", "").split(",")[-1].strip()
        if area_label not in CORRIDOR_AREA_LABELS:
            unscoped.append((props.get("name"), area_label))
            continue

        lat, lon = float(props["latitude"]), float(props["longitude"])
        nearest, dist_m = nearest_station_distance_m(lat, lon)
        old = props.get("jarak_ke_stasiun_meter")
        props["jarak_ke_stasiun_meter"] = round(dist_m, 1)
        props["nearest_station_verified"] = nearest  # audit trail -- which real reference point this used
        fixed += 1
        if old is not None and abs(float(old) - dist_m) > 50:
            print(f"  {props.get('name', '?'):40s} area={area_label:15s} old={old!s:>8} -> real={dist_m:7.1f}m (nearest: {nearest})")

    with open(SURVEY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"\nFixed {fixed} rows in the LRT Cibubur-Bogor corridor (real haversine distance to nearest verified station).")
    print(f"Left {len(unscoped)} rows untouched -- outside the corridor, area label not in {sorted(CORRIDOR_AREA_LABELS)}.")
    from collections import Counter
    print("Unscoped rows by area label:", Counter(label for _, label in unscoped).most_common())


if __name__ == "__main__":
    run()
