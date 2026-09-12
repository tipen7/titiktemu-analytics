"""ONE-OFF, TEMPORARY script -- generates a mock-filled version of the
survey CSV for pipeline testing, per explicit instruction: real complete
data is coming later, this is for testing only.

Deliberately kept OUT of src/ -- this is not part of the real cleaning
pipeline (src/ingestion/umkm_survey.py) and must never be imported by
run_pipeline.py. Mixing "generate fake data" into the real ingestion module
would risk it silently running against real data later.

What this does, per the exact instructions given:
  - Rows 1-41: only fills the 6 missing `rent_price_annual` cells, using the
    MEDIAN of the 35 real values in that column (central tendency, right-
    skewed financial field -> median, same rule as impute_missing()).
    Also fills the 4 columns blank across ALL of rows 1-41
    (tanggal_survey, kondisi_bangunan, moda_terhubung,
    jarak_ke_stasiun_meter) -- not explicitly requested, but left entirely
    null would break anything depending on them; flagged in the accompanying
    chat message, not silently done.
  - Rows 42-100: name + every other attribute is fully mocked (lat/long are
    the only real values, already fixed by fix_corrupted_coordinate()).

Every mocked cell is tracked in `is_mock_*` boolean columns in the output,
so nothing downstream can mistake mock data for real data by accident.
"""

import numpy as np
import pandas as pd
from src.ingestion.umkm_survey import load_raw_survey, parse_currency, parse_rent_with_period

RNG = np.random.default_rng(42)

TENANT_TYPES = ["umkm-tetap", "umkm-seasonal", "franchise-tetap", "franchise-seasonal"]
PRODUCTS = ["minuman (kopi, teh)", "makanan berat (nasi)", "makanan ringan", "dessert", "bakery"]
TARGET_MARKETS = ["pejalan kaki", "pekerja kantoran", "warga", "pejalan kaki/pekerja kantoran"]
BUILDING_CONDITIONS = ["baik", "sedang", "perlu renovasi"]
TRANSIT_MODES = ["MRT", "KRL", "TransJakarta", "MRT/TransJakarta"]
MOCK_NAME_PREFIXES = ["Warung", "Kedai", "Toko", "Kios", "Gerai"]
MOCK_NAME_SUFFIXES = ["Berkah", "Jaya", "Sejahtera", "Makmur", "Sentosa", "Mandiri", "Barokah"]


def mock_name(i: int) -> str:
    return f"{RNG.choice(MOCK_NAME_PREFIXES)} {RNG.choice(MOCK_NAME_SUFFIXES)} {i}"


def generate() -> pd.DataFrame:
    df = load_raw_survey("data/umkm_survey_raw.csv")
    df["is_mock_rent"] = False
    df["is_mock_new_fields"] = False
    df["is_mock_full_row"] = False

    first_41 = df["id"] <= 41
    rest = ~first_41

    # --- Rows 1-41: fill the 6 missing rent_price_annual with the median
    # of the real (annualized) values in that column (central tendency,
    # per instruction). Uses parse_rent_with_period since that's this
    # export's actual format ('13 juta/bulan' etc), not the old '13jt' one.
    rent_numeric = df.loc[first_41, "rent_price_annual"].apply(parse_rent_with_period)
    rent_median_raw = rent_numeric.median()
    missing_rent = first_41 & df["rent_price_annual"].isna()
    df.loc[missing_rent, "rent_price_annual"] = f"{int(rent_median_raw)}"
    df.loc[missing_rent, "is_mock_rent"] = True

    # --- Rows 1-41: the 4 columns blank across the board -- realistic mock fill.
    n1 = first_41.sum()
    df["jarak_ke_stasiun_meter"] = df["jarak_ke_stasiun_meter"].astype(object)
    df.loc[first_41, "tanggal_survey"] = pd.date_range("2026-07-01", periods=n1, freq="D").strftime("%Y-%m-%d")
    df.loc[first_41, "kondisi_bangunan"] = RNG.choice(BUILDING_CONDITIONS, n1)
    df.loc[first_41, "moda_terhubung"] = RNG.choice(TRANSIT_MODES, n1)
    df.loc[first_41, "jarak_ke_stasiun_meter"] = RNG.integers(50, 800, n1)
    df.loc[first_41, "is_mock_new_fields"] = True

    # --- Rows 42-100: fully mocked except lat/long (already real+fixed).
    n2 = rest.sum()
    idx = df.index[rest]
    df.loc[idx, "name"] = [mock_name(i) for i in df.loc[idx, "id"]]
    df.loc[idx, "description (produk)"] = RNG.choice(PRODUCTS, n2)
    df.loc[idx, "tenant_type (umkm/franchise-tetap/seasonal)"] = RNG.choice(TENANT_TYPES, n2)
    df.loc[idx, "period (tahun/bulan/minggu)"] = [f"{RNG.integers(1,3)} tahun" for _ in range(n2)]
    df.loc[idx, "transaction_per_day (ribu/juta/milliar)"] = [f"{RNG.integers(200,2000)} ribu" for _ in range(n2)]
    df.loc[idx, "tenant_area (meters squared)"] = [f"{RNG.integers(2,6)} m x {RNG.integers(2,6)} m" for _ in range(n2)]
    df.loc[idx, "target_market (pejalan kaki, pekerja kantoran, warga)"] = RNG.choice(TARGET_MARKETS, n2)
    df.loc[idx, "revenue_per_month"] = [f"{RNG.integers(5,60)} juta" for _ in range(n2)]
    df.loc[idx, "transaction_per_buyer"] = [f"{RNG.integers(10,100)} ribu" for _ in range(n2)]
    df.loc[idx, "rent_trend"] = [f"{RNG.integers(0,15)}%" for _ in range(n2)]
    df.loc[idx, "rent_price_annual"] = [f"{RNG.integers(5,80)}jt" for _ in range(n2)]
    df.loc[idx, "rent_expiry_date"] = pd.date_range("2027-01-01", periods=n2, freq="7D").strftime("%Y-%m-%d")
    df.loc[idx, "tanggal_survey"] = pd.date_range("2026-08-01", periods=n2, freq="D").strftime("%Y-%m-%d")
    df.loc[idx, "kondisi_bangunan"] = RNG.choice(BUILDING_CONDITIONS, n2)
    df.loc[idx, "moda_terhubung"] = RNG.choice(TRANSIT_MODES, n2)
    df.loc[idx, "jarak_ke_stasiun_meter"] = RNG.integers(50, 800, n2)
    df.loc[idx, "is_mock_full_row"] = True

    return df


if __name__ == "__main__":
    result = generate()
    result.to_csv("data/umkm_survey_mock.csv", index=False)
    print(f"Wrote {len(result)} rows to data/umkm_survey_mock.csv")
    print(f"  is_mock_rent: {result['is_mock_rent'].sum()}")
    print(f"  is_mock_new_fields: {result['is_mock_new_fields'].sum()}")
    print(f"  is_mock_full_row: {result['is_mock_full_row'].sum()}")
