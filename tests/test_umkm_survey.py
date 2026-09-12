import pandas as pd
from src.ingestion.umkm_survey import (
    fix_corrupted_coordinate, parse_rent_with_period, split_transaction_per_day,
    normalize_tenant_type, parse_period_months,
)


def test_fix_corrupted_coordinate_reconstructs_known_value():
    # Real value from the actual survey file, verified by hand against
    # neighboring rows -- regression guard against re-breaking the parser.
    assert abs(fix_corrupted_coordinate("-6.202.065", integer_digits=1) - (-6.202065)) < 1e-9
    assert abs(fix_corrupted_coordinate("-62.017.784.033.739.400", integer_digits=1) - (-6.2017784033739400)) < 1e-6
    assert abs(fix_corrupted_coordinate("-6,202313", integer_digits=1) - (-6.202313)) < 1e-9


def test_fix_corrupted_coordinate_handles_missing():
    assert fix_corrupted_coordinate(None, integer_digits=1) is None


def test_parse_rent_with_period_annualizes_correctly():
    assert parse_rent_with_period("13 juta/bulan") == 13_000_000 * 12
    assert parse_rent_with_period("160 ribu/hari") == 160_000 * 365
    assert parse_rent_with_period("2 juta/5 hari") == 2_000_000 * (365 / 5)
    assert parse_rent_with_period("100 juta/tahun") == 100_000_000
    assert parse_rent_with_period("50 juta") == 50_000_000  # bare -> already-annual
    assert parse_rent_with_period(None) is None


def test_split_transaction_per_day_triple():
    result = split_transaction_per_day("2 juta;900 ribu;300 ribu")
    assert result == {"txn_high": 2_000_000, "txn_normal": 900_000, "txn_low": 300_000}


def test_split_transaction_per_day_single():
    result = split_transaction_per_day("5 juta")
    assert result["txn_normal"] == 5_000_000
    assert result["txn_high"] is None


def test_normalize_tenant_type_known_values():
    assert normalize_tenant_type("umkm-tetap") == "umkm_tetap"
    assert normalize_tenant_type("franchise-seasonal") == "franchise_seasonal"


def test_normalize_tenant_type_unknown_returns_none_not_guess():
    assert normalize_tenant_type("something-weird") is None


def test_parse_period_months_handles_decimals():
    assert parse_period_months("1.5 tahun") == 18.0
    assert parse_period_months("7.5 bulan") == 7.5
