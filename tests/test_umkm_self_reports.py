import pandas as pd

from src.ingestion.umkm_self_reports import clean_self_reports


def _row(**overrides):
    base = {
        "id": "r1",
        "business_name": "Warung Test",
        "category": "kuliner",
        "tenant_type": "umkm_tetap",
        "latitude": -6.2,
        "longitude": 106.8,
        "tenant_area_m2": 12.0,
        "rent_price_amount": 1_000_000.0,
        "rent_period_unit": "bulan",
        "revenue_per_month_idr": 20_000_000.0,
        "txn_high_idr": None,
        "txn_normal_idr": None,
        "txn_low_idr": None,
        "transaction_per_buyer_idr": 25_000.0,
        "rent_trend_pct": None,
        "status": "exported",
    }
    base.update(overrides)
    return base


def test_rent_annualized_from_structured_amount_and_period_no_text_parsing():
    df = pd.DataFrame([_row(rent_price_amount=1_000_000.0, rent_period_unit="bulan")])
    clean = clean_self_reports(df)
    assert clean.loc[0, "rent_price_annual_idr"] == 12_000_000.0


def test_rent_annualized_for_daily_and_yearly_periods():
    df = pd.DataFrame([
        _row(id="daily", rent_price_amount=100_000.0, rent_period_unit="hari"),
        _row(id="yearly", rent_price_amount=50_000_000.0, rent_period_unit="tahun"),
    ])
    clean = clean_self_reports(df)
    assert clean.loc[0, "rent_price_annual_idr"] == 100_000.0 * 365
    assert clean.loc[1, "rent_price_annual_idr"] == 50_000_000.0


def test_vulnerability_only_computed_when_both_rent_and_revenue_present():
    df = pd.DataFrame([
        _row(id="complete", rent_price_amount=1_000_000.0, rent_period_unit="bulan", revenue_per_month_idr=20_000_000.0),
        _row(id="missing_revenue", rent_price_amount=1_000_000.0, rent_period_unit="bulan", revenue_per_month_idr=None),
        _row(id="missing_rent", rent_price_amount=None, rent_period_unit=None, revenue_per_month_idr=20_000_000.0),
    ])
    clean = clean_self_reports(df)
    assert clean.loc[0, "vulnerability"] == 12_000_000.0 / (20_000_000.0 * 12)
    assert pd.isna(clean.loc[1, "vulnerability"])
    assert pd.isna(clean.loc[2, "vulnerability"])


def test_tenant_type_passed_through_unchanged_no_normalization_needed():
    # The self-report form's dropdown already emits TENANT_TYPE_MAP's
    # exact normalized values -- unlike the free-text survey files, there
    # is nothing here to normalize or guess.
    df = pd.DataFrame([_row(tenant_type="franchise_tetap")])
    clean = clean_self_reports(df)
    assert clean.loc[0, "tenant_type"] == "franchise_tetap"


def test_data_confidence_reflects_real_field_completeness():
    complete = pd.DataFrame([_row()])
    sparse = pd.DataFrame([_row(category=None, tenant_type=None, rent_price_amount=None, rent_period_unit=None)])
    clean_complete = clean_self_reports(complete)
    clean_sparse = clean_self_reports(sparse)
    assert clean_complete.loc[0, "data_confidence"] == 1.0
    # Only revenue_per_month_idr present out of the 4 completeness fields.
    assert clean_sparse.loc[0, "data_confidence"] == 0.25
