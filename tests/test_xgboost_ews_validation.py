import numpy as np
import pytest
from src.modeling.xgboost_ews import (
    wilson_score_interval, validate_ews_against_survey, real_survey_ews_cutoffs, vulnerability_to_ews,
)


def test_wilson_interval_matches_known_value():
    # Textbook check: 8/10 successes, 95% Wilson CI is approx (0.492, 0.944)
    # (per standard reference tables for this exact n/k pair).
    low, high = wilson_score_interval(8, 10)
    assert low == pytest.approx(0.492, abs=0.01)
    assert high == pytest.approx(0.944, abs=0.01)


def test_wilson_interval_zero_n_is_zero():
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_stays_within_bounds():
    for successes, n in [(0, 5), (5, 5), (3, 7)]:
        low, high = wilson_score_interval(successes, n)
        assert 0.0 <= low <= high <= 1.0


def test_validate_ews_against_survey_perfect_match():
    true_vuln = np.array([0.1, 0.5, 0.9, 0.15, 0.55])
    pred_vuln = true_vuln.copy()  # LOOCV predicted == true -> 100% match

    result = validate_ews_against_survey(true_vuln, pred_vuln)

    assert result["n"] == 5
    assert result["accuracy_pct"] == 100.0
    assert result["confidence_level"] == "low"  # n=5 is well below the 30-point floor


def test_validate_ews_against_survey_uses_real_data_scale_not_grid_scale():
    # Regression guard for the actual bug found in production: real survey
    # vulnerability (~0-1 ratio scale) compared against GRID-based cutoffs
    # (which can sit at a totally different, extrapolation-skewed scale,
    # e.g. mostly clipped to 0) made every real point trivially land in the
    # same bucket regardless of whether predictions were actually accurate.
    # Cutoffs anchored to the real data's OWN scale must actually discriminate.
    true_vuln = np.array([0.05, 0.1, 0.5, 0.9, 0.95])  # spans low/mid/high on ITS OWN scale
    pred_vuln = np.array([0.95, 0.9, 0.5, 0.1, 0.05])  # exactly reversed -> should score badly, not 100%

    result = validate_ews_against_survey(true_vuln, pred_vuln)

    assert result["accuracy_pct"] < 100.0


def test_validate_ews_against_survey_drops_nan_pairs():
    true_vuln = np.array([0.1, np.nan, 0.9])
    pred_vuln = np.array([0.1, 0.5, np.nan])

    result = validate_ews_against_survey(true_vuln, pred_vuln)

    assert result["n"] == 1  # only index 0 has both real


def test_validate_ews_against_survey_confidence_level_thresholds():
    rng = np.random.default_rng(0)

    for n, expected_level in [(20, "low"), (50, "moderate"), (150, "high")]:
        true_vuln = rng.uniform(0, 1, n)
        pred_vuln = true_vuln.copy()
        result = validate_ews_against_survey(true_vuln, pred_vuln)
        assert result["confidence_level"] == expected_level


def test_real_survey_ews_cutoffs_are_terciles_of_input():
    vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    low, high = real_survey_ews_cutoffs(vals)
    assert low == pytest.approx(np.quantile(vals, 1 / 3))
    assert high == pytest.approx(np.quantile(vals, 2 / 3))


def test_vulnerability_to_ews_with_fixed_cutoffs():
    import pandas as pd
    vulnerability = pd.Series([0.0, 0.05, 0.5, 0.9, 100.0])
    result = vulnerability_to_ews(vulnerability, cutoffs=(0.2, 0.8))
    assert list(result) == [0, 0, 1, 2, 2]


def test_vulnerability_to_ews_degrades_gracefully_on_heavy_ties():
    # Regression guard: a series where most values are tied at one floor
    # (e.g. many grid cells clipped to 0 -- see gwr.predict_gwr_surface)
    # used to crash pd.qcut with "Bin edges must be unique" when no fixed
    # cutoffs were given.
    import pandas as pd
    vulnerability = pd.Series([0.0] * 20 + [0.5, 1.0])
    result = vulnerability_to_ews(vulnerability)  # no cutoffs -> self-adaptive fallback
    assert len(result) == len(vulnerability)
