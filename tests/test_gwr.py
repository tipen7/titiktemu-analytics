import numpy as np
import pandas as pd
from src.modeling.gwr import fit_gwr, predict_gwr_surface


def test_fit_gwr_excludes_column_constant_across_real_training_rows():
    # Regression guard for the production bug found by actually running the
    # pipeline: within_walk_isochrone was 0 for every one of 35 real survey
    # rows. Jittering alone (see _jitter_coincident_covariates) avoided a
    # crash but produced an arbitrary, noise-driven coefficient that then
    # blew up under predict_gwr_surface's linear extrapolation to grid
    # cells where the column genuinely varies. fit_gwr must exclude a
    # truly-constant-in-training column from the regression entirely.
    rng = np.random.default_rng(0)
    n = 15
    df = pd.DataFrame({
        "x": rng.uniform(0, 1000, n),
        "y": rng.uniform(0, 1000, n),
        "varying_col": rng.uniform(0, 10, n),
        "constant_col": np.zeros(n),  # identical for every real row
        "target": rng.uniform(0, 1, n),
    })

    fit = fit_gwr(df, y_col="target", x_cols=["varying_col", "constant_col"], verbose=False)

    assert fit["used_x_cols"] == ["varying_col"]
    assert fit["dropped_x_cols"] == ["constant_col"]
    assert fit["params"].shape[1] == 2  # intercept + 1 real predictor, not 3


class _FakeModel:
    def __init__(self, coords):
        self.coords = coords


def test_predict_gwr_surface_clips_negative_extrapolation_to_zero():
    # Vulnerability is a rent/revenue ratio -- mathematically >= 0. A
    # negative prediction is always extrapolation overshoot, never signal.
    fit = {
        "model": _FakeModel([(0, 0), (10, 10)]),
        "params": np.array([[10.0, -5.0], [10.0, -5.0]]),  # intercept=10, coef=-5
        "used_x_cols": ["x1"],
    }
    new_X = pd.DataFrame({"x1": [10.0]})  # 10 + (-5 * 10) = -40 -> must clip to 0

    result = predict_gwr_surface(fit, [(5, 5)], new_X)

    assert result[0] == 0.0


def test_predict_gwr_surface_selects_used_x_cols_from_dataframe():
    # A caller passing the FULL feature set (including a column fit_gwr
    # excluded as constant) must not misalign against params, which only
    # has coefficients for used_x_cols.
    fit = {
        "model": _FakeModel([(0, 0)]),
        "params": np.array([[1.0, 2.0]]),  # intercept=1, coef=2 for the single used column
        "used_x_cols": ["kept"],
    }
    new_X = pd.DataFrame({"kept": [3.0], "dropped": [999.0]})

    result = predict_gwr_surface(fit, [(0, 0)], new_X)

    assert result[0] == 1.0 + 2.0 * 3.0  # 999.0 in "dropped" must be ignored
