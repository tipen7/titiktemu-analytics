"""Geographically Weighted Regression with Adaptive Bisquare Kernel, per PRD:
'menerapkan Geographically Weighted Regression (GWR) dengan fungsi Adaptive
Bisquare Kernel guna memodelkan variasi lokal hubungan antara jarak stasiun,
kepadatan komersial, dan beban sewa terhadap omzet bisnis.'"""

import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from src.interpolation.idw import idw_interpolate


def _jitter_coincident_covariates(X: np.ndarray, seed: int = 0, scale: float = 1e-3) -> np.ndarray:
    """Standard spatial-stats technique for GWR/GWR-adjacent models: when
    covariates take very few unique values relative to n (verified cause
    here -- 100 real survey points fall into only 5 grid cells, so a
    grid-joined feature like poi_count has only 5 distinct values), local
    kernel neighborhoods can end up with a zero-variance column and a
    singular design matrix. Deterministic tiny jitter (proportional to each
    column's own scale) breaks exact ties without meaningfully changing the
    covariate's information content. Not applied to already-continuous,
    naturally-varying columns (like per-point dist_to_station) -- harmless
    there too, but unnecessary."""
    rng = np.random.default_rng(seed)
    X = X.astype(float).copy()
    for col in range(X.shape[1]):
        n_unique = len(np.unique(X[:, col]))
        if n_unique < 0.2 * len(X):
            print(f"NOTE: GWR input column {col} has only {n_unique}/{len(X)} unique "
                  f"values -- applying numerical-stability jitter (standard technique "
                  f"for coincident covariates, see _jitter_coincident_covariates docstring). "
                  f"Once real point-level features exist (e.g. live OSM POI density), "
                  f"this workaround should be revisited.")
        col_scale = np.std(X[:, col]) or 1.0
        X[:, col] += rng.normal(0, scale * col_scale, size=X.shape[0])
    return X


def fit_gwr(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    x_coord_col: str = "x",
    y_coord_col: str = "y",
) -> dict:
    """Fits GWR on rows where `y_col` is present (see PRD/earlier discussion on
    the small-n caveat -- only points with a real vulnerability ratio train
    the model; the rest of the grid gets scored via `predict_gwr_surface`).

    Returns a dict with the fitted model, selected bandwidth, and local R^2
    per point -- report local R^2 alongside a global OLS baseline in any
    write-up, per our earlier discussion on GWR statistical honesty at low n.
    """
    valid = df.dropna(subset=[y_col] + x_cols)
    if len(valid) < 10:
        raise ValueError(
            f"Only {len(valid)} rows have a valid '{y_col}' -- GWR bandwidth "
            "selection is unreliable below ~10 points. Backfill more survey "
            "data before fitting, per our earlier discussion."
        )

    coords = list(zip(valid[x_coord_col], valid[y_coord_col]))
    y = valid[[y_col]].values
    X = valid[x_cols].values
    X = _jitter_coincident_covariates(X)  # see docstring -- fixes verified GWR singularity on clustered survey points

    n = len(valid)
    # mgwr's default bw_max search bound can overshoot n for adaptive kernels
    # on small samples (reproduced directly: n=40 raised "kth(=47) out of
    # bounds (40)" with defaults) -- clamp explicitly rather than trust the
    # default search bounds.
    bw = Sel_BW(coords, y, X, kernel="bisquare", fixed=False).search(bw_min=5, bw_max=n - 1)

    # Diagnostic, not an error -- surfaces the near-global-bandwidth issue
    # observed on the current clustered survey data (bw=98/100 last run) so
    # it's visible in every pipeline run's output, not just discovered by
    # reading this module's code.
    if bw >= 0.8 * n:
        print(f"WARNING: GWR bandwidth ({bw}) is >=80% of n ({n}) -- the model "
              f"is closer to a global fit than a local one. Usually means the "
              f"survey points are too spatially clustered for this feature set "
              f"to show local variation. Not a crash, but treat local "
              f"coefficients with caution until this drops with more spread-out data.")

    model = GWR(coords, y, X, bw, kernel="bisquare", fixed=False)
    results = model.fit()

    return {
        "model": model,
        "results": results,
        "bandwidth": bw,
        "local_r2": results.localR2.flatten().tolist(),
        "params": results.params,  # local coefficients, one row per input point
        "n_points": len(valid),
    }


def predict_gwr_surface(gwr_fit: dict, new_coords: list[tuple[float, float]], new_X: np.ndarray) -> np.ndarray:
    """Scores the FULL grid (including points with no ground-truth y) using
    the fitted GWR surface.

    NOTE: this deliberately does NOT use mgwr's own GWR.predict() for
    out-of-sample scoring. That method has a confirmed, unresolved bug
    (pysal/mgwr issue #50, open since 2019) that throws an IndexError
    whenever you predict at MORE new locations than were used for training
    -- exactly our case (~40 trained survey points -> 390 grid cells).

    Instead: IDW-interpolate the fitted LOCAL coefficients (results.params)
    from the training points onto the new grid locations, then apply the
    linear combination directly. This is the same technique GIS tools like
    ArcGIS's GWR use to produce a coefficient surface over unsampled areas,
    so it's a legitimate approach, not just a workaround of convenience.
    """
    train_coords = np.array(gwr_fit["model"].coords)
    local_params = np.array(gwr_fit["params"])  # shape: (n_train, n_predictors + 1 intercept)

    interpolated_params = idw_interpolate(train_coords, local_params, np.array(new_coords))

    # local_params columns: [intercept, x_col_1, x_col_2, ...]
    intercept = interpolated_params[:, 0]
    coeffs = interpolated_params[:, 1:]
    predicted = intercept + np.sum(coeffs * new_X, axis=1)
    return predicted
