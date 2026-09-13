"""Geographically Weighted Regression with Adaptive Bisquare Kernel, per PRD:
'menerapkan Geographically Weighted Regression (GWR) dengan fungsi Adaptive
Bisquare Kernel guna memodelkan variasi lokal hubungan antara jarak stasiun,
kepadatan komersial, dan beban sewa terhadap omzet bisnis.'"""

import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from src.interpolation.idw import idw_interpolate


def _jitter_coincident_covariates(X: np.ndarray, seed: int = 0, scale: float = 1e-3, verbose: bool = True) -> np.ndarray:
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
        if verbose and n_unique < 0.2 * len(X):
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
    verbose: bool = True,
    bw_criterion: str = "AICc",
) -> dict:
    """Fits GWR on rows where `y_col` is present (see PRD/earlier discussion on
    the small-n caveat -- only points with a real vulnerability ratio train
    the model; the rest of the grid gets scored via `predict_gwr_surface`).

    Returns a dict with the fitted model, selected bandwidth, and local R^2
    per point -- report local R^2 alongside a global OLS baseline in any
    write-up, per our earlier discussion on GWR statistical honesty at low n.

    `verbose=False` silences the diagnostic prints below -- used by
    `loocv_validate()`, which calls this in a loop and would otherwise
    spam the same warning n times for one pipeline run.
    """
    valid = df.dropna(subset=[y_col] + x_cols)
    if len(valid) < 10:
        raise ValueError(
            f"Only {len(valid)} rows have a valid '{y_col}' -- GWR bandwidth "
            "selection is unreliable below ~10 points. Backfill more survey "
            "data before fitting, per our earlier discussion."
        )

    # A column that is LITERALLY CONSTANT across every real training row
    # (verified case: within_walk_isochrone was 0 for every one of 35 real
    # survey points -- none of the surveyed UMKM happen to sit within the
    # 800m isochrone) carries zero real signal about that column's effect
    # on vulnerability. Jittering it (see _jitter_coincident_covariates)
    # avoids a singular design matrix, but the "coefficient" GWR then fits
    # is pure noise, not signal -- and applying a noise-driven coefficient
    # to GRID cells where this column genuinely varies (many DO fall inside
    # the isochrone) blows up under `predict_gwr_surface`'s linear
    # extrapolation (verified: full-grid vulnerability_index range was
    # -94.8 to +3.2 -- nonsensical for a rent/revenue ratio that's
    # mathematically >= 0 -- versus 0.13-0.34 at the actual training
    # locations). Excluding it from GWR entirely (not just jittering) is
    # the honest fix; it stays in XGBoost's feature set (train_xgboost_ews
    # uses the same FEATURE_COLS by design) since tree-based models don't
    # extrapolate linearly and can still use its grid-level variation
    # safely for the surface-fit approximation.
    used_x_cols = [c for c in x_cols if valid[c].nunique() > 1]
    dropped_cols = [c for c in x_cols if c not in used_x_cols]
    if dropped_cols and verbose:
        print(f"WARNING: GWR excluding {dropped_cols} from the regression -- "
              f"constant across all {len(valid)} real survey rows, so no real "
              f"local coefficient can be learned for it. It stays in "
              f"XGBoost's feature set. Revisit once survey coverage varies "
              f"on this dimension.")

    coords = list(zip(valid[x_coord_col], valid[y_coord_col]))
    y = valid[[y_col]].values
    X = valid[used_x_cols].values
    X = _jitter_coincident_covariates(X, verbose=verbose)  # see docstring -- fixes verified GWR singularity on clustered survey points

    n = len(valid)
    # mgwr's default bw_max search bound can overshoot n for adaptive kernels
    # on small samples (reproduced directly: n=40 raised "kth(=47) out of
    # bounds (40)" with defaults) -- clamp explicitly rather than trust the
    # default search bounds.
    bw = Sel_BW(coords, y, X, kernel="bisquare", fixed=False).search(
        criterion=bw_criterion, bw_min=5, bw_max=n - 1,
    )

    # Diagnostic, not an error -- surfaces the near-global-bandwidth issue
    # observed on the current clustered survey data (bw=98/100 last run) so
    # it's visible in every pipeline run's output, not just discovered by
    # reading this module's code.
    if verbose and bw >= 0.8 * n:
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
        "used_x_cols": used_x_cols,  # may be a subset of x_cols -- see constant-column exclusion above
        "dropped_x_cols": dropped_cols,
    }


def predict_gwr_surface(
    gwr_fit: dict, new_coords: list[tuple[float, float]], new_X: pd.DataFrame | np.ndarray,
    clip_non_negative: bool = True,
) -> np.ndarray:
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

    `new_X` may be a DataFrame with (at least) `gwr_fit["used_x_cols"]`
    columns -- this function selects exactly those columns itself, so a
    caller passing the full FEATURE_COLS set (including any column
    `fit_gwr` excluded as constant-in-training, see its docstring) can't
    accidentally misalign columns against `gwr_fit["params"]`, which only
    has coefficients for `used_x_cols`. A plain ndarray is still accepted
    for callers that have already selected the right columns themselves.
    """
    if isinstance(new_X, pd.DataFrame):
        new_X = new_X[gwr_fit["used_x_cols"]].values

    train_coords = np.array(gwr_fit["model"].coords)
    local_params = np.array(gwr_fit["params"])  # shape: (n_train, n_predictors + 1 intercept)

    interpolated_params = idw_interpolate(train_coords, local_params, np.array(new_coords))

    # local_params columns: [intercept, x_col_1, x_col_2, ...] (used_x_cols order)
    intercept = interpolated_params[:, 0]
    coeffs = interpolated_params[:, 1:]
    predicted = intercept + np.sum(coeffs * new_X, axis=1)

    # Vulnerability is, by definition, a rent/revenue ratio -- mathematically
    # non-negative. A negative prediction here is never real signal, only
    # linear extrapolation overshoot (the IDW-interpolated local
    # coefficients are only reliable near the training locations; grid
    # cells far from any surveyed UMKM can extrapolate past zero). Clipping
    # the floor is a domain constraint, not a tuning choice -- it does NOT
    # paper over the excluded-constant-column issue above, which is the
    # actual fix; this just stops physically-impossible negative "rent
    # burden" values from reaching the EWS classifier and the dashboard.
    #
    # `clip_non_negative=False` is for callers fitting GWR on a TRANSFORMED
    # y (e.g. log(vulnerability), see the log-transform experiment in
    # scripts/experiment_v3_v5_accuracy.py) -- a negative value there is a
    # perfectly valid prediction (log of a ratio below 1), not extrapolation
    # overshoot, so clipping it to 0 would corrupt it before the caller's
    # own inverse-transform (exp()) runs.
    if not clip_non_negative:
        return predicted
    return np.clip(predicted, 0.0, None)


def loocv_predict(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    x_coord_col: str = "x",
    y_coord_col: str = "y",
    bw_criterion: str = "AICc",
) -> np.ndarray:
    """Leave-one-out cross-validated GWR predictions against REAL survey
    ground truth -- this is the genuine accuracy check that
    train_xgboost_ews()'s test_accuracy is NOT: that number measures how
    well XGBoost reproduces the GWR-fitted surface (curve-fitting fidelity
    on ~linear inputs, near-100% by construction once GWR's bandwidth is
    near-global), not whether the pipeline's predictions match reality at
    real, held-out UMKM points.

    For each row with a real y_col value, refits GWR on every OTHER such
    row (bandwidth search included) and predicts at the held-out point's
    own coordinates/features via the same IDW-coefficient technique
    `predict_gwr_surface` uses for the full grid. Tractable because n here
    is real survey rows (tens), not grid cells (hundreds) -- each fold's
    fit is on n-1 points.

    Returns an array the same length as the real-y subset of `df` (NaN for
    any fold that fails, e.g. n-1 dropping below fit_gwr's 10-point floor).
    """
    valid = df.dropna(subset=[y_col] + x_cols).reset_index(drop=True)
    n = len(valid)
    preds = np.full(n, np.nan)
    y_max = valid[y_col].max()
    unstable_folds = 0
    for i in range(n):
        train = valid.drop(index=i)
        try:
            fit = fit_gwr(
                train, y_col=y_col, x_cols=x_cols, x_coord_col=x_coord_col, y_coord_col=y_coord_col,
                verbose=False, bw_criterion=bw_criterion,
            )
        except ValueError:
            continue  # already-small n dropped below the ~10-point floor with one point held out
        held_out_coord = [(valid.loc[i, x_coord_col], valid.loc[i, y_coord_col])]
        held_out_X = valid.loc[[i]]  # DataFrame -- predict_gwr_surface selects fit["used_x_cols"] itself
        preds[i] = predict_gwr_surface(fit, held_out_coord, held_out_X)[0]
        # Doesn't change the EWS validation accuracy (bucket comparison, not
        # magnitude) but is worth surfacing: at this real sample size, a
        # handful of already-thin covariates (see the jitter NOTE above --
        # some columns have as few as 3 unique values across 35 real rows)
        # can leave one particular n-1 fold's local design matrix poorly
        # conditioned, producing a wildly-off-scale single prediction.
        if y_max > 0 and preds[i] > 10 * y_max:
            unstable_folds += 1
    if unstable_folds:
        print(f"NOTE: {unstable_folds}/{n} LOOCV fold(s) produced a numerically unstable "
              f"prediction (>10x the real data's max value) -- doesn't change the "
              f"validation accuracy (bucket comparison, not magnitude) but reflects "
              f"how thin some real covariates still are at this sample size.")
    return preds
