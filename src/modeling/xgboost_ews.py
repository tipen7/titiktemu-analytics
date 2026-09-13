"""XGBoost EWS classification + Predicted Matching Score, per PRD:
'model machine learning XGBoost yang menjalankan dua fungsi: mengklasifikasikan
tingkat kerentanan grid ke dalam status EWS (Aman, Waspada, Bahaya) serta
mengkalkulasi Predicted Matching Score (0-100).'

EWS codes: 0 = Aman, 1 = Waspada, 2 = Bahaya -- derived from vulnerability_index
terciles per our earlier PRD discussion (GWR surface -> bucketed classes,
since supervised training data is too sparse to learn EWS classes from
scratch independently -- see PRD Section 4, 'AI Integration', framing note)."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


EWS_LABELS = {0: "aman", 1: "waspada", 2: "bahaya"}


def real_survey_ews_cutoffs(true_vulnerability: np.ndarray) -> tuple[float, float]:
    """Tercile cutoffs computed from REAL, measured UMKM vulnerability
    (rent/revenue ratio) -- the ground-truth calibration source for what
    "aman/waspada/bahaya" actually MEANS. Anchoring cutoffs here, not to
    the interpolated grid's own distribution, matters because most of the
    390-cell grid sits far from any of the ~35 real training points: GWR's
    IDW-interpolated coefficients extrapolate to ~0 there (see
    predict_gwr_surface's non-negativity clip), so a majority of grid
    cells now share one tied value. Computing terciles from THAT
    distribution instead (the old behavior) is doubly wrong: it crashes
    (`pd.qcut` needs unique bin edges) or, if forced through, produces
    cutoffs that don't mean anything real -- see CONTEXT.md Sec. 2 on why
    the grid's own distribution isn't a reliable scale reference."""
    return tuple(np.quantile(true_vulnerability, [1 / 3, 2 / 3]))


def vulnerability_to_ews(vulnerability_index: pd.Series, cutoffs: tuple[float, float] | None = None) -> pd.Series:
    """Buckets the continuous GWR-derived vulnerability index into EWS
    classes (0=aman, 1=waspada, 2=bahaya).

    `cutoffs`, when given (see `real_survey_ews_cutoffs`), are FIXED
    thresholds anchored to real survey data -- used for the production grid
    so "waspada"/"bahaya" mean the same real-world rent-burden level
    regardless of how much of the grid GWR can currently reach with real
    signal.

    Without `cutoffs` (standalone/test use), falls back to self-adaptive
    terciles of whatever series is passed in, with `duplicates='drop'` so a
    series with a large tied mass (e.g. many grid cells clipped to the same
    floor -- see predict_gwr_surface) degrades gracefully to fewer than 3
    classes instead of raising."""
    if cutoffs is not None:
        return pd.Series(
            np.digitize(vulnerability_index, cutoffs), index=vulnerability_index.index,
        ).astype(int)
    # labels=False (not [0,1,2]) -- duplicates='drop' can yield fewer than 3
    # bins on a heavily-tied series, and fixed 3-item labels would then
    # raise ("Bin labels must be one fewer than the number of bin edges").
    return pd.qcut(vulnerability_index, q=3, labels=False, duplicates="drop").astype(int)


def train_xgboost_ews(grid: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Trains XGBoost to approximate the GWR-derived EWS surface using only
    structural features (poi_count, dist_to_station, ndbi_mean, etc) --
    this lets the trained model score NEW grid cells cheaply at serving
    time without re-running the full GWR fit each time. Per our earlier
    discussion: this is XGBoost approximating/operationalizing the GWR
    surface for speed, not an independently-validated classifier -- keep
    that framing in any report/dashboard copy."""
    X = grid[feature_cols].values
    y_ews = grid["ews_code"].values

    # Which of the 3 EWS codes actually show up varies by run now that
    # cutoffs are anchored to real survey data (see vulnerability_to_ews)
    # instead of an artificial self-adaptive tercile that always produced
    # all 3 classes by construction -- a run can genuinely produce e.g.
    # only {aman, bahaya} with nothing in the "waspada" band (verified:
    # {0, 2} with no 1s). XGBoost's sklearn wrapper requires labels to
    # literally BE contiguous 0..k-1 already -- it does NOT relabel {0,2}
    # to {0,1} itself (verified: raises "Invalid classes inferred... got
    # [0 2]" even with objective/num_class left unset). So labels ARE
    # remapped here, but objective/num_class are deliberately left for
    # XGBClassifier to infer from the now-contiguous y_train (hardcoding
    # objective="multi:softprob" with an explicit num_class=2 was tried and
    # broke sklearn's own scoring path with "mix of binary and
    # multilabel-indicator targets" -- letting it auto-pick
    # binary:logistic vs multi:softprob from contiguous labels avoids that).
    present_classes = sorted(np.unique(y_ews).tolist())
    class_to_idx = {c: i for i, c in enumerate(present_classes)}
    y_encoded = np.array([class_to_idx[c] for c in y_ews])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    return {
        "model": clf, "train_accuracy": train_acc, "test_accuracy": test_acc,
        "feature_cols": feature_cols, "present_classes": present_classes,
    }


def wilson_score_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% (z=1.96) Wilson score interval for a proportion -- unlike the naive
    p +/- 1.96*sqrt(p(1-p)/n) interval, this doesn't break down (go negative,
    or exceed 1) at the small n this pipeline's real survey data has. No
    extra dependency needed (statsmodels isn't in requirements.txt) -- this
    is the textbook closed-form formula."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def validate_ews_against_survey(
    true_vulnerability: np.ndarray,
    loocv_predicted_vulnerability: np.ndarray,
) -> dict:
    """The genuine, real-ground-truth accuracy check `ews_model_accuracy_pct`
    was missing: buckets both the REAL survey vulnerability and the
    leave-one-out cross-validated GWR prediction (see gwr.loocv_predict) for
    the same real UMKM points into EWS terciles, using cutoffs anchored to
    the real survey's OWN distribution (`real_survey_ews_cutoffs` --
    the SAME cutoffs `run_pipeline.py` uses for the production grid, so
    this measures "does the app's real displayed zone color match this
    real UMKM's actual measured risk," not an abstract stat).

    NOTE: earlier versions of this function used the production GRID's own
    vulnerability_index terciles instead. That was wrong and produced a
    meaningless 100% "accuracy": most of the 390-cell grid sits far enough
    from any real training point that GWR's extrapolated prediction clips
    to a shared floor (see predict_gwr_surface's non-negativity clip), so
    real survey points -- which sit exactly where the model DOES have real
    signal -- were trivially on the "high" side of the grid's skewed
    distribution regardless of whether the model discriminated among them
    correctly. Anchoring cutoffs to the real survey data itself avoids that.

    Small n is the honest state of the data (see CONTEXT.md Sec. 2/3) --
    reported with a 95% Wilson CI and a sample-size-based confidence_level
    instead of a bare percentage, so a thin sample can't imply false
    precision the way a single unqualified number would.
    """
    valid = ~(np.isnan(true_vulnerability) | np.isnan(loocv_predicted_vulnerability))
    true_v = true_vulnerability[valid]
    pred_v = loocv_predicted_vulnerability[valid]
    n = len(true_v)

    cutoffs = real_survey_ews_cutoffs(true_v)
    true_bucket = np.digitize(true_v, cutoffs)
    pred_bucket = np.digitize(pred_v, cutoffs)

    matches = int((true_bucket == pred_bucket).sum())
    accuracy = matches / n if n else 0.0
    ci_low, ci_high = wilson_score_interval(matches, n)

    if n >= 100:
        confidence_level = "high"
    elif n >= 30:
        confidence_level = "moderate"
    else:
        confidence_level = "low"

    return {
        "n": n,
        "matches": matches,
        "accuracy_pct": round(accuracy * 100, 1),
        "ci_95_low_pct": round(ci_low * 100, 1),
        "ci_95_high_pct": round(ci_high * 100, 1),
        "confidence_level": confidence_level,
    }


def score_matching(grid: pd.DataFrame, feature_cols: list[str], ews_fit: dict) -> np.ndarray:
    """Predicted Matching Score (0-100) -- how suitable a grid cell is for
    new tenant placement. Inversely related to vulnerability/EWS risk,
    scaled by commercial intensity (poi_count) as a demand signal."""
    clf = ews_fit["model"]
    X = grid[feature_cols].values
    ews_proba = clf.predict_proba(X)  # (n, k) -- column order follows ews_fit["present_classes"], NOT always [aman, waspada, bahaya]

    # Column 0 is only "P(aman)" when class 0 (aman) is actually the first
    # present class -- not guaranteed now that cutoffs are real-survey-
    # anchored (see train_xgboost_ews) and a run can genuinely have no
    # "waspada" cells, shifting column order. 0.0 if aman didn't occur at
    # all in this run's grid (no cell was ever safe -- an honest score of
    # 0 suitability, not a silent misread of some other class's column).
    present_classes = ews_fit["present_classes"]
    aman_col = present_classes.index(0) if 0 in present_classes else None
    safety_score = (ews_proba[:, aman_col] * 100) if aman_col is not None else np.zeros(len(grid))
    poi_norm = (grid["poi_count"] - grid["poi_count"].min()) / (
        grid["poi_count"].max() - grid["poi_count"].min() + 1e-9
    )
    matching_score = 0.7 * safety_score + 0.3 * (poi_norm * 100)
    return np.clip(matching_score, 0, 100)
