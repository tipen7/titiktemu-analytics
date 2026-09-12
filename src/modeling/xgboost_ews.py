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


def vulnerability_to_ews(vulnerability_index: pd.Series) -> pd.Series:
    """Buckets the continuous GWR-derived vulnerability index into EWS
    terciles. Deliberately NOT a hardcoded threshold -- terciles adapt to
    whatever range this particular batch run's grid actually spans."""
    return pd.qcut(vulnerability_index, q=3, labels=[0, 1, 2]).astype(int)


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ews, test_size=0.2, random_state=42, stratify=y_ews
    )

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    return {"model": clf, "train_accuracy": train_acc, "test_accuracy": test_acc, "feature_cols": feature_cols}


def score_matching(grid: pd.DataFrame, feature_cols: list[str], ews_fit: dict) -> np.ndarray:
    """Predicted Matching Score (0-100) -- how suitable a grid cell is for
    new tenant placement. Inversely related to vulnerability/EWS risk,
    scaled by commercial intensity (poi_count) as a demand signal."""
    clf = ews_fit["model"]
    X = grid[feature_cols].values
    ews_proba = clf.predict_proba(X)  # (n, 3) probability of [aman, waspada, bahaya]

    safety_score = ews_proba[:, 0] * 100  # P(aman) as base score
    poi_norm = (grid["poi_count"] - grid["poi_count"].min()) / (
        grid["poi_count"].max() - grid["poi_count"].min() + 1e-9
    )
    matching_score = 0.7 * safety_score + 0.3 * (poi_norm * 100)
    return np.clip(matching_score, 0, 100)
