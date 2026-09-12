"""Inverse Distance Weighting interpolation, per PRD:
'interpolasi spasial Inverse Distance Weighting (IDW) untuk membentuk
lapisan Heatmap Ketahanan Bisnis.'

Also used internally by src/modeling/gwr.py to interpolate the GWR local
coefficient surface onto unsampled grid cells -- see that module for why
mgwr's own .predict() isn't used for that step."""

import numpy as np


def idw_interpolate(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    query_coords: np.ndarray,
    power: float = 2.0,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """known_coords: (n, 2), known_values: (n,) or (n, k) for multi-column
    interpolation, query_coords: (m, 2). Returns (m,) or (m, k)."""
    known_coords = np.asarray(known_coords, dtype="float64")
    query_coords = np.asarray(query_coords, dtype="float64")
    known_values = np.asarray(known_values, dtype="float64")

    # (m, n) distance matrix
    diff = query_coords[:, None, :] - known_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    # Exact match -> weight dominates via epsilon guard instead of div-by-zero
    weights = 1.0 / (dist ** power + epsilon)
    weights /= weights.sum(axis=1, keepdims=True)

    if known_values.ndim == 1:
        return weights @ known_values
    return weights @ known_values
