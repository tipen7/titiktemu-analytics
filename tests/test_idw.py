import numpy as np
from src.interpolation.idw import idw_interpolate


def test_idw_exact_at_known_point():
    known_coords = np.array([[0, 0], [10, 0], [0, 10]])
    known_values = np.array([1.0, 2.0, 3.0])
    # Querying very close to a known point should return a value very close to it.
    result = idw_interpolate(known_coords, known_values, np.array([[0.0001, 0.0001]]))
    assert abs(result[0] - 1.0) < 0.01


def test_idw_weighted_average_between_two_equidistant_points():
    known_coords = np.array([[-1, 0], [1, 0]])
    known_values = np.array([0.0, 10.0])
    result = idw_interpolate(known_coords, known_values, np.array([[0, 0]]))
    assert abs(result[0] - 5.0) < 0.01  # equidistant -> simple average
