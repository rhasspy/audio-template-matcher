"""Dynamic time warping implementation."""
import math
from typing import List, Tuple

import numpy as np
import scipy


def compute_optimal_path(
    x: np.ndarray,
    y: np.ndarray,
    distance_func: str = "cosine",
) -> Tuple[float, np.ndarray]:
    """Computes optimal path between x and y.

    Returns distance and cost matrix.
    """
    m = len(x)
    n = len(y)

    # Need 2-D arrays for distance calculation
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    distance_matrix = scipy.spatial.distance.cdist(x, y, metric=distance_func)

    cost_matrix = np.full(shape=(m, n), fill_value=math.inf, dtype=float)
    cost_matrix[0][0] = distance_matrix[0][0]

    for row in range(1, m):
        cost = distance_matrix[row, 0]
        cost_matrix[row][0] = cost + cost_matrix[row - 1][0]

    for col in range(1, n):
        cost = distance_matrix[0, col]
        cost_matrix[0][col] = cost + cost_matrix[0][col - 1]

    for row in range(1, m):
        for col in range(1, n):
            cost = distance_matrix[row, col]
            cost_matrix[row][col] = cost + min(
                cost_matrix[row - 1][col],  # insertion
                cost_matrix[row][col - 1],  # deletion
                cost_matrix[row - 1][col - 1],  # match
            )

    distance = cost_matrix[m - 1][n - 1]

    return distance, cost_matrix


def compute_optimal_path_with_window(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 5,
    step_pattern: float = 2,
    distance_func: str = "cosine",
) -> Tuple[float, np.ndarray]:
    """Computes optimal path between x and y using a window.

    Returns distance and cost matrix.
    """
    n = len(x)
    m = len(y)

    # Avoid case where endpoint lies outside band
    window = max(window, abs(m - n))

    # Need 2-D arrays for distance calculation
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # Pre-compute distance between all pairs
    distance_matrix = scipy.spatial.distance.cdist(x, y, metric=distance_func)

    cost_matrix = np.full(shape=(n + 1, m + 1), fill_value=math.inf, dtype=float)

    cost_matrix[0][0] = 0
    for row in range(1, n + 1):
        col_start = max(1, row - window)
        col_end = min(m, row + window)

        for col in range(col_start, col_end + 1):
            cost = distance_matrix[row - 1, col - 1]

            # symmetric step pattern
            cost_matrix[row][col] = min(
                (step_pattern * cost) + cost_matrix[row - 1][col - 1],
                cost + cost_matrix[row - 1][col],
                cost + cost_matrix[row][col - 1],
            )

    distance = cost_matrix[n][m]

    return distance, cost_matrix[1:, 1:]


def get_path(cost_matrix: np.ndarray, eps: float = 1e-14) -> List[Tuple[int, int]]:
    """Get actual path from cost matrix."""
    m, n = cost_matrix.shape
    row = m - 1
    col = n - 1
    path = [(row, col)]

    while (row > 0) or (col > 0):
        if (row > 0) and (col > 0):
            min_cost = min(
                cost_matrix[row - 1][col],  # insertion
                cost_matrix[row][col - 1],  # deletion
                cost_matrix[row - 1][col - 1],  # match
            )

            if math.isclose(min_cost, cost_matrix[row - 1][col - 1], rel_tol=eps):
                row = row - 1
                col = col - 1
            elif math.isclose(min_cost, cost_matrix[row - 1][col], rel_tol=eps):
                row = row - 1
            elif math.isclose(min_cost, cost_matrix[row][col - 1], rel_tol=eps):
                col = col - 1
        elif (row > 0) and (col == 0):
            row = row - 1
        elif (row == 0) and (col > 0):
            col = col - 1

        path.append((row, col))

    return list(reversed(path))
