import numpy as np

UNIFORM = np.full(3, 1/3)
SIMPLEX = np.array([UNIFORM, *([a/20, b/20, (20-a-b)/20]
                              for a in range(2,17) for b in range(2,19-a))])


def choose_weights(query_deltas):
    """Rows are support update directions; columns are held-out query conditions."""
    change = np.asarray(query_deltas, dtype=float)
    if change.shape != (3,3) or not np.isfinite(change).all():
        raise ValueError("Expected nine finite query loss changes")
    scale = max(abs(change).max(), 1e-8)
    # Interpolate the three measured displacements, penalizing the largest loss increase.
    predicted = SIMPLEX @ (change / scale)
    cost = predicted.mean(axis=1) + np.maximum(predicted.max(axis=1), 0)
    cost += 0.05 * ((SIMPLEX - UNIFORM)**2).sum(axis=1)
    return SIMPLEX[cost.argmin()].copy()
