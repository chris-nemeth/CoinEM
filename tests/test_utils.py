from coinem.utils import ComputeDistances, cum_mean, procrustes

import pytest
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array

array_a = jnp.array([[1.0], [2.0], [3.0]])
true_dists_a = jnp.array(
    [[[0.0], [-1.0], [-2.0]], [[1.0], [0.0], [-1.0]], [[2.0], [1.0], [0.0]]]
)
true_square_dists_a = jnp.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])

array_b = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
true_dists_b = jnp.array(
    [
        [[0.0, 0.0], [-2.0, -2.0], [-4.0, -4.0]],
        [[2.0, 2.0], [0.0, 0.0], [-2.0, -2.0]],
        [[4.0, 4.0], [2.0, 2.0], [0.0, 0.0]],
    ]
)
true_square_dists_b = jnp.array([[0.0, 8.0, 32.0], [8.0, 0.0, 8.0], [32.0, 8.0, 0.0]])


@pytest.mark.parametrize(
    "array, true_dists, true_square_dists",
    [
        (array_a, true_dists_a, true_square_dists_a),
        (array_b, true_dists_b, true_square_dists_b),
    ],
)
def test_compute_distances(
    array: Float[Array, "N D"],
    true_dists: Float[Array, "N N D"],
    true_square_dists: Float[Array, "N N"],
) -> None:
    N = array.shape[0]
    D = array.shape[-1]

    distances = ComputeDistances(array)

    # Check shapes:
    assert distances.dists.shape == (N, N, D)
    assert distances.square_dists.shape == (N, N)

    # Check values:
    assert jnp.allclose(distances.dists, true_dists)
    assert jnp.allclose(distances.square_dists, true_square_dists)


def test_cum_mean():
    # Define test input array
    rng = jr.PRNGKey(0)
    x = jr.normal(rng, (5, 3))

    # Compute expected result using NumPy
    expected_result = jnp.cumsum(x, axis=0) / jnp.arange(1, x.shape[0] + 1)[:, None]

    # Compare result with expected result using pytest
    assert jnp.allclose(cum_mean(x), expected_result)


def test_procrustes():
    A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    B = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    import scipy.spatial as ss

    expected_result = ss.procrustes(A, B)[2]

    assert jnp.allclose(procrustes(A, B)[2], expected_result)
