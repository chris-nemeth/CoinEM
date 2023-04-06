from coinem.utils import ComputeDistances

import pytest
import jax.numpy as jnp
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