from coinem.kernels import RBF, MedianRBF, ComputeDistances

import pytest
import jax.numpy as jnp
from jaxtyping import Float, Array


# Arrays to test distances
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


@pytest.mark.parametrize("h", [0.1, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize(
    "array, true_dists, true_square_dists",
    [
        (array_a, true_dists_a, true_square_dists_a),
        (array_b, true_dists_b, true_square_dists_b),
    ],
)
def test_rbf(
    h: float,
    array: Float[Array, "N D"],
    true_dists: Float[Array, "N N D"],
    true_square_dists: Float[Array, "N N"],
) -> None:
    N = array.shape[0]
    D = array.shape[-1]

    rbf = RBF(h=jnp.array([h]))

    # Check init:
    assert rbf.h == h

    # Compute K and dK:
    K, dK = rbf.K_dK(array)

    # Check shapes:
    assert K.shape == (N, N)
    assert dK.shape == (N, D)

    # Check values:
    assert jnp.allclose(K, jnp.exp(-0.5 * true_square_dists / h**2))
    assert jnp.allclose(
        dK,
        jnp.sum(
            jnp.exp(-0.5 * true_square_dists / h**2)[:, :, None] * true_dists, axis=1
        )
        / h**2,
    )

    # Check that the kernel is positive definite:
    jitter = jnp.eye(array.shape[0]) * 1e-6
    assert jnp.all(jnp.linalg.eigvals(K) + jitter > 0)


median_bandwith_a = jnp.array([0.6005612])
median_bandwith_b = jnp.array([1.6986436])


@pytest.mark.parametrize(
    "array, true_dists, true_square_dists, true_h",
    [
        (array_a, true_dists_a, true_square_dists_a, median_bandwith_a),
        (array_b, true_dists_b, true_square_dists_b, median_bandwith_b),
    ],
)
def test_median_rbf(
    array: Float[Array, "N D"],
    true_dists: Float[Array, "N N D"],
    true_square_dists: Float[Array, "N N"],
    true_h: Float[Array, "1"],
) -> None:
    N = array.shape[0]
    D = array.shape[-1]

    rbf = MedianRBF()

    # Compute K and dK:
    K, dK = rbf.K_dK(array)

    # Check shapes:
    assert K.shape == (N, N)
    assert dK.shape == (N, D)

    # Check values:
    assert jnp.allclose(K, jnp.exp(-0.5 * true_square_dists / true_h**2))
    assert jnp.allclose(
        dK,
        jnp.sum(
            jnp.exp(-0.5 * true_square_dists / true_h**2)[:, :, None] * true_dists,
            axis=1,
        )
        / true_h**2,
    )

    # Check that the kernel is positive definite:
    jitter = jnp.eye(array.shape[0]) * 1e-6
    assert jnp.all(jnp.linalg.eigvals(K) + jitter > 0)
