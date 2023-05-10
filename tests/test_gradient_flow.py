import pytest
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap

from coinem.gradient_flow import stein_grad

# Define Gaussian mixture:
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance,
)

dist_a = MultivariateNormalFullCovariance(
    loc=[0.0, 0.0], covariance_matrix=[[1.0, 0.9], [0.9, 1.0]]
)
dist_b = MultivariateNormalFullCovariance(
    loc=[0.0, 0.0], covariance_matrix=[[1.0, -0.9], [-0.9, 1.0]]
)
log_prob = lambda x: jnp.logaddexp(dist_a.log_prob(x), dist_b.log_prob(x))


from coinem.kernels import MedianRBF, MeanRBF, AutoRBF, AbstractKernel


@pytest.mark.parametrize("num_particles", [1, 2, 10])
@pytest.mark.parametrize("kernel", [MedianRBF(), MeanRBF(), AutoRBF()])
def test_stein_grad(num_particles: int, kernel: AbstractKernel) -> None:
    # Create particles:
    key = jr.PRNGKey(123)
    num_particles = 100
    particles = jr.normal(key, (num_particles, 2))

    # Compute Stein gradient:
    sg = stein_grad(
        particles=particles, score=vmap(lambda p: grad(log_prob)(p)), kernel=kernel
    )

    # Check that the gradient is finite:
    assert jnp.all(jnp.isfinite(sg))

    # Check that the gradient is not zero:
    assert jnp.any(sg != 0.0)

    # Check that the gradient is not NaN:
    assert jnp.all(~jnp.isnan(sg))

    # Check shape:
    assert sg.shape == (num_particles, 2)
