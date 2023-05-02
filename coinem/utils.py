from dataclasses import dataclass
from jaxtyping import Array, Float
from simple_pytree import Pytree, static_field
from functools import partial
import jax.numpy as jnp

__all__ = ["ComputeDistances", "cum_mean"]

"""A field that is not initialized and not printed in repr and not a Pytree leaf."""
static_hidden_field = partial(static_field, init=False, repr=False)


@dataclass
class ComputeDistances(Pytree):
    """Compute distances between points in a batch.

    Args:
        x (Float[Array, "N D"]): Input data.
    """

    x: Float[Array, "N D"]

    # dists attribute for caching distances!
    dists: Float[Array, "N N D"] = static_hidden_field()

    # square_dists attribute for caching sqaured-distances!
    square_dists: Float[Array, "N N"] = static_hidden_field()

    def __post_init__(self):
        # Matrix of entries [(x - y)].
        self.dists = self.x[:, None, :] - self.x[None, :, :]  # [N N D]

        # Matrix of entries [(x - y)^2].
        self.square_dists = jnp.sum(self.dists**2, axis=-1)  # [N N]


def cum_mean(x: Float[Array, "*"], axis: int = 0) -> Float[Array, "*"]:
    """
    Computes the cumulative mean of a JAX array along the specified axis.

    Args:
      x (Float[Array, "*"]): A JAX array.
      axis (int): An integer specifying the axis along which to compute the
        cumulative mean. Default is 0.

    Returns:
      A JAX array containing the cumulative mean along the specified axis.
    """
    n = jnp.arange(1, x.shape[axis] + 1)
    cumsum = jnp.cumsum(x, axis=axis)
    n_shape = [1] * x.ndim
    n_shape[axis] = -1
    n = n.reshape(n_shape)
    return cumsum / n
