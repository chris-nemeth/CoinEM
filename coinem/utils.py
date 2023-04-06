from beartype import beartype
from dataclasses import dataclass
from jaxtyping import Array, Float, jaxtyped
from simple_pytree import Pytree, static_field
from functools import partial
import jax.numpy as jnp

"""A field that is not initialized and not printed in repr and not a Pytree leaf."""
static_hidden_field = partial(static_field, init=False, repr=False)


@jaxtyped
@beartype
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
