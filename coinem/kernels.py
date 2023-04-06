from beartype import beartype
from dataclasses import dataclass
from jaxtyping import Array, Float, jaxtyped
from typing import Tuple
from simple_pytree import Pytree, static_field
import jax.numpy as jnp


@jaxtyped
@beartype
@dataclass
class ComputeDistances(Pytree):
    """Compute distances between points in a batch.

    Args:
        x (Float[Array, "N D"]): Input data.
    """

    x: Float[Array, "N D"]
    dists: Float[Array, "N N D"] = static_field(
        init=False, repr=False
    )  # dists attribute for caching distances!
    square_dists: Float[Array, "N N"] = static_field(
        init=False, repr=False
    )  # square_dists attribute for caching sqaured-distances!

    def __post_init__(self):
        self.dists = (
            self.x[:, None, :] - self.x[None, :, :]
        )  # [N N D], Matrix of entries [(x - y)].
        self.square_dists = jnp.sum(
            self.dists**2, axis=-1
        )  # [N N], Matrix of entries [(x - y)^2].


@jaxtyped
@beartype
@dataclass
class AutoRBF(Pytree):
    """RBF kernel with automatic bandwidth selection as the median of the pairwise distances."""

    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute RBF kernel Gram matrix and gradient.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix.
        """
        distances = ComputeDistances(x)
        h = jnp.sqrt(
            0.5
            * jnp.median(distances.square_dists)
            / jnp.log(distances.square_dists.shape[0] + 1.0)
        )

        K = jnp.exp(-0.5 * distances.square_dists / h**2)  # [N N]

        # ∇x Kxx = [∑_i exp(-0.5 * (x - y)^2 / h^2) * (x - y) / h^2]
        dK = jnp.sum(K[:, :, None] * distances.dists, axis=1) / h**2  # [N N D]

        return K, dK  # Kxx, ∇x Kxx


@jaxtyped
@beartype
@dataclass
class RBF(Pytree):
    """RBF kernel with static bandwidth selection.

    Args:
        h (Float[Array, "1"]): Bandwidth parameter if median_trick is False. Defaults to 1.0.
    """

    h: Float[Array, "1"] = jnp.array([1.0])

    def K_dK(
        self, x: Float[Array, "N D"]
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N D"]]:
        """Compute RBF kernel Gram matrix and gradient.

        Args:
            x (Float[Array, "N D"])): Input data.

        Returns:
            Tuple[Float[Array, "N N"], Float[Array, "N N D"]]: Kernel Gram matrix.
        """
        distances = ComputeDistances(x)

        K = jnp.exp(-0.5 * distances.square_dists / self.h**2)  # [N N]

        # ∇x Kxx = [∑_i exp(-0.5 * (x - y)^2 / h^2) * (x - y) / h^2]
        dK = jnp.sum(K[:, :, None] * distances.dists, axis=1) / self.h**2  # [N N D]

        return K, dK  # Kxx, ∇x Kxx


# -------- ASIDE ------------
# Alternative way to compute K and dK, would be to use vmap and grad.
# This is useful if we want to do arbitrary kernel functions.
#
#
