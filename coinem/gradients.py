from beartype import beartype
from dataclasses import dataclass
from jaxtyping import Array, Float
from simple_pytree import Pytree
from beartype.typing import Callable
import jax.numpy as jnp
from jax import grad, vmap

from abc import abstractmethod
from coinem.kernels import AbstractKernel


class AbstractGradient(Pytree):
    """Base class for gradients."""

    @abstractmethod
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """Compute gradient of the objective function at x.

        Args:
            x (Float[Array, "N D"]): Input data.

        Returns:
            Float[Array, "N D"]: Gradient of the objective function at x.
        """
        raise NotImplementedError


@dataclass
class SVGD(AbstractGradient):
    log_prob: Callable[[Float[Array, "D"]], Float[Array, "1"]]
    kernel: AbstractKernel

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        """
        Args:
            x (Float[Array, "N D"]): The current particles.

        Returns:
            Float[Array, "N D"]: The updated particles.
        """
        N = x.shape[0]  # N
        K, dK = self.kernel.K_dK(x)  # Kxx, ∇x Kxx
        s = score(self.log_prob)(x)  # ∇x p(x)

        # Φ(x) = (Kxx ∇x p(x) + ∇x Kxx) / N
        return (jnp.matmul(K, s) + dK) / N


def score(
    log_prob: Callable[[Float[Array, "D"]], Float[Array, "1"]]
) -> Callable[[Float[Array, "N D"]], Float[Array, "N D"]]:
    """Construct a score function from a log-probability function."""

    def score_fn(x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        return vmap(lambda xi: grad(log_prob)(xi))(x)  # ∇x p(x)

    return score_fn
