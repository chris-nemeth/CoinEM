from jaxtyping import Array, Float
from beartype.typing import Callable
import jax.numpy as jnp

from coinem.kernels import AbstractKernel


def svgd(
    particles: Float[Array, "N D"],
    score: Callable[[Float[Array, "N D"]], Float[Array, "N 1"]],
    kernel: AbstractKernel,
) -> Float[Array, "N D"]:
    """
    Args:
        particles (Float[Array, "N D"]): The current particles.
        score (Callable[[Float[Array, "N D"]], Float[Array, "N 1"]]): The score function.
        kernel (AbstractKernel): The kernel.

    Returns:
        Float[Array, "N D"]: The updated particles.
    """
    N = particles.shape[0]  # N
    K, dK = kernel.K_dK(particles)  # Kxx, ∇x Kxx
    s = score(particles)  # ∇x p(x)

    # Φ(x) = (Kxx ∇x p(x) + ∇x Kxx) / N
    return (jnp.matmul(K, s) + dK) / N
