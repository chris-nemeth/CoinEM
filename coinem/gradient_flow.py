from jaxtyping import Array, Float
from beartype.typing import Callable
import jax.numpy as jnp

from coinem.kernels import AbstractKernel


def stein_grad(
    particles: Float[Array, "N *"],
    score: Callable[[Float[Array, "N D"]], Float[Array, "N 1"]],
    kernel: AbstractKernel,
) -> Float[Array, "N D"]:
    """
    Computes the kernelised Stein gradient of the particles (Liu and Wang, 2016).

    Args:
        particles (Float[Array, "N D"]): The current particles.
        score (Callable[[Float[Array, "N D"]], Float[Array, "N 1"]]): The score function.
        kernel (AbstractKernel): The kernel.

    Returns:
        Float[Array, "N D"]: The updated particles.
    """

    # Shape information:
    shape = particles.shape
    num_particles = shape[0]  # N

    # Compute the score function:
    s = score(particles)  # ∇x p(x)

    # Flatten the particles and the score function:
    flatten_s = jnp.reshape(s, (num_particles, -1))
    flatten_particles = jnp.reshape(particles, (num_particles, -1))

    # Compute the kernel and its gradient:
    K, dK = kernel.K_dK(flatten_particles)  # Kxx, ∇x Kxx

    # Compute the Stein gradient Φ(x) = (Kxx ∇x p(x) + ∇x Kxx) / N:
    flatten_stein = (jnp.matmul(K, flatten_s) + dK) / num_particles

    # Reshape the Stein gradient:
    stein = jnp.reshape(flatten_stein, shape)

    return stein


# def stein_grad(
#     particles: Float[Array, "N D"],
#     score: Callable[[Float[Array, "N D"]], Float[Array, "N 1"]],
#     kernel: AbstractKernel,
# ) -> Float[Array, "N D"]:
#     """
#     Computes the kernelised Stein gradient of the particles (Liu and Wang, 2016).

#     Args:
#         particles (Float[Array, "N D"]): The current particles.
#         score (Callable[[Float[Array, "N D"]], Float[Array, "N 1"]]): The score function.
#         kernel (AbstractKernel): The kernel.

#     Returns:
#         Float[Array, "N D"]: The updated particles.
#     """
#     N = particles.shape[0]  # N
#     K, dK = kernel.K_dK(particles)  # Kxx, ∇x Kxx
#     s = score(particles)  # ∇x p(x)

#     # Φ(x) = (Kxx ∇x p(x) + ∇x Kxx) / N
#     return (jnp.matmul(K, s) + dK) / N
