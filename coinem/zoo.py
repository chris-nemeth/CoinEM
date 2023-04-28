import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
from jax import lax

from jaxtyping import Array, Float

import optax as ox

from coinem.kernels import MedianRBF, MeanRBF, RBF
from coinem.gradient_transforms import adacoin
from coinem.gradient_flow import svgd
from coinem.model import AbstractModel

# TODO: Abstract out E-step and M-step, make it modular.
# TODO: Abstract out the kernel, to override the default MedianRBF.


def coin_svgd(
    model: AbstractModel,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    alpha: float = 0.0,
):
    # Define coinEM optimiser:
    optimizer = adacoin(alpha)

    # Initialise optimiser states:
    theta_opt_state = optimizer.init(theta_init)
    latent_opt_state = optimizer.init(latent_init)

    # Define optimisation step:
    def step(carry: tuple, iter_num: int):
        # Get params and opt_state
        latent, theta, latent_opt_state, theta_opt_state = carry

        # E-step: Update particles
        latent_score = Partial(model.score_latent_particles, theta=theta)
        grads = svgd(
            particles=latent, score=latent_score, kernel=MedianRBF()
        )  # <- TODO: Make kernel a parameter.
        latent_new, latent_new_opt_state = optimizer.update(
            grads, latent_opt_state, latent
        )

        # M-step: Update parameters
        theta_new, theta_new_opt_state = optimizer.update(
            model.average_score_theta(latent, theta), theta_opt_state, theta
        )

        # Update the carry
        carry = (latent_new, theta_new, latent_new_opt_state, theta_new_opt_state)

        # Return the carry, and the new params
        return carry, {"latent": latent, "theta": theta}

    _, hist = lax.scan(
        step,
        (latent_init, theta_init, latent_opt_state, theta_opt_state),
        jnp.arange(num_steps),
    )

    return hist["latent"], hist["theta"]


def coin_svgd_em(
    model: AbstractModel,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
):
    # Define coinEM optimiser:
    optimizer = adacoin()

    # Initialise optimiser states:
    theta_opt_state = optimizer.init(theta_init)
    latent_opt_state = optimizer.init(latent_init)

    # Define optimisation step:
    def step(carry: tuple, iter_num: int):
        # Get params and opt_state
        latent, theta, latent_opt_state, theta_opt_state = carry

        # E-step: Update particles
        latent_score = Partial(model.score_latent_particles, theta=theta)
        grads = svgd(
            particles=latent, score=latent_score, kernel=MedianRBF()
        )  # <- TODO: Make kernel a parameter.
        latent_new, latent_new_opt_state = optimizer.update(
            grads, latent_opt_state, latent
        )

        # M-step: Update parameters
        theta_new, theta_new_opt_state = optimizer.update(
            model.average_score_theta(latent_new, theta), theta_opt_state, theta
        )

        # Update the carry
        carry = (latent_new, theta_new, latent_new_opt_state, theta_new_opt_state)

        # Return the carry, and the new params
        return carry, {"latent": latent, "theta": theta}

    _, hist = lax.scan(
        step,
        (latent_init, theta_init, latent_opt_state, theta_opt_state),
        jnp.arange(num_steps),
    )

    return hist["latent"], hist["theta"]


def coin_svgd_me(
    model: AbstractModel,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
):
    # Define coinEM optimiser:
    optimizer = adacoin()

    # Initialise optimiser states:
    theta_opt_state = optimizer.init(theta_init)
    latent_opt_state = optimizer.init(latent_init)

    # Define optimisation step:
    def step(carry: tuple, iter_num: int):
        # Get params and opt_state
        latent, theta, latent_opt_state, theta_opt_state = carry

        # M-step: Update parameters
        theta_new, theta_new_opt_state = optimizer.update(
            model.average_score_theta(latent, theta), theta_opt_state, theta
        )

        # E-step: Update particles
        latent_score = Partial(model.score_latent_particles, theta=theta_new)
        grads = svgd(
            particles=latent, score=latent_score, kernel=MedianRBF()
        )  # <- TODO: Make kernel a parameter.
        latent_new, latent_new_opt_state = optimizer.update(
            grads, latent_opt_state, latent
        )

        # Update the carry
        carry = (latent_new, theta_new, latent_new_opt_state, theta_new_opt_state)

        # Return the carry, and the new params
        return carry, {"latent": latent, "theta": theta}

    _, hist = lax.scan(
        step,
        (latent_init, theta_init, latent_opt_state, theta_opt_state),
        jnp.arange(num_steps),
    )

    return hist["latent"], hist["theta"]


def mean_coin_svgd(
    model: AbstractModel,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
):
    # Define coinEM optimiser:
    optimizer = adacoin()

    # Initialise optimiser states:
    theta_opt_state = optimizer.init(theta_init)
    latent_opt_state = optimizer.init(latent_init)

    # Define optimisation step:
    def step(carry: tuple, iter_num: int):
        # Get params and opt_state
        latent, theta, latent_opt_state, theta_opt_state = carry

        # E-step: Update particles
        latent_score = Partial(model.score_latent_particles, theta=theta)
        grads = svgd(
            particles=latent, score=latent_score, kernel=MeanRBF()
        )  # <- TODO: Make kernel a parameter.
        latent_new, latent_new_opt_state = optimizer.update(
            grads, latent_opt_state, latent
        )

        # M-step: Update parameters
        theta_new, theta_new_opt_state = optimizer.update(
            model.average_score_theta(latent, theta), theta_opt_state, theta
        )

        # Update the carry
        carry = (latent_new, theta_new, latent_new_opt_state, theta_new_opt_state)

        # Return the carry, and the new params
        return carry, {"latent": latent, "theta": theta}

    _, hist = lax.scan(
        step,
        (latent_init, theta_init, latent_opt_state, theta_opt_state),
        jnp.arange(num_steps),
    )

    return hist["latent"], hist["theta"]


from coinem.utils import ComputeDistances
from coinem.kernels import AbstractKernel
from dataclasses import dataclass
import jax


def gradient_step_coin_svgd(
    model: AbstractModel,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
):
    # Define coinEM optimiser:
    optimizer = adacoin()

    distances = ComputeDistances(latent_init)
    h = jnp.sqrt(
        jnp.array(0.5)
        * jnp.mean(distances.square_dists)
        / jnp.log(distances.square_dists.shape[0] + 1.0)
    )

    kernel = RBF(h=jnp.array([h]))

    # Initialise optimiser states:
    theta_opt_state = optimizer.init(theta_init)
    latent_opt_state = optimizer.init(latent_init)
    kernel_opt_state = optimizer.init(kernel)

    # Define optimisation step:
    def step(carry: tuple, iter_num: int):
        # Get params and opt_state
        (
            latent,
            theta,
            latent_opt_state,
            theta_opt_state,
            kernel_opt_state,
            kernel,
        ) = carry

        # E-step: Update particles
        latent_score = Partial(model.score_latent_particles, theta=theta)
        grads = svgd(
            particles=latent, score=latent_score, kernel=kernel
        )  # <- TODO: Make kernel a parameter.
        latent_new, latent_new_opt_state = optimizer.update(
            grads, latent_opt_state, latent
        )

        # M-step: Update parameters
        theta_new, theta_new_opt_state = optimizer.update(
            model.average_score_theta(latent_new, theta), theta_opt_state, theta
        )

        def kernel_stuff(
            kernel: AbstractKernel, x: Float[Array, "N D"]
        ) -> Float[Array, "N D"]:
            """
            Args:
                x (Float[Array, "N D"]): The current particles.

            Returns:
                Float[Array, "N D"]: The updated particles.
            """
            N = x.shape[0]  # N
            K, dK = kernel.K_dK(x)  # Kxx, ∇x Kxx
            s = Partial(model.score_latent_particles, theta=theta_new)(x)  # ∇x p(x)

            # Φ(x) = (Kxx ∇x p(x) + ∇x Kxx) / N
            phi = (jnp.matmul(K, s) + dK) / N

            return jnp.linalg.norm(phi)

        # K-step: Update kernel
        grad_kern = jax.grad(kernel_stuff)(kernel, latent_new)
        kernel_new, opt_kern_new = optimizer.update(grad_kern, kernel_opt_state, kernel)

        # Update the carry
        carry = (
            latent_new,
            theta_new,
            latent_new_opt_state,
            theta_new_opt_state,
            opt_kern_new,
            kernel_new,
        )

        # Return the carry, and the new params
        return carry, {"latent": latent, "theta": theta}

    _, hist = lax.scan(
        step,
        (
            latent_init,
            theta_init,
            latent_opt_state,
            theta_opt_state,
            kernel_opt_state,
            kernel,
        ),
        jnp.arange(num_steps),
    )

    return hist["latent"], hist["theta"]


def adam_svgd(
    model: AbstractModel,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    theta_step_size=1e-4,
    latent_step_size=1e-4,
):
    # Define coinEM optimiser:
    theta_opt = ox.adam(learning_rate=theta_step_size)
    latent_opt = ox.adam(learning_rate=latent_step_size)

    # Initialise optimiser states:
    theta_opt_state = theta_opt.init(theta_init)
    latent_opt_state = latent_opt.init(latent_init)

    # Define optimisation step:
    def step(carry: tuple, iter_num: int):
        # Get params and opt_state
        latent, theta, latent_opt_state, theta_opt_state = carry

        # E-step: Update particles
        latent_score = Partial(model.score_latent_particles, theta=theta)
        grads = svgd(
            particles=latent, score=latent_score, kernel=MedianRBF()
        )  # <- TODO: Make kernel a parameter.

        latent_updates, latent_new_opt_state = latent_opt.update(
            grads, latent_opt_state, latent
        )
        latent_new = ox.apply_updates(latent, latent_updates)

        # M-step: Update parameters
        theta_updates, theta_new_opt_state = latent_opt.update(
            model.average_score_theta(latent, theta), theta_opt_state, theta
        )
        theta_new = ox.apply_updates(theta, theta_updates)

        # Update the carry
        carry = (latent_new, theta_new, latent_new_opt_state, theta_new_opt_state)

        # Return the carry, and the new params
        return carry, {"latent": latent, "theta": theta}

    _, hist = lax.scan(
        step,
        (latent_init, theta_init, latent_opt_state, theta_opt_state),
        jnp.arange(num_steps),
    )

    return hist["latent"], hist["theta"]


def pgd(
    model: AbstractModel,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    step_size=1e-2,
    key=jr.PRNGKey(42),
):
    h = step_size

    def step(carry: tuple, iter_num: int):
        # Get params and key
        latent, theta, key = carry

        # Split the PRNG key
        key, subkey = jr.split(key)

        # E-step: Update particles
        latent_new = (
            latent
            + h * model.score_latent_particles(latent, theta)
            + jnp.sqrt(2.0 * h) * jr.normal(subkey, shape=latent.shape)
        )

        # M-step: Update parameters
        theta_new = theta + h * model.average_score_theta(latent, theta)

        # Update the carry
        carry = (latent_new, theta_new, key)

        # Return the carry, and the new params
        return carry, {"latent": latent, "theta": theta}

    _, hist = lax.scan(step, (latent_init, theta_init, key), jnp.arange(num_steps))

    return hist["latent"], hist["theta"]
