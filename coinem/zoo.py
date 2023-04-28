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


def adam_svgd(
    model: AbstractModel,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    theta_step_size=1e-5,
    latent_step_size=1e-5,
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


def soul(
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

        # E-step: Update particles via ULA chain

        def body_fun(carry, particle_key):
            particle, key = carry

            key, subkey = jr.split(key)

            new = (
                particle
                + h * model.score_latent(particle, theta)
                + jnp.sqrt(2.0 * h) * jr.normal(subkey, shape=particle.shape)
            )

            return (new, key), new

        _, latent_new = lax.scan(
            body_fun, (latent[-1], subkey), jnp.arange(latent.shape[0])
        )

        # M-step: Update parameters
        theta_new = theta + h * model.average_score_theta(latent, theta)

        # Update the carry
        carry = (latent_new, theta_new, key)

        # Return the carry, and the new params
        return carry, {"latent": latent, "theta": theta}

    _, hist = lax.scan(step, (latent_init, theta_init, key), jnp.arange(num_steps))

    return hist["latent"], hist["theta"]
