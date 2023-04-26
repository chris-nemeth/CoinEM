import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
from jax import lax

from jaxtyping import Array, Float

from coinem.kernels import MedianRBF
from coinem.gradient_transforms import adacoin
from coinem.gradient_flow import svgd
from coinem.model import AbstractModel

# TODO: Abstract out E-step and M-step, make it modular.


def coinEM(
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


def pgd(
    model: AbstractModel,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    num_steps: int,
    key=jr.PRNGKey(42),
    step_size=1e-2,
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
