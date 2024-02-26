import jax.numpy as jnp
import jax.random as jr
from jax.random import KeyArray
from jaxtyping import Array, Float
from typing import Tuple, Dict, Callable

from coinem.expectation_step import AbstractExpectationStep
from coinem.maximisation_step import AbstractMaximisationStep
from coinem.dataset import Dataset, get_batch

import jax.tree_util as jtu


# Define a timer.
import jax
import time
import jax.experimental.host_callback as hcb
_timer = lambda : hcb.call(lambda _: time.perf_counter(), None, result_shape=jax.ShapeDtypeStruct((), jax.numpy.float32))



def expectation_maximisation(
    expectation_step: AbstractExpectationStep,
    maximisation_step: AbstractMaximisationStep,
    data: Dataset,
    latent_init: Float[Array, "N D"],
    theta_init: Float[Array, "Q"],
    max_time: float,
    batch_size: int = -1,
    key: KeyArray = jr.PRNGKey(42),
    metrics: Dict[
        str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]
    ] = None,
) -> Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]:
    """
    Performs the expectation-maximisation algorithm.

    The E-step and M-step are performed in an alternating fashion, but independently of each other.

    Args:
        expectation_step (AbstractExpectationStep): The E-step.
        maximisation_step (AbstractMaximisationStep): The M-step.
        data (Dataset): The dataset.
        latent_init (Float[Array, "N D"]): The initial latent particles.
        theta_init (Float[Array, "Q"]): The initial parameters.
        num_steps (int): The number of steps to perform, K.
        batch_size (int, optional): The batch size. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to jr.PRNGKey(42).
        metrics (Dict[str, Callable[[Float[Array, "N D"], Float[Array, "N D"]], Float[Array, "1"]]], optional): The metrics to compute. Defaults to {}.

    Returns:
        Tuple[Float[Array, "K N D"], Float[Array, "K Q"]]: The latent particles and parameters.
    """

    t0 = time.perf_counter()

    key, exp_key, max_key = jr.split(key, 3)

    # Initialise the E-step
    expectation_state = expectation_step.init(latent_init, exp_key)

    # Initialise the M-step
    maximisation_state = maximisation_step.init(theta_init, max_key)

    # Define optimisation step:
    def _step(carry: tuple):
        # Get params and opt_state
        latent, theta, expectation_state, maximisation_state, key = carry

        # Produce a new key
        key, subkey = jr.split(key)

        # Get a batch of data

        if batch_size != -1:
            data_batch = get_batch(data, batch_size, subkey)
        else:
            data_batch = data

        # E-step: Update particles
        latent_new, expectation_state_new = expectation_step.update(
            expectation_state=expectation_state,
            latent=latent,
            theta=theta,
            data=data_batch,
        )

        # M-step: Update parameters
        theta_new, maximisation_state_new = maximisation_step.update(
            maximisation_state=maximisation_state,
            latent=latent,
            theta=theta,
            data=data_batch,
        )

        # Update the carry
        carry = (
            latent_new,
            theta_new,
            expectation_state_new,
            maximisation_state_new,
            key,
        )

        if metrics is not None:
            return carry, jtu.tree_map(
                lambda metric: metric(latent_new, theta_new), metrics
            )

        # Return the carry, and the new params
        return carry

    
    _init_state = (latent_init, theta_init, expectation_state, maximisation_state, key)
    _cond = lambda _: _timer() - t0 < max_time


    latent, theta, *_ = jax.lax.while_loop(
        _cond, 
        _step, 
        _init_state,
    )

    return latent, theta
