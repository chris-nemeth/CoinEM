from __future__ import annotations

from beartype.typing import Callable

import jax.tree_util as jtu
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree


from typing import NamedTuple, Callable, Tuple, Any

# 🛑 We assume a PyTree of JAX arrays as input.

OptimiserState = Any


class GradientTransformation(NamedTuple):
    """A stateful gradient transformation. Optax inspired.

    Attributes:
      init: A function that initializes the state of the transformation.
      update: A function that applies the transformation to the gradient updates.
    """

    init: Callable[[PyTree], OptimiserState]
    update: Callable[[PyTree, OptimiserState, PyTree], PyTree]


class AdaCoinState(NamedTuple):
    """State for the Adam algorithm."""

    init_params: PyTree
    grad_sum: PyTree
    abs_grad_sum: PyTree
    reward: PyTree
    Lipschitz: PyTree


def adacoin() -> GradientTransformation:
    """Stateless identity transformation that leaves input gradients untouched.

    This function passes through the *gradient updates* unchanged.

    Note, this should not to be confused with `set_to_zero`, which maps the input
    updates to zero - which is the transform required for the *model parameters*
    to be left unchanged when the updates are applied to them.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(params: PyTree) -> AdaCoinState:
        """Initializes the state for AdaCoin.

        Args:
          params(PyTree): a tree of parameters.

        Returns:
          AdaCoinState: An initial state.
        """

        zeros = jtu.tree_map(jnp.zeros_like, params)
        return AdaCoinState(params, zeros, zeros, zeros, zeros)

    def update_fn(
        gradient: PyTree, state: AdaCoinState, params: PyTree
    ) -> Tuple[PyTree, AdaCoinState]:
        """Applies AdaCoin to the gradient updates.

        Args:
          gradient (PyTree): a tree of gradient updates.
          state (AdaCoin): the current state.
          params (PyTree): a tree of parameters.

        Returns:
          Tuple[PyTree, AdaCoinState]: A tuple of the updated gradient and the updated state.
        """

        params0 = state.init_params
        grad_sum = state.grad_sum
        abs_grad_sum = state.abs_grad_sum
        rt_minus_1 = state.reward
        lt_minus_1 = state.Lipschitz

        # Gradient
        ct = gradient

        # Absolute |ct|
        abs_ct = jtu.tree_map(jnp.abs, ct)

        # sum of gradients
        grad_sum = jtu.tree_map(jnp.add, grad_sum, ct)
        abs_grad_sum = jtu.tree_map(jnp.add, abs_grad_sum, abs_ct)

        # Maximum observed scalem Lt
        lt = jtu.tree_map(
            lambda ct_i, lt_m1_i: jnp.maximum(ct_i, lt_m1_i), abs_ct, lt_minus_1
        )

        # Reward
        rt = jtu.tree_map(
            lambda rt_mt_i, params_i, ct_i, params0_i: jnp.maximum(
                rt_mt_i + jnp.multiply(params_i - params0_i, ct_i), 0
            ),
            rt_minus_1,
            params,
            ct,
            params0,
        )

        # Param updates
        updates = jtu.tree_map(
            lambda params0_i, grad_sum_i, abs_grad_sum_i, rt_i, lt_i: params0_i
            + grad_sum_i / (lt_i * (abs_grad_sum_i + lt_i)) * (lt_i + rt_i),
            params0,
            grad_sum,
            abs_grad_sum,
            rt,
            lt,
        )

        # Update state
        new_state = AdaCoinState(params0, grad_sum, abs_grad_sum, rt, lt)

        return updates, new_state

    return GradientTransformation(init_fn, update_fn)
