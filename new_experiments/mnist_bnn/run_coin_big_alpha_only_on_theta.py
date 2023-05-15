import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd

from dataclasses import dataclass

from coinem.model import AbstractModel
from coinem.dataset import Dataset

from jaxtyping import Array, Float

import distrax as dx


@dataclass
class BayesNN(AbstractModel):
    """Base class for p(θ, x)."""

    def log_prob(
        self, latent: Float[Array, "D"], theta: Float[Array, "Q"], data: Dataset
    ) -> Float[Array, ""]:
        """Compute gradient of the objective function at x.

        Args:
            latent (Float[Array, "D"]): Input weights of shape (D,).
            theta (Float[Array, "Q"]): Parameters of shape (2,).

        Returns:
            Float[Array, ""]: log-probability of the data.
        """
        X = data.X  # [Batchsize, 28, 28]
        y = data.y  # [Batchsize, 1]

        # Compute prior:
        log_prior_w = (
            dx.Normal(0.0, jnp.sqrt(jnp.exp(2.0 * theta["alpha"].squeeze())))
            .log_prob(latent["w"].ravel())
            .sum()
        )
        log_prior_v = (
            dx.Normal(0.0, jnp.sqrt(jnp.exp(2.0 * theta["beta"].squeeze())))
            .log_prob(latent["v"].ravel())
            .sum()
        )

        def _log_nn(image):
            # Log of the network's output when evaluated at image with weights w, v.
            return jax.nn.log_softmax(
                jnp.dot(latent["v"], jnp.tanh(jnp.dot(latent["w"], image.flatten())))
            )

        def _log_nn_vec(images):
            # _log_nn vectorized over particles.
            return jax.vmap(_log_nn)(images)

        def _log_likelihood(images, labels):
            # Log-likelihood for set of images and labels, vectorized over particles.
            return vmap(lambda v, l: v[l])(_log_nn_vec(images), labels).sum()

        # Compute log-probability.
        return (
            _log_likelihood(X, y) + log_prior_w + log_prior_v
        ) / data.n  # log p(y|x) + log p(x|theta)

    def optimal_theta(
        self, latent_particles: Float[Array, "N D *"]
    ) -> Float[Array, "Q *"]:
        """Optimal parameter for weight cloud w."""
        mom2 = (jax.vmap(_normsq)(latent_particles["w"])).mean()  # Second moment
        mom4 = (jax.vmap(_normsq)(latent_particles["v"])).mean()  # Fourth moment

        return {
            "alpha": jnp.log(mom2 / (latent_particles["w"][0].size)) / 2,
            "beta": jnp.log(mom4 / (latent_particles["v"][0].size)) / 2,
        }


def _normsq(x):
    # Squared Frobenius norm of x.
    v = x.reshape((x.size))
    return jnp.dot(v, v)


if __name__ == "__main__":
    from jax import vmap

    # @title Load, subsample, and normalize MNIST dataset.
    import numpy as np
    from coinem.dataset import Dataset

    # Load dataset:
    from keras.datasets import mnist

    (images, labels), _ = mnist.load_data()
    images = np.array(images).astype(float)
    labels = np.array(labels).astype(int)

    # Keep only datapoints with labels 4 and 9:
    indices = (labels == 4) | (labels == 9)
    labels = labels[indices]
    images = images[indices, :, :]

    # Relabel as 4 as 0 and 9 as 1:
    for n in range(labels.size):
        if labels[n] == 4:
            labels[n] = 0
        else:
            labels[n] = 1

    # Sub-sample 1000 images:
    from sklearn.model_selection import train_test_split

    images, _, labels, _ = train_test_split(
        images, labels, train_size=1000, random_state=0
    )

    # Normalize non-zero entries so that they have mean zero and unit standard
    # across the dataset:'''
    i = images.std(0) != 0
    images[:, i] = (images[:, i] - images[:, i].mean(0)) / images[:, i].std(0)

    # Number of steps, particles and replicates
    num_steps = 500
    num_particles = 10
    num_replicates = 10
    step_sizes = jnp.logspace(-9, 1, num=50)

    # Model
    model = BayesNN()

    # Rmse calculation
    def log_pointwise_predrictive_density(model, latent_particles, images, labels):
        """Returns LPPD for set of (test) images and labels."""

        def _latent_cal(latent):
            def _log_nn(image):
                # Log of the network's output when evaluated at image with weights w, v.
                return jax.nn.log_softmax(
                    jnp.dot(
                        latent["v"], jnp.tanh(jnp.dot(latent["w"], image.flatten()))
                    )
                )

            def _log_nn_vec(images):
                # _log_nn vectorized over particles.
                return jax.vmap(_log_nn)(images)

            return _log_nn_vec(images)

        s = jax.vmap(_latent_cal)(latent_particles).mean(0)
        return vmap(lambda v, l: v[l])(s, labels).mean()

    def _predict(model, latent_particles, images):
        """Returns LPPD for set of (test) images and labels."""

        def _latent_cal(latent):
            def _log_nn(image):
                # Log of the network's output when evaluated at image with weights w, v.
                return jax.nn.log_softmax(
                    jnp.dot(
                        latent["v"], jnp.tanh(jnp.dot(latent["w"], image.flatten()))
                    )
                )

            def _log_nn_vec(images):
                # _log_nn vectorized over particles.
                return jax.vmap(_log_nn)(images)

            return _log_nn_vec(images)

        s = jax.vmap(_latent_cal)(latent_particles).mean(0)
        return jnp.argmax(s, axis=1)

    def test_error(model, latent_particles, images, labels):
        """Returns fraction of misclassified images in test set."""
        return jnp.abs(
            labels.squeeze() - _predict(model, latent_particles, images)
        ).mean()

    from time import perf_counter

    from coinem.zoo import (
        coin_svgd,
        pgd,
        soul,
        adam_svgd,
        ada_svgd,
        rms_pgd,
        rms_soul,
        ada_pgd,
        ada_soul,
        adam_pgd,
        adam_soul,
        rms_svgd,
    )
    from coinem.marginal_zoo import (
        marginal_coin_svgd,
        marginal_pgd,
        marginal_soul,
        marginal_ada_svgd,
        marginal_adam_svgd,
    )

    import pickle

    # Output
    results = {}

    # Store step sizes
    results["step_sizes"] = step_sizes

    # Store runtime
    results["runtime"] = {}
    results["runtime"]["coin_svgd"] = []
    results["runtime"]["adam_svgd"] = []
    results["runtime"]["ada_svgd"] = []
    results["runtime"]["rms_svgd"] = []
    results["runtime"]["marginal_adam_svgd"] = []
    results["runtime"]["marginal_ada_svgd"] = []
    results["runtime"]["marginal_coin_svgd"] = []

    results["runtime"]["pgd"] = []
    results["runtime"]["adam_pgd"] = []
    results["runtime"]["ada_pgd"] = []
    results["runtime"]["rms_pgd"] = []
    results["runtime"]["marginal_pgd"] = []

    results["runtime"]["soul"] = []
    results["runtime"]["adam_soul"] = []
    results["runtime"]["ada_soul"] = []
    results["runtime"]["rms_soul"] = []
    results["runtime"]["marginal_soul"] = []

    # Store AUC
    results["auc"] = {}
    results["auc"]["coin_svgd"] = []
    results["auc"]["adam_svgd"] = []
    results["auc"]["ada_svgd"] = []
    results["auc"]["rms_svgd"] = []
    results["auc"]["marginal_adam_svgd"] = []
    results["auc"]["marginal_ada_svgd"] = []
    results["auc"]["marginal_coin_svgd"] = []

    results["auc"]["pgd"] = []
    results["auc"]["adam_pgd"] = []
    results["auc"]["ada_pgd"] = []
    results["auc"]["rms_pgd"] = []
    results["auc"]["marginal_pgd"] = []

    results["auc"]["soul"] = []
    results["auc"]["adam_soul"] = []
    results["auc"]["ada_soul"] = []
    results["auc"]["rms_soul"] = []
    results["auc"]["marginal_soul"] = []

    # Store error
    results["error"] = {}
    results["error"]["coin_svgd"] = []
    results["error"]["adam_svgd"] = []
    results["error"]["ada_svgd"] = []
    results["error"]["rms_svgd"] = []
    results["error"]["marginal_adam_svgd"] = []
    results["error"]["marginal_ada_svgd"] = []
    results["error"]["marginal_coin_svgd"] = []

    results["error"]["pgd"] = []
    results["error"]["adam_pgd"] = []
    results["error"]["ada_pgd"] = []
    results["error"]["rms_pgd"] = []
    results["error"]["marginal_pgd"] = []

    results["error"]["soul"] = []
    results["error"]["adam_soul"] = []
    results["error"]["ada_soul"] = []
    results["error"]["rms_soul"] = []
    results["error"]["marginal_soul"] = []

    # Store log predictive density
    results["lppd"] = {}
    results["lppd"]["coin_svgd"] = []
    results["lppd"]["adam_svgd"] = []
    results["lppd"]["ada_svgd"] = []
    results["lppd"]["rms_svgd"] = []
    results["lppd"]["marginal_adam_svgd"] = []
    results["lppd"]["marginal_ada_svgd"] = []
    results["lppd"]["marginal_coin_svgd"] = []

    results["lppd"]["pgd"] = []
    results["lppd"]["adam_pgd"] = []
    results["lppd"]["ada_pgd"] = []
    results["lppd"]["rms_pgd"] = []
    results["lppd"]["marginal_pgd"] = []

    results["lppd"]["soul"] = []
    results["lppd"]["adam_soul"] = []
    results["lppd"]["ada_soul"] = []
    results["lppd"]["rms_soul"] = []
    results["lppd"]["marginal_soul"] = []

    from jaxtyping import Array, Float
    from coinem.model import AbstractModel
    from coinem.dataset import Dataset
    from coinem.gradient_transforms import cocob
    from coinem.zoo import svgd

    for seed in range(num_replicates):
        key = jr.PRNGKey(seed)

        # Split data into 80/20 training and testing sets:
        itrain, itest, ltrain, ltest = train_test_split(
            images, labels, test_size=0.2, random_state=seed
        )

        train = Dataset(jnp.array(itrain), jnp.array(ltrain).reshape(-1, 1))
        test = Dataset(jnp.array(itest), jnp.array(ltest).reshape(-1, 1))

        lppd = vmap(
            lambda p: log_pointwise_predrictive_density(
                model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)
            )
        )
        error = vmap(
            lambda p: test_error(
                model, p, jnp.array(itest), jnp.array(ltest).reshape(-1, 1)
            )
        )

        # Initialise the model parameters
        key, subkey = jr.split(key)

        alpha = jnp.array(0.0)
        beta = jnp.array(0.0)

        # Initialize particle cloud by sampling prior:'
        w_init = jnp.exp(alpha) * jr.normal(
            subkey, (num_particles, 40, 28**2)
        )  # Input layer weights.
        v_init = jnp.exp(beta) * jr.normal(
            subkey, (num_particles, 2, 40)
        )  # Output layer weights.

        theta_init = {"alpha": alpha, "beta": beta}
        latent_init = {"w": w_init, "v": v_init}

        import jax.random as jr

        # COIN EM:
        coin_start = perf_counter()
        latent_coin, theta_coin = svgd(model, train, cocob(), cocob(100.0), num_steps)
        coin_end = perf_counter()

        results["runtime"]["coin_svgd"].append(coin_end - coin_start)
        results["error"]["coin_svgd"].append(error(latent_coin))
        results["lppd"]["coin_svgd"].append(lppd(latent_coin))

    with open(f"results/coin_mnist_new.pkl", "wb") as f:
        pickle.dump(results, f)
