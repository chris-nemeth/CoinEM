import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd

from dataclasses import dataclass

from coinem.model import AbstractModel


@dataclass
class BNN(AbstractModel):
    """Bayesian Neural Network for regression.

    Attributes:
        num_datapoints: Number of datapoints.
        feature_dim: Dimensionality of the input.
        num_hidden: Number of hidden units.
    """

    num_datapoints: int
    feature_dim: int
    num_hidden: int

    def init_params(self, num_particles, a0=1.0, b0=0.1, key=jr.PRNGKey(42)):
        """Initialize the parameters of the model.

        Args:
            num_particles: Number of particles.
            a0: Hyperparameter of the prior Gamma distribution.
            b0: Hyperparameter of the prior Gamma distribution.

        Returns:
            Latent variables and hyperparameters.
        """

        key_w1, key_b1, key_w2, key_b2, key_gamma, key_lambda = jr.split(key, 6)

        w1 = (
            1.0
            / jnp.sqrt(self.feature_dim + 1)
            * jr.normal(key_w1, (num_particles, self.feature_dim, self.num_hidden))
        )
        b1 = jr.normal(key_b1, (num_particles, self.num_hidden))

        w2 = (
            1.0
            / jnp.sqrt(self.num_hidden + 1)
            * jr.normal(key_w2, (num_particles, self.num_hidden))
        )
        b2 = jr.normal(key_b2, (num_particles, 1))

        log_gamma = jnp.log(
            tfd.Gamma(a0, b0).sample(seed=key_gamma, sample_shape=(num_particles, 1))
        )
        log_lambda = jnp.log(
            tfd.Gamma(a0, b0).sample(seed=key_lambda, sample_shape=(num_particles, 1))
        )

        latent = {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "log_gamma": log_gamma,
            "log_lambda": log_lambda,
        }

        theta = {
            "log_gamma": {"a0": jnp.log(a0), "b0": jnp.log(b0)},
            "log_lambda": {"a0": jnp.log(a0), "b0": jnp.log(b0)},
        }
        return latent, theta

    def __call__(self, latent, test_points):
        """Predict the output for the test data."""
        return (
            jnp.dot(
                jax.nn.relu(jnp.dot(test_points, latent["w1"]) + latent["b1"]),
                latent["w2"],
            )
            + latent["b2"]
        )

    def log_prob(self, latent, theta, batch):
        """Compute the log joint density of the model."""
        X, y = batch.X, batch.y
        num_vars = (
            self.feature_dim * self.num_hidden + self.num_hidden * 2 + 3
        )  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 2 variances
        foward = self.__call__(latent, X)
        log_lik_data = -0.5 * X.shape[0] * (
            jnp.log(2 * jnp.pi) - latent["log_gamma"]
        ) - (jnp.exp(latent["log_gamma"]) / 2) * jnp.sum(
            jnp.power(foward.squeeze() - y.squeeze(), 2)
        )
        log_prior_data = (
            (jnp.exp(theta["log_gamma"]["a0"]) - 1) * latent["log_gamma"]
            - jnp.exp(theta["log_gamma"]["b0"]) * jnp.exp(latent["log_gamma"])
            + latent["log_gamma"]
        )
        log_prior_w = (
            -0.5 * (num_vars - 2.0) * (jnp.log(2.0 * jnp.pi) - latent["log_lambda"])
            - (jnp.exp(latent["log_lambda"]) / 2.0)
            * (
                (latent["w1"] ** 2).sum()
                + (latent["w2"] ** 2).sum()
                + (latent["b1"] ** 2).sum()
                + latent["b2"] ** 2
            )
            + (jnp.exp(theta["log_lambda"]["a0"]) - 1) * latent["log_lambda"]
            - jnp.exp(theta["log_lambda"]["b0"]) * jnp.exp(latent["log_lambda"])
            + latent["log_lambda"]
        )

        return (
            log_lik_data * self.num_datapoints / batch.n + log_prior_data + log_prior_w
        ).squeeze()


if __name__ == "__main__":
    from coinem.zoo import coin_svgd, pgd, soul, adam_svgd, ada_svgd
    from coinem.uci import (
        Boston,
        Concrete,
        Energy,
        Kin8nm,
        Naval,
        Power,
        Protein,
        Wine,
        Yacht,
    )
    from jax import vmap
    import pickle

    from time import perf_counter

    # datasets = [
    #     Boston(),
    #     Concrete(),
    #     Energy(),
    #     Kin8nm(),
    # ]

    datasets = [Naval(), Power(), Protein(), Wine(), Yacht()]

    # Number of steps, particles and replicates
    num_steps = 1000
    num_particles = 20
    num_replicates = 10
    step_sizes = jnp.logspace(-9, 1, num=30)

    for data in datasets:
        # Output
        results = {}

        # Store step sizes
        results["step_sizes"] = step_sizes

        # Store runtime
        results["runtime"] = {}
        results["runtime"]["coin"] = []
        results["runtime"]["adam"] = []
        results["runtime"]["ada"] = []
        results["runtime"]["pgd"] = []
        results["runtime"]["soul"] = []

        # Create empty lists for each algorithm
        results["rmse"] = {}
        results["rmse"]["coin"] = []
        results["rmse"]["adam"] = []
        results["rmse"]["ada"] = []
        results["rmse"]["pgd"] = []
        results["rmse"]["soul"] = []

        # Model
        model = BNN(num_datapoints=data.n, feature_dim=data.in_dim, num_hidden=50)

        # Rmse calculation
        def rmse_trace(particles, eval):
            return vmap(
                lambda x_hat: jnp.sqrt(
                    jnp.mean(
                        (eval.y - vmap(lambda x: model(x, eval.X))(x_hat).mean(0)) ** 2
                    )
                )
            )(particles)

        for seed in range(num_replicates):
            key = jr.PRNGKey(seed)

            # Split dataset
            train, test = data.preprocess(test_size=0.1, random_state=seed)

            # Batch size
            if train.n < 500:
                batch_size = -1

            else:
                batch_size = 100

            # Initialise the model parameters
            key, subkey = jr.split(key)
            latent_init, theta_init = model.init_params(num_particles, key=key)

            # Run coinem that is learning rate free
            coin_start = perf_counter()
            latent_coin, theta_coin = coin_svgd(
                model,
                train,
                latent_init,
                theta_init,
                num_steps,
                alpha=100.0,
                batch_size=batch_size,
            )
            coin_end = perf_counter()

            # Store runtime
            results["runtime"]["coin"].append(coin_end - coin_start)

            # Evaluate coinem performance
            results["rmse"]["coin"].append(rmse_trace(latent_coin, test))

            for step_size in step_sizes:
                # Split keys for stochastic algorithms
                key_pgd, key_soul = jr.split(key)

                # Run other algorithms
                adam_start = perf_counter()
                latent_adam, theta_adam = adam_svgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                )
                adam_end = perf_counter()

                adam_start = perf_counter()
                latent_ada, theta_ada = ada_svgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                )
                adam_end = perf_counter()

                pgd_start = perf_counter()
                X_pgd, th_pgd = pgd(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_pgd,
                )
                pgd_end = perf_counter()

                soul_start = perf_counter()
                X_soul, th_soul = soul(
                    model,
                    train,
                    latent_init,
                    theta_init,
                    num_steps,
                    theta_step_size=step_size,
                    latent_step_size=step_size,
                    batch_size=batch_size,
                    key=key_soul,
                )
                soul_end = perf_counter()

                # Store runtime
                results["runtime"]["adam"].append(adam_end - adam_start)
                results["runtime"]["ada"].append(adam_end - adam_start)
                results["runtime"]["pgd"].append(pgd_end - pgd_start)
                results["runtime"]["soul"].append(soul_end - soul_start)

                # Evaluate performance
                results["rmse"]["adam"].append(rmse_trace(latent_adam, test))
                results["rmse"]["ada"].append(rmse_trace(latent_ada, test))
                results["rmse"]["pgd"].append(rmse_trace(X_pgd, test))
                results["rmse"]["soul"].append(rmse_trace(X_soul, test))

        with open(f"results/{data.name}.pkl", "wb") as f:
            pickle.dump(results, f)
