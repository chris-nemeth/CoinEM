from coinem.dataset import Dataset
from dataclasses import dataclass


@dataclass
class BayesianNeuralNetworkLearn(AbstractModel):
    """Bayesian Neural Network for regression."""

    num_datapoints: int
    feature_dim: int
    num_hidden: int = 50

    def init_params(self, num_particles, a0=1.0, b0=0.1, key=jr.PRNGKey(42)):
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

    def predict(self, latent, X_test):
        foward = (
            jnp.dot(
                jax.nn.relu(jnp.dot(X_test, latent["w1"]) + latent["b1"]), latent["w2"]
            )
            + latent["b2"]
        )
        return foward

    def log_prob(self, latent, theta, batch):
        X, y = batch.X, batch.y

        # theta = {"log_gamma": {"a0": jnp.log(1.0), "b0": jnp.log(0.1)}, "log_lambda": {"a0": jnp.log(1.0), "b0": jnp.log(0.1)}}

        num_vars = (
            self.feature_dim * self.num_hidden + self.num_hidden * 2 + 3
        )  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 2 variances

        foward = (
            jnp.dot(jax.nn.relu(jnp.dot(X, latent["w1"]) + latent["b1"]), latent["w2"])
            + latent["b2"]
        )
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

        # # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations

        return (
            log_lik_data * self.num_datapoints / batch.n + log_prior_data + log_prior_w
        ).squeeze()


if __name__ == "__main__":
    from uci import Boston

    names = ["Boston"]
    datasets = [Boston()]
