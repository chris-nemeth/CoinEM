import jax.numpy as jnp

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
from coinem.dataset import Dataset

names = [
    "Boston",
    "Concrete",
    "Energy",
    "Kin8nm",
    "Naval",
    "Power",
    "Protein",
    "Wine",
    "Yacht",
]
datasets = [
    Boston(),
    Concrete(),
    Energy(),
    Kin8nm(),
    Naval(),
    Power(),
    Protein(),
    Wine(),
    Yacht(),
]


def test_names() -> None:
    for name, d in zip(names, datasets):
        assert name == d.name


import pytest


@pytest.mark.parametrize("dataset", datasets)
@pytest.mark.parametrize("test_size", [0.1, 0.2])
@pytest.mark.parametrize("scale_X", [True, False])
def test_preprocessing(dataset, test_size, scale_X) -> None:
    assert isinstance(dataset, Dataset)

    train, test = dataset.preprocess(
        test_size=test_size, random_state=42, scale_X=scale_X
    )

    assert isinstance(train, Dataset)
    assert isinstance(test, Dataset)

    assert train.X.shape[0] == train.y.shape[0]
    assert test.X.shape[0] == test.y.shape[0]
    assert train.X.shape[1] == test.X.shape[1]

    new_train, new_test = dataset.preprocess(
        test_size=test_size, random_state=123, scale_X=scale_X
    )

    assert not jnp.allclose(train.X, new_train.X)
    assert not jnp.allclose(train.y, new_train.y)
