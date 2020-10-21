import pytest
import numpy as np

from hmirls.operators import SamplingMatrixOperator


@pytest.fixture
def matrix():
    u = np.array([1.0, 10.0, -2.0, 0.1]).reshape((-1, 1))
    v = np.array([1.0, 2.0, 3.0, 4.0]).reshape((-1, 1))
    X = np.matmul(u, v.transpose())
    return X


@pytest.fixture
def row_indices():
    return [1, 3, 2, 3, 3, 0, 1]


@pytest.fixture
def column_indices():
    return [0, 0, 1, 1, 2, 3, 3]


@pytest.fixture
def input_shape():
    return 4, 4


@pytest.fixture
def sampling_operator(row_indices, column_indices, input_shape):
    return SamplingMatrixOperator(row_indices, column_indices, input_shape)


def test_sampling(sampling_operator, matrix):
    sample = sampling_operator(matrix)
    assert np.allclose(np.squeeze(sample), np.array([10.0, 0.1, -4.0, 0.2, 0.3, 4.0, 40.0]))
