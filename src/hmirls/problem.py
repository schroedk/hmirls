from abc import ABC, abstractmethod
from typing import Union
from scipy.sparse.linalg import (
    LinearOperator as scipyLinearOperator,
    aslinearoperator,
)
import numpy as np
from scipy.sparse import eye
from .operators import MatrixOperator, InverseWeightOperator

from .svd import SVDEngine, ScipySVDEngine
from .weighted_least_squares import (
    WeightedLeastSquaresSolver,
    ScipyCgWeightedLeastSquaresSolver,
)


class StoppingCriteria(ABC):
    @abstractmethod
    def satisfied(self, *args, **kwargs) -> bool:
        pass


class ResidualNormStoppingCriteria(StoppingCriteria):
    def __init__(self, tol=1e-9):
        self._tol = tol

    @property
    def tol(self):
        return self._tol

    def satisfied(self, previous_iterate: np.ndarray, current_iterate: np.ndarray):
        return (
            np.linalg.norm(previous_iterate - current_iterate)
            / np.linalg.norm(previous_iterate)
            < self.tol
        )


class Problem:
    def __init__(
        self,
        measurement_operator: Union[scipyLinearOperator, np.ndarray, MatrixOperator],
        data: np.array,
        rank_estimate: int,
        indexing_order="F",
        input_shape=None,
    ):
        """

        :param measurement_operator:
        :param data:
        :param indexing_order:
        :param input_shape:
        """

        if not isinstance(measurement_operator, MatrixOperator):
            if indexing_order is None or input_shape is None:
                raise ValueError(
                    f"In case provided operator is not wrapped as MatrixOperator, "
                    f"provide indexing order and input shape"
                )
            measurement_operator = MatrixOperator(
                measurement_operator, input_shape, data.shape, order=indexing_order
            )
        self.data = data
        self.rank_estimate = rank_estimate
        self.measurement_operator = measurement_operator

    def solve(
        self,
        schatten_p_parameter,
        max_iter=1000,
        eps_min=1e-12,
        svd_engine: SVDEngine = ScipySVDEngine,
        weighted_least_squares_solver: WeightedLeastSquaresSolver = ScipyCgWeightedLeastSquaresSolver(
            tol=1e-10
        ),
        stopping_criteria: StoppingCriteria = ResidualNormStoppingCriteria(tol=1e-9),
    ):
        """

        :param stopping_criteria:
        :param schatten_p_parameter:
        :param max_iter:
        :param eps_min:
        :param svd_engine:
        :param weighted_least_squares_solver:
        :return:
        """

        def _initialize_inverse_weight_matrix_operator():
            """
            Initialize inverse weight matrix operator with identity matrix
            :return: MatrixOperator
            """
            flattened_input_length = np.prod(self.measurement_operator.input_shape)
            order = self.measurement_operator.order
            input_shape = self.measurement_operator.input_shape
            return MatrixOperator(
                aslinearoperator(eye(flattened_input_length)),
                input_shape,
                input_shape,
                representing_matrix=eye(flattened_input_length),
                order=order,
            )

        def _compute_inverse_weight_matrix_operator(
            left_singular_vectors: np.ndarray,
            right_singular_vectors: np.ndarray,
            singular_values: np.array,
            _regularization_parameter: np.float64,
        ):
            return InverseWeightOperator(
                left_singular_vectors,
                right_singular_vectors,
                singular_values,
                schatten_p_parameter,
                _regularization_parameter,
                order=self.measurement_operator.order,
            )

        inverse_weight_matrix_operator = _initialize_inverse_weight_matrix_operator()
        regularization_parameter = 1.0
        iteration = 0
        old_result = np.random.randn(*self.measurement_operator.input_shape)
        while iteration < max_iter:
            print(iteration)
            result = weighted_least_squares_solver.solve(
                self.measurement_operator, inverse_weight_matrix_operator, self.data
            )
            if stopping_criteria.satisfied(old_result, result):
                return result
            old_result = result
            u, s, v = svd_engine.svd(result)
            regularization_parameter = min(
                max(s[self.rank_estimate], eps_min), regularization_parameter
            )
            inverse_weight_matrix_operator = _compute_inverse_weight_matrix_operator(
                u, v, s, regularization_parameter
            )
            iteration += 1
        return result
