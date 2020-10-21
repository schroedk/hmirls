from abc import ABC, abstractmethod
from typing import Union, Tuple
from scipy.sparse.linalg import (
    LinearOperator as scipyLinearOperator,
    aslinearoperator,
)
import numpy as np
from scipy.sparse import eye
from .operators import MatrixOperator, InverseWeightOperator
from .regularization import RegularizationRule, FixedRankSpectralShiftRegularizationRule

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
        indexing_order: MatrixOperator.IndexingOrder = None,
        input_shape: Tuple[int, int] = None,
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
        self.measurement_operator = measurement_operator

    def solve(
        self,
        schatten_p_parameter,
        max_iter=1000,
        rank_estimate: int = None,
        regularization_rule: RegularizationRule = None,
        weighted_least_squares_solver: WeightedLeastSquaresSolver = ScipyCgWeightedLeastSquaresSolver(
            tol=1e-10
        ),
        stopping_criteria: StoppingCriteria = ResidualNormStoppingCriteria(tol=1e-9),
    ):
        """

        :param schatten_p_parameter:
        :param max_iter:
        :param rank_estimate:
        :param regularization_rule:
        :param weighted_least_squares_solver:
        :param stopping_criteria:
        :return:
        """

        if regularization_rule is None:
            if rank_estimate is None:
                raise ValueError(
                    f"Provide either a rank estimate or a regularization rule."
                )
            regularization_rule = FixedRankSpectralShiftRegularizationRule(
                rank_estimate
            )

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

        inverse_weight_matrix_operator = _initialize_inverse_weight_matrix_operator()
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
            (
                left_inverse_weight,
                right_inverse_weight,
            ) = regularization_rule.compute_regularized_inverse_weights(
                result, schatten_p_parameter
            )
            inverse_weight_matrix_operator = InverseWeightOperator(
                left_inverse_weight,
                right_inverse_weight,
                self.measurement_operator.order,
            )
            iteration += 1
        return result
