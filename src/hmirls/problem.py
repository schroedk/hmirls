from abc import ABC, abstractmethod
from typing import Union, Tuple
import logging
import scipy
import numpy as np
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import LinearOperator

from .operators import MatrixOperator, InverseWeightOperator
from .regularization import RegularizationRule, FixedRankSpectralShiftRegularizationRule

from .weighted_least_squares import (
    WeightedLeastSquaresSolver,
    ScipyCgWeightedLeastSquaresSolver,
)

log = logging.getLogger(__name__)


class StoppingCriteria(ABC):
    """
    Abstract base class for iteration stopping criteria
    """
    @abstractmethod
    def satisfied(self, previous_iterate: np.ndarray, current_iterate: np.ndarray) -> bool:
        """
        Specifies if criteria is satisfied.
        :param current_iterate:
        :param previous_iterate:
        :return:
        """
        pass


class ResidualNormStoppingCriteria(StoppingCriteria):
    def __init__(self, tol=1e-9):
        """
        Stopping criteria based on the relative residual between two iterations is less than a given tolerance
        :param tol: tolerance for residual
        """
        self._tol = tol
        self._log = log.getChild(self.__class__.__name__)

    @property
    def tol(self):
        return self._tol

    def satisfied(self, previous_iterate: np.ndarray, current_iterate: np.ndarray):
        """
        Returns true if

        .. math::

            \\frac{\\|X_{\\text{prev}} - X_{\\text{curr}}\\||_{F}}{\\|X_{\\text{prev}}\\|_{F}} < tol.

        :param previous_iterate: :math:`X_{\\text{prev}`
        :param current_iterate: :math:`X_{\\text{curr}`
        :return:
        """
        residual = np.linalg.norm(previous_iterate - current_iterate) / np.linalg.norm(previous_iterate)
        self._log.info(f"Current {residual=}")
        return (
                residual
                < self.tol
        )


class Problem:
    def __init__(
        self,
        measurement_operator: Union[LinearOperator, np.ndarray, csr_matrix, MatrixOperator],
        data: np.array,
        indexing_order: MatrixOperator.IndexingOrder = None,
        input_shape: Tuple[int, int] = None,
    ):
        """
        Description for a rank minimization problem

        .. math::

            \\min_{x \\in \\mathbb{C}^{d_1 \\times d_2}} \\operatorname{rank}(x), \\text{s.t.} \\Phi(x) = y,


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
                measurement_operator, input_shape, data.shape, indexing_order
            )

        self.data = data
        self.measurement_operator = measurement_operator
        self._log = log.getChild(self.__class__.__name__)

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
            flattened_input_length = np.product(self.measurement_operator.input_shape)
            order = self.measurement_operator.order
            input_shape = self.measurement_operator.input_shape
            return MatrixOperator(
                scipy.sparse.linalg.aslinearoperator(eye(flattened_input_length)),
                input_shape,
                input_shape,
                representing_matrix=eye(flattened_input_length),
                order=order,
            )

        inverse_weight_matrix_operator = _initialize_inverse_weight_matrix_operator()
        iteration = 0
        old_result = np.random.randn(*self.measurement_operator.input_shape)
        result = None
        while iteration < max_iter:
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
            self._log.info(f"Finished {iteration=}")
            iteration += 1
        return result
