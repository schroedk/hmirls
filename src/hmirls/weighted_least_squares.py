from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator, cg, lsqr, spsolve, minres
from scipy.sparse import eye, csr_matrix
from .operators import MatrixOperator


class WeightedLeastSquaresSolver(ABC):
    """
    Abstract base class for weighted least square solvers
    """

    def solve(
        self,
        operator: Union[np.ndarray, MatrixOperator],
        inverse_weight_operator: MatrixOperator,
        data: np.ndarray,
    ):
        """

        :param operator: :class:`~MatrixOperator`
        :param inverse_weight_operator: :class:`~MatrixOperator`
        :param data:
        :return:
        """
        combined_operator = operator * inverse_weight_operator * operator.H
        x = self._solve_linear_equation(combined_operator.flattened_operator, data)
        x = x.reshape(
            operator.output_shape,
            order=operator.order.value,
        )
        return (inverse_weight_operator * operator.H)(x)

    @abstractmethod
    def _solve_linear_equation(
        self, operator: MatrixOperator, data: np.ndarray
    ) -> np.ndarray:
        pass


class ScipyCgWeightedLeastSquaresSolver(WeightedLeastSquaresSolver):
    def __init__(self, tol=1e-10):
        self._tol = tol

    def _solve_linear_equation(
        self, operator: ScipyLinearOperator, data: np.ndarray
    ) -> np.ndarray:
        x, _ = cg(operator, data, tol=self._tol)
        return x


class ScipyLsqrWeightedLeastSquareSolver(WeightedLeastSquaresSolver):

    def _solve_linear_equation(self, operator: MatrixOperator, data: np.ndarray) -> np.ndarray:
        x = lsqr(operator, data)
        return x[0]


class ScipyDirect(WeightedLeastSquaresSolver):

    def _solve_linear_equation(self, operator: MatrixOperator, data: np.ndarray) -> np.ndarray:
        x = spsolve(csr_matrix(operator @ np.eye(*operator.shape)), data)
        return x


class ScipyMinRes(WeightedLeastSquaresSolver):
    def __init__(self, tol: float = 1e-5):
        self._tol = tol

    def _solve_linear_equation(self, operator: MatrixOperator, data: np.ndarray) -> np.ndarray:
        x = minres(operator, data, tol=self._tol)
        return x[0]