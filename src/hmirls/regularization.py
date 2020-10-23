from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import diags

from hmirls.svd import ScipySVDEngine, SVDEngine


class RegularizationRule(ABC):

    @abstractmethod
    def compute_regularized_inverse_weights(
        self, x: np.ndarray, schatten_p_parameter: np.float64
    ):
        """
        Abstraction of two regularization functions

        .. math::

            R_1 : \\mathbb{C}^{d_1 \\times d_1} \\rightarrow \\mathbb{C}^{d_1 \\times d_1},
            R_2 : \\mathbb{C}^{d_2 \\times d_2} \\rightarrow \\mathbb{C}^{d_2 \\times d_2}

        representing a regularization rule for :math:`xx^{\\star}`,resp. :math:`x^{\\star}x`

        :param x: array of shape :math:`(d_1, d_2)`
        :param schatten_p_parameter:
        :return: :math:`R_1(xx^{\\star})^{\\frac{2-p}{2}}, R_2(x^{\\star}x)^{\\frac{2-p}{2}}`
        """
        pass


class FixedRankSpectralShiftRegularizationRule(RegularizationRule):
    def __init__(
        self,
        rank_estimate: int,
        minimal_shift=1e-6,
        initial_shift_parameter=1.0,
        svd_engine: SVDEngine = ScipySVDEngine,
    ):
        self._minimal_shift = minimal_shift
        self._shift_parameter = initial_shift_parameter
        self._svd_engine = svd_engine
        self._rank_estimate = rank_estimate

    @property
    def rank_estimate(self):
        return self._rank_estimate

    @property
    def shift_parameter(self):
        return self._shift_parameter

    def compute_regularized_inverse_weights(
        self, x: np.ndarray, schatten_p_parameter: np.float64
    ):
        """
        Regularization via spectral shift

        .. math::

            R_1 : \\mathbb{C}^{d_1 \\times d_1} \\rightarrow \\mathbb{C}^{d_1 \\times d_1}, x \\mapsto xx^{\\star}+\\varepsilon \cdot Id_{d_1},

            R_2 : \\mathbb{C}^{d_2 \\times d_2} \\rightarrow \\mathbb{C}^{d_2 \\times d_2}, x \\mapsto x^{\\star}x +\\varepsilon \cdot Id_{d_2}.

        Spectral shift :math:`\\varepsilon` is given by

        .. math::

            \\varepsilon = \\min(\\max(\\sigma_{\\text{rank_estimate}}(x), \\text{minimal_shift}), \\text{shift_parameter}),

        where :math:`\sigma_{\\text{rank_estimate}}(x)` is the rank_estimate-th singular value of x.

        :param x: array of shape :math:`(d_1, d_2)`
        :param schatten_p_parameter:
        :return: :math:`\\left(xx^{\\star}+\\varepsilon \cdot Id_{d_1}\\right)^{\\frac{2-p}{2}}, \\left(x^{\\star}x +\\varepsilon \cdot Id_{d_2}\\right)^{\\frac{2-p}{2}}`
        """

        u, s, v = self._svd_engine.svd(x)
        min_non_zero_dim = min(u.shape[0], v.shape[0])
        regularized_singular_values = self._compute_regularized_singular_values(
            s, self._shift_parameter, schatten_p_parameter
        )
        left_inverse_weight = self._construct_one_sided_inverse_weight(
            u, regularized_singular_values, min_non_zero_dim
        )
        right_inverse_weight = self._construct_one_sided_inverse_weight(
            v, regularized_singular_values, min_non_zero_dim
        )
        self._update_shift_parameter(s)
        return left_inverse_weight, right_inverse_weight

    def _update_shift_parameter(self, singular_values):
        self._shift_parameter = min(
            max(singular_values[self.rank_estimate], self._minimal_shift),
            self._shift_parameter,
        )

    def _construct_one_sided_inverse_weight(
        self,
        singular_vectors: np.ndarray,
        regularized_singular_values,
        min_non_zero_dim,
    ):
        dimension = singular_vectors.shape[0]
        diagonal_matrix = self._construct_shifted_diagonal_matrix(
            dimension, regularized_singular_values, min_non_zero_dim
        )
        return singular_vectors @ diagonal_matrix @ singular_vectors.conj().transpose()

    @staticmethod
    def _construct_shifted_diagonal_matrix(
        dimension, regularized_singular_values, min_non_zero_dim
    ):
        _regularized_singular_values = np.zeros(dimension)
        _regularized_singular_values[:min_non_zero_dim] = regularized_singular_values
        _regularized_singular_values[min_non_zero_dim:] = 0
        return diags(_regularized_singular_values, shape=(dimension, dimension))

    @staticmethod
    def _compute_regularized_singular_values(
        singular_values, regularization_parameter, schatten_p_parameter
    ):
        return np.power(
            np.power(singular_values, 2) + regularization_parameter ** 2,
            (2 - schatten_p_parameter) / 2.0,
        )
