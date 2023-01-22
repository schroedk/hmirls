from enum import Enum
from typing import Tuple, List, Union, Callable, Optional
import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import LinearOperator, aslinearoperator


class MatrixOperatorCompatibility:
    """
    Compatibility check for objects of :class:`~MatrixOperator`
    """

    @staticmethod
    def compatible_for_addition(
        first_operator: "MatrixOperator", second_operator: "MatrixOperator"
    ):

        same_input_shape = first_operator.input_shape == second_operator.input_shape
        same_flattened_shape = (
            first_operator.flattened_operator.shape
            == second_operator.flattened_operator.shape
        )
        same_dtype = MatrixOperatorCompatibility._same_dtype(
            first_operator, second_operator
        )
        same_indexing_order = MatrixOperatorCompatibility._same_indexing_order(
            first_operator, second_operator
        )
        return (
            same_flattened_shape
            and same_input_shape
            and same_dtype
            and same_indexing_order
        )

    @staticmethod
    def compatible_for_composition(
        first_operator: "MatrixOperator", second_operator: "MatrixOperator"
    ):
        same_dtype = MatrixOperatorCompatibility._same_dtype(
            first_operator, second_operator
        )
        same_indexing_order = MatrixOperatorCompatibility._same_indexing_order(
            first_operator, second_operator
        )
        compatible_shapes = first_operator.input_shape == second_operator.output_shape
        return same_dtype and same_indexing_order and compatible_shapes

    @staticmethod
    def _same_dtype(
        first_operator: "MatrixOperator", second_operator: "MatrixOperator"
    ):
        return (
            first_operator.flattened_operator.dtype
            == second_operator.flattened_operator.dtype
        )

    @staticmethod
    def _same_indexing_order(
        first_operator: "MatrixOperator", second_operator: "MatrixOperator"
    ):
        return first_operator.order == second_operator.order


class MatrixOperator:
    class IndexingOrder(Enum):
        """
        Enum for order argument for :func:`numpy.reshape` and :func:`numpy.flatten`
        """
        ROW_MAJOR = "C"
        COLUMN_MAJOR = "F"

    def __init__(
        self,
        flattened_operator: Union[LinearOperator, np.ndarray, csr_matrix],
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        order: IndexingOrder,
        representing_matrix: Optional[Union[np.ndarray, csr_matrix]] = None,
    ):
        """
        Represents an operator

        .. math::

            \\Phi: \\mathbb{C}^{d_1 \\times d_2} \\rightarrow \\mathbb{C}^{m_1 \\times m_2}.

        Takes care on flattening and reshaping. Wraps a scipy linear operator or numpy array.

        :param flattened_operator: Represents the flattened map

            .. math::

                \\varphi: \\mathbb{C}^{d_1 \\cdot d_2} \\rightarrow \\mathbb{C}^{m_1 \\cdot m_2},

            such that :math:`\\varphi(\\operatorname{vec}(x)) = \\operatorname{vec}(\\Phi(x))`.

        :param input_shape: :math:`(d_1,d_2)`
        :param output_shape: :math:`(m_1,m_2)`
        :param order: order for vectorization :math:`\\operatorname{vec}`
        :param representing_matrix: matrix representation of flattened operator

        """
        self._output_shape = output_shape
        self._input_shape = input_shape
        self._order = order
        self._representing_matrix = representing_matrix
        if isinstance(flattened_operator, np.ndarray):
            self._representing_matrix = flattened_operator
            flattened_operator = aslinearoperator(flattened_operator)
        self._flattened_operator = flattened_operator

    def __add__(self, other: "MatrixOperator"):
        if isinstance(other, MatrixOperator):
            if MatrixOperatorCompatibility.compatible_for_addition(self, other):
                representing_matrix = None
                if (
                    self._representing_matrix is not None
                    and other._representing_matrix is not None
                ):
                    representing_matrix = (
                        self._representing_matrix + other._representing_matrix
                    )
                return MatrixOperator(
                    self._flattened_operator + other._flattened_operator,
                    self._input_shape,
                    self._output_shape,
                    self._order,
                    representing_matrix,
                )
            raise ValueError(f"Operators are not compatible")
        return NotImplemented

    def __call__(self, x):
        return self * x

    def __mul__(self, x):
        return self.dot(x)

    def dot(self, x):
        """Matrix-matrix multiplication (operator application) or operator composition.

        Parameters
        ----------
        x : array_like of shape :math:`(d_1, d_2)` or MatrixOperator compatible for composition or scalar

        Returns
        -------
            :math:`\\Phi(x)` : array or MatrixOperator that represents
            the result of applying this linear operator on x.

        """
        if isinstance(x, MatrixOperator):
            if MatrixOperatorCompatibility.compatible_for_composition(self, x):
                representing_matrix = None
                if (
                    self.representing_matrix is not None
                    and x.representing_matrix is not None
                ):
                    representing_matrix = (
                        self.representing_matrix @ x.representing_matrix
                    )
                return MatrixOperator(
                    self.flattened_operator.dot(x.flattened_operator),
                    x.input_shape,
                    self.output_shape,
                    self.order,
                    representing_matrix,
                )
        elif np.isscalar(x):
            return MatrixOperator._from_callable(self, lambda op: x * op)
        else:
            if x.shape == self.input_shape:
                x_flattened = x.flatten(order=self.order.value)
                return self.flattened_operator(x_flattened).reshape(
                    self.output_shape, order=self.order.value
                )
            else:
                raise ValueError(
                    f"expected 2-d array or matrix of shape{self.input_shape}, got {x}"
                )

    def __matmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, " "use '*' instead")
        return self.__mul__(other)

    def __rmatmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, " "use '*' instead")
        return self.__rmul__(other)

    def __rmul__(self, x):
        if np.isscalar(x):
            return self.dot(x)
        else:
            return NotImplemented

    def __pow__(self, p):
        if np.isscalar(p):
            if self.input_shape != self.output_shape:
                raise ValueError("square LinearOperator expected, got %r" % self)
            return MatrixOperator._from_callable(self, lambda op: op ** p)
        else:
            return NotImplemented

    def __neg__(self):
        return MatrixOperator._from_callable(self, lambda op: -op)

    def __sub__(self, x):
        return self.__add__(-x)

    def __repr__(self):
        if self.flattened_operator.dtype is None:
            dt = "unspecified dtype"
        else:
            dt = "dtype=" + str(self.flattened_operator.dtype)

        return f"{(self.output_shape, self.input_shape, self.__class__.__name__, dt)}"

    @classmethod
    def _from_callable(
        cls,
        matrix_operator: "MatrixOperator",
        fcn: Callable[
            [Union[LinearOperator, np.ndarray]],
            Union[LinearOperator, np.ndarray],
        ],
    ):
        """
        Instantiation of a :class: ~`MatrixOperator` from an :class: ~`MatrixOperator` object and a callable
        :param matrix_operator:
        :param fcn:
        :return:
        """
        representing_matrix = None
        if matrix_operator.representing_matrix is not None:
            representing_matrix = fcn(matrix_operator.representing_matrix)
        return cls(fcn(matrix_operator.flattened_operator),
                   matrix_operator.input_shape,
                   matrix_operator.output_shape,
                   matrix_operator.order,
                   representing_matrix,
                   )

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def order(self):
        return self._order

    @property
    def flattened_operator(self):
        return self._flattened_operator

    @property
    def representing_matrix(self):
        return self._representing_matrix

    def adjoint(self):
        """Hermitian adjoint.

        Returns a :class:`~MatrixOperator` that represents the Hermitian adjoint of this one

        Can be abbreviated self.H instead of self.adjoint().

        Returns
        -------
        :math:`\\Phi^{\\star}` : :class:`~MatrixOperator`
            Hermitian adjoint of self.
        """
        return self._adjoint()

    H = property(adjoint)

    def transpose(self):
        """Transpose this linear operator.

        Returns a :class:`~MatrixOperator` that represents the transpose of this one.
        Can be abbreviated self.T instead of self.transpose().

        Returns
        -------
        :math:`\\Phi^{\\star}` : :class:`~MatrixOperator`
             adjoint of self.
        """

        return self._transpose()

    T = property(transpose)

    def _adjoint(self):
        """Default implementation of _adjoint"""
        representing_matrix = None
        if self.representing_matrix is not None:
            representing_matrix = self._representing_matrix.conjugate().transpose()
        return MatrixOperator(
            self._flattened_operator.H,
            self._output_shape,
            self._input_shape,
            self.order,
            representing_matrix,
        )

    def _transpose(self):
        """ Default implementation of _transpose"""
        return MatrixOperator._from_callable(self, lambda op: op.T)


class SamplingOperator(LinearOperator):
    def __init__(self, indices: Union[List[int], np.ndarray], input_dimension: int):
        """

        :type input_dimension:
        :type indices: linear indices
        """
        super().__init__(dtype=np.float64, shape=(len(indices), input_dimension))
        self._indices = indices
        self._sampling_matrix = self._construct_sampling_matrix()
        self._input_dimension = input_dimension

    @property
    def indices(self):
        return self._indices

    @property
    def input_dimension(self):
        return self._input_dimension

    @property
    def sampling_matrix(self):
        return self._sampling_matrix

    def _matvec(self, x: np.array):
        return x[self.indices]

    def _rmatvec(self, x: np.array):
        return csr_matrix(
            (x.flatten(), (self.indices, np.zeros(len(self.indices)))),
            shape=(self.shape[1], 1),
            dtype=self.dtype,
        ).toarray()

    def _construct_sampling_matrix(self):
        data = np.ones(len(self.indices))
        row_indices = list(range(len(self.indices)))
        column_indices = self.indices
        return csr_matrix((data, (row_indices, column_indices)), shape=self.shape)

    @classmethod
    def from_matrix_indices(
        cls, row_indices: List[int], column_indices: List[int], shape: Tuple[int, int],
            order: MatrixOperator.IndexingOrder = MatrixOperator.IndexingOrder.COLUMN_MAJOR
    ):
        if len(row_indices) != len(column_indices):
            raise ValueError("Row and column indices must be of same length")

        return cls(
            np.ravel_multi_index((row_indices, column_indices), shape, order=order.value),
            input_dimension=np.product(shape),
        )


class SamplingMatrixOperator(MatrixOperator):
    def __init__(
        self,
        row_indices: List[int],
        column_indices: List[int],
        shape: Tuple[int, int],
        order: MatrixOperator.IndexingOrder = MatrixOperator.IndexingOrder.COLUMN_MAJOR,
    ):
        flattened_operator = SamplingOperator.from_matrix_indices(
            row_indices, column_indices, shape, order
        )
        super().__init__(
            flattened_operator,
            shape,
            (len(row_indices), 1),
            order,
            flattened_operator.sampling_matrix,
        )

    @classmethod
    def from_index_tuples(cls, indices: List[int], shape: Tuple[int, int]):
        row_indices = []
        column_indices = []
        for (row_idx, col_idx) in indices:
            row_indices.append(row_idx)
            column_indices.append(col_idx)
        return cls(row_indices, column_indices, shape)


class InverseWeightOperator(MatrixOperator):
    def __init__(
        self,
        left_inverse_weight: np.ndarray,
        right_inverse_weight: np.ndarray,
        order,
    ):
        """
        Constructs inverse weight operator from one sided inverse weights.

        :param left_inverse_weight:
        :param right_inverse_weight:
        :param order:
        """
        shape = (left_inverse_weight.shape[0], right_inverse_weight.shape[0])
        weight_matrix = 0.5 * (
            kron(eye(right_inverse_weight.shape[0]), left_inverse_weight)
            + kron(right_inverse_weight, eye(left_inverse_weight.shape[0]))
        )
        flattened_operator = aslinearoperator(weight_matrix)
        super().__init__(
            flattened_operator,
            shape,
            shape,
            order,
            weight_matrix,

        )
