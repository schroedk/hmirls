from enum import Enum
from typing import Tuple, List, Union, Callable
import numpy as np
from scipy.sparse import csr_matrix, kron, diags, eye
from scipy.sparse.linalg import LinearOperator as scipyLinearOperator, aslinearoperator


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
        ROW_MAJOR = "C"
        COLUMN_MAJOR = "F"

    def __init__(
        self,
        flattened_operator: Union[scipyLinearOperator, np.ndarray, csr_matrix],
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        representing_matrix: Union[np.ndarray, csr_matrix] = None,
        order=IndexingOrder.COLUMN_MAJOR.value,
    ):
        """
        # ToDO More detailed description, doc tests
        Takes care on flattening and reshaping.
        Wraps a scipy linear operator or numpy array.

        :type output_shape:
        :param flattened_operator:
        :param input_shape:
        :param order:
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
                    representing_matrix,
                    self._order,
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
        x : array_like or MatrixOperator or scalar

        Returns
        -------
        Ax : array or MatrixOperator that represents
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
                    input_shape=x.input_shape,
                    output_shape=self.output_shape,
                    representing_matrix=representing_matrix,
                    order=self.order,
                )
        elif np.isscalar(x):
            return MatrixOperator._from_callable(self, lambda op: x * op)
        else:
            x = np.asarray(x)

            if x.shape == self.input_shape:
                x_flattened = x.flatten(self.order)
                return self.flattened_operator(x_flattened).reshape(
                    self.output_shape, order=self.order
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
            [Union[scipyLinearOperator, np.ndarray]],
            Union[scipyLinearOperator, np.ndarray],
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
        return cls(
            fcn(matrix_operator.flattened_operator),
            input_shape=matrix_operator.input_shape,
            output_shape=matrix_operator.output_shape,
            representing_matrix=representing_matrix,
            order=matrix_operator.order,
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
        A_H : :class:`~MatrixOperator`
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
        A_T : :class:`~MatrixOperator`
             adjoint of self.
        """

        return self._transpose()

    T = property(transpose)

    def _adjoint(self):
        """Default implementation of _adjoint"""
        representing_matrix = None
        if self.representing_matrix is not None:
            representing_matrix = self._representing_matrix.H
        return MatrixOperator(
            self._flattened_operator.H,
            self._output_shape,
            self._input_shape,
            representing_matrix,
            self.order,
        )

    def _transpose(self):
        """ Default implementation of _transpose"""
        representing_matrix = None
        if self.representing_matrix is not None:
            representing_matrix = self._representing_matrix.T
        return MatrixOperator(
            self._flattened_operator.T,
            self._output_shape,
            self._input_shape,
            representing_matrix,
            self.order,
        )


class SamplingOperator(scipyLinearOperator):
    def __init__(self, indices: List[int], input_dimension: int):
        """

        :type input_dimension:
        :type indices: linear indices
        :param dtype: data type of representing matrix, defaults to double,
                      e.g. for complex operators choose np.complex_
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
        # ToDo proper dimension check
        # ToDO check if toarray can be ommitted
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
        cls, row_indices: List[int], column_indices: List[int], shape, order="F"
    ):
        return cls(
            np.ravel_multi_index((row_indices, column_indices), shape, order=order),
            input_dimension=np.prod(shape),
        )


class SamplingMatrixOperator(MatrixOperator):
    def __init__(
        self,
        row_indices: List[int],
        column_indices: List[int],
        shape: Tuple[int, int],
        order="F",
    ):
        flattened_operator = SamplingOperator.from_matrix_indices(
            row_indices, column_indices, shape, order
        )
        super(SamplingMatrixOperator, self).__init__(
            flattened_operator,
            shape,
            (len(row_indices), 1),
            flattened_operator.sampling_matrix,
            order,
        )


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
            representing_matrix=weight_matrix,
            order=order,
        )
