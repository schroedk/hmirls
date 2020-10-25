import numpy as np

from hmirls.operators import SamplingMatrixOperator
from hmirls.problem import Problem

"""
Simple example for recovering a low-rank matrix via minimization

    .. math::

        \\min_{x \\in \\mathbb{C}^{d_1 \\times d_2}} \\operatorname{rank}(x), \\text{s.t.} \\Phi(x) = y,


"""

u = np.array([1.0, 10.0, -2.0, 0.1]).reshape((-1, 1))
v = np.array([1.0, 2.0, 3.0, 4.0]).reshape((-1, 1))
X = np.matmul(u, v.transpose())

row_indices = [1, 3, 2, 3, 3, 0, 1]
column_indices = [0, 0, 1, 1, 2, 3, 3]
data = np.ones(len(row_indices))
input_shape = (4, 4)

measurement_operator = SamplingMatrixOperator(row_indices, column_indices, input_shape)
measurements = measurement_operator(X)

problem = Problem(
    measurement_operator,
    measurements,
)

if __name__ == "__main__":
    X_tilde = problem.solve(schatten_p_parameter=0.1, max_iter=1000, rank_estimate=1)
