import numpy as np
from modules import lu_decomposition
from utils.echelon_tools import get_lower_triangular, get_upper_triangular

matrix = np.random.rand(100, 100) * 10 + 2
decompose_matrix_2, pivots_2 = lu_decomposition(matrix, 2)
low_2 = get_lower_triangular(decompose_matrix_2)
up_2 = get_upper_triangular(decompose_matrix_2)

np.testing.assert_array_almost_equal(np.linalg.inv(pivots_2) @ low_2 @ up_2, matrix, decimal=12)

print("test of decomposition with partial pivoting completed")

decompose_matrix_1, pivots_1 = lu_decomposition(matrix, 1)
low_1 = get_lower_triangular(decompose_matrix_1)
up_1 = get_upper_triangular(decompose_matrix_1)

np.testing.assert_array_almost_equal(np.linalg.inv(pivots_1) @ low_1 @ up_1, matrix, decimal=12)
print("test of decomposition with essential pivoting completed")
