import numpy as np
from reduce_column import reduce_column


def lu_decomposition_without_pivoting(matrix: np.ndarray) -> None:
    _, cols_count = matrix.shape
    for base_index in range(0, cols_count - 1):
        if matrix[base_index, base_index] == 0:
            raise Exception(f'The diagonal element [{base_index}, {base_index}] is Zero')
        reduce_column(matrix, base_index)
    return None


def lu_decomposition_essential_pivoting(matrix: np.ndarray):
    return None


def lu_decomposition_partial_pivoting(matrix: np.ndarray):
    return None


pivoting_options = [
    lu_decomposition_without_pivoting,
    lu_decomposition_essential_pivoting,
    lu_decomposition_partial_pivoting,
]


def lu_decomposition(matrix: np.ndarray, pivoting: int = 0):
    if pivoting >= len(pivoting_options):
        raise Exception('the pivoting is not valid')
    purposed_function = pivoting_options[pivoting]
    if not callable(purposed_function):
        raise Exception('there is no callable function for the desired pivoting')
    return purposed_function(matrix.astype(np.float64))


M = np.array([[2, 3, 4], [4, 6, 8], [4, 7, 2]])
print(M[1, 1:])
print(lu_decomposition(M))
