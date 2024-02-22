import numpy as np
from utils.echelon_tools import column_reduction, swap_rows, find_nonzero_index


def __lu_decomposition_without_pivoting(matrix: np.ndarray) -> np.ndarray:
    _, cols_count = matrix.shape
    for base_index in range(cols_count - 1):
        if matrix[base_index, base_index] == 0:
            raise Exception(f'The diagonal element [{base_index}, {base_index}] is Zero')
        column_reduction(matrix, base_index)
    return np.identity(cols_count, dtype=np.int8)


def __lu_decomposition_essential_pivoting(matrix: np.ndarray) -> np.ndarray:
    _, cols_count = matrix.shape
    pivots = np.identity(cols_count, dtype=np.int8)
    for base_index in range(cols_count - 1):
        if matrix[base_index, base_index] == 0:
            nonzero_index = find_nonzero_index(matrix, base_index)
            if nonzero_index == -1:
                raise Exception(f'The matrix is not full rank, there is no nonzero for [{base_index},{base_index}]')
            swap_rows(matrix, pivots, nonzero_index, base_index)
        column_reduction(matrix, base_index)
    return pivots


def __lu_decomposition_partial_pivoting(matrix: np.ndarray) -> np.ndarray:
    _, cols_count = matrix.shape
    pivots = np.identity(cols_count, dtype=np.int8)
    for base_index in range(cols_count - 1):
        max_index = np.argmax(matrix[base_index:, base_index])
        swap_rows(matrix, pivots, max_index + base_index, base_index)
        column_reduction(matrix, base_index)
    return pivots


pivoting_options = [
    __lu_decomposition_without_pivoting,
    __lu_decomposition_essential_pivoting,
    __lu_decomposition_partial_pivoting,
]


def lu_decomposition(matrix: np.ndarray, pivoting: int = 1):
    rows_count, cols_count = matrix.shape
    if rows_count != cols_count:
        raise Exception('the matrix must be square')
    if pivoting >= len(pivoting_options):
        raise Exception('the pivoting is not valid')
    purposed_function = pivoting_options[pivoting]
    if not callable(purposed_function):
        raise Exception('there is no callable function for the desired pivoting')
    reformatted_matrix = matrix.astype(np.float64)
    pivots = purposed_function(reformatted_matrix)
    return reformatted_matrix, pivots
