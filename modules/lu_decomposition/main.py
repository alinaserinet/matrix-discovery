import numpy
import numpy as np
from reduce_column import reduce_column
from swap_rows import swap_rows


def lu_decomposition_without_pivoting(matrix: np.ndarray) -> None:
    _, cols_count = matrix.shape
    for base_index in range(cols_count - 1):
        if matrix[base_index, base_index] == 0:
            raise Exception(f'The diagonal element [{base_index}, {base_index}] is Zero')
        reduce_column(matrix, base_index)
    return None


def lu_decomposition_essential_pivoting(matrix: np.ndarray):
    _, cols_count = matrix.shape
    pivots = np.identity(cols_count, dtype=np.int8)
    for base_index in range(cols_count - 1):
        if matrix[base_index, base_index] == 0:
            for i in range(base_index + 1, cols_count):
                if matrix[i, base_index] != 0:
                    swap_rows(matrix, pivots, i, base_index)
                    break
        reduce_column(matrix, base_index)
    return pivots


def lu_decomposition_partial_pivoting(matrix: np.ndarray):
    _, cols_count = matrix.shape
    pivots = np.identity(cols_count, dtype=np.int8)
    for base_index in range(cols_count - 1):
        max_index = np.argmax(matrix[base_index:, base_index])
        swap_rows(matrix, pivots, max_index + base_index, base_index)
        reduce_column(matrix, base_index)
    return pivots


pivoting_options = [
    lu_decomposition_without_pivoting,
    lu_decomposition_essential_pivoting,
    lu_decomposition_partial_pivoting,
]


def lu_decomposition(matrix: np.ndarray, pivoting: int = 2):
    rows_count, cols_count = matrix.shape
    if rows_count != cols_count:
        raise Exception('the matrix must be square')
    if pivoting >= len(pivoting_options):
        raise Exception('the pivoting is not valid')
    purposed_function = pivoting_options[pivoting]
    if not callable(purposed_function):
        raise Exception('there is no callable function for the desired pivoting')
    m = matrix.astype(np.float64)
    print(m)
    p = purposed_function(m)
    u = numpy.triu(m)
    print(u)
    l = numpy.tril(m)
    numpy.fill_diagonal(l, 1)
    print(l@u)
    return m, p


M = np.array([[2, 3, 4], [5, 2, 1], [4, 7, 2]])
print(lu_decomposition(M))
