import numpy as np


def column_reduction(matrix: np.ndarray, col_index: int, save_coff=True) -> None:
    rows_count, cols_count = matrix.shape
    if col_index < 0 or col_index >= cols_count:
        raise Exception(f"col_index must be between {0} and {cols_count - 1}")
    for i in range(col_index + 1, rows_count):
        change_coff = matrix[i, col_index] / matrix[col_index, col_index]
        if change_coff == 0:
            continue
        base_modified = change_coff * matrix[col_index, col_index:]
        matrix[i, col_index:] -= base_modified
        if save_coff:
            matrix[i, col_index] = change_coff


def swap_rows(matrix: np.ndarray, pivots: np.ndarray, first_row: int, second_row: int) -> None:
    rows_count, _ = matrix.shape
    if first_row < 0 or first_row >= rows_count:
        raise Exception(f"first_row must be between {0} and {rows_count - 1}")
    if second_row < 0 or second_row >= rows_count:
        raise Exception(f"second_row must be between {0} and {rows_count - 1}")
    matrix[[first_row, second_row]] = matrix[[second_row, first_row]]
    pivots[[first_row, second_row]] = pivots[[second_row, first_row]]


def find_nonzero_index(matrix: np.ndarray, col_index: int) -> int:
    rows_count, cols_count = matrix.shape
    for row_index in range(col_index + 1, cols_count):
        if matrix[row_index, col_index] != 0:
            return row_index
    return -1


def reverse_lower_triangular(matrix: np.ndarray) -> None:
    rows_count, cols_count = matrix.shape
    for row_index in range(rows_count):
        diagonal_element = matrix[row_index, row_index]
        if diagonal_element != 0:
            matrix[row_index, :row_index] *= -diagonal_element
            matrix[row_index, row_index] = 1 / diagonal_element


def get_lower_triangular(matrix: np.ndarray, diagonal_filler=1) -> np.ndarray:
    low = np.tril(matrix)
    if diagonal_filler is not None:
        np.fill_diagonal(low, diagonal_filler)
    return low


def get_upper_triangular(matrix: np.ndarray, diagonal_filler=None) -> np.ndarray:
    up = np.triu(matrix)
    if diagonal_filler is not None:
        np.fill_diagonal(up, diagonal_filler)
    return up
