import numpy as np


def swap_rows(matrix: np.ndarray, pivots: np.ndarray, first_row: int, second_row: int):
    rows_count, _ = matrix.shape
    if first_row < 0 or first_row >= rows_count:
        raise Exception(f"first_row must be between {0} and {rows_count - 1}")
    if second_row < 0 or second_row >= rows_count:
        raise Exception(f"second_row must be between {0} and {rows_count - 1}")
    matrix[[first_row, second_row]] = matrix[[second_row, first_row]]
    pivots[[first_row, second_row]] = pivots[[second_row, first_row]]
