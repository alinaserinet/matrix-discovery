import numpy as np


def reduce_column(matrix: np.ndarray, col_index: int) -> None:
    rows_count, cols_count = matrix.shape
    if col_index < 0 or col_index >= cols_count:
        raise Exception(f"col_index must be between {0} and {cols_count - 1}")
    for i in range(col_index + 1, rows_count):
        change_coff = matrix[i, col_index] / matrix[col_index, col_index]
        if change_coff == 0:
            continue
        base_modified = change_coff * matrix[col_index, col_index:]
        matrix[i, col_index:] -= base_modified
        matrix[i, col_index] = change_coff
