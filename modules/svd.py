import numpy as np


def __generate_sigma_matrix(singular_values: np.ndarray):
    if singular_values.ndim != 1:
        raise Exception('singular values must be an one dimensional array')
    singular_values_count = np.shape(singular_values)[0]
    base_sigma_matrix = np.zeros((singular_values_count, singular_values_count), dtype=np.float64)
    np.fill_diagonal(base_sigma_matrix, singular_values)
    return base_sigma_matrix


def __generate_left_singular_matrix(matrix: np.ndarray, normal_vectors: np.ndarray, singular_values: np.ndarray):
    if singular_values.ndim != 1:
        raise Exception('singular values must be an one dimensional array')
    singular_values_count = np.shape(singular_values)[0]
    vector_rows, vector_cols = np.shape(normal_vectors)
    matrix_rows = np.shape(matrix)[0]
    if singular_values_count > vector_cols:
        raise Exception('singular values are more than vectors')
    result_matrix = np.zeros((matrix_rows, singular_values_count), dtype=np.float64)
    for i in range(singular_values_count):
        result_matrix[:, i] = 1 / singular_values[i] * (matrix @ normal_vectors[:, i])
    return result_matrix


def __generate_right_singular_matrix(normal_vectors: np.ndarray):
    return normal_vectors


def svd(matrix: np.ndarray):
    singular_matrix = matrix.transpose() @ matrix
    eigen_values, normal_eigen_vectors = np.linalg.eigh(singular_matrix)
    singular_values = np.sqrt(np.round(np.abs(eigen_values), 15))
    non_zero_singular_values = singular_values[singular_values != 0]
    right_singular_matrix = __generate_right_singular_matrix(normal_eigen_vectors)
    left_singular_matrix = __generate_left_singular_matrix(matrix, normal_eigen_vectors, non_zero_singular_values)
    singular_values_matrix = __generate_sigma_matrix(non_zero_singular_values)
    return left_singular_matrix, singular_values_matrix, right_singular_matrix
