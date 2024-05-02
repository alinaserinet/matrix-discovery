import numpy as np


class __SVD:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    # SVD using A.T @ A matrix
    def ata(self):
        symmetric_matrix = self.matrix.T @ self.matrix

        # Calculate eigen values and eigen vectors of symmetric matrix
        eigen_values, eigen_vectors = np.linalg.eigh(symmetric_matrix)

        # Calculate singular value form symmetric eigen values
        singular_values = np.sqrt(np.round(eigen_values, decimals=14))

        # Find descending sorted indexes of singular values
        singular_values_dsc_index = np.argsort(singular_values)[::-1]

        # Descending sorted of singular values
        sorted_dsc_singular_values = singular_values[singular_values_dsc_index]

        # Separate non zero singular values
        non_zeros_singular_values = sorted_dsc_singular_values[np.nonzero(sorted_dsc_singular_values)]

        # Calculate right singular vectors (V)
        right_singular_matrix = np.round(eigen_vectors, decimals=14)[:, singular_values_dsc_index]

        # Calculate left singular vectors (U)
        non_zeros_singular_values_count = np.shape(non_zeros_singular_values)[0]
        left_singular_shape = (non_zeros_singular_values_count, non_zeros_singular_values_count)
        left_singular_matrix = np.zeros(left_singular_shape, dtype=np.float64)
        for i in range(non_zeros_singular_values_count):
            left_singular_matrix[:, i] = 1 / non_zeros_singular_values[i] * self.matrix @ right_singular_matrix[:, i]

        # Calculate sigma matrix
        sigma = np.zeros((left_singular_matrix.shape[1], right_singular_matrix.shape[0]))
        np.fill_diagonal(sigma, non_zeros_singular_values)

        return left_singular_matrix, sigma, right_singular_matrix.T

    # SVD using A @ A.T matrix
    def aat(self):
        return self.matrix


def svd(matrix: np.ndarray):
    __svd = __SVD(matrix)
    return __svd.ata()
