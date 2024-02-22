import numpy as np
from modules import lu_decomposition
from utils.echelon_tools import get_lower_triangular, get_upper_triangular

matrix = np.array([[2, 3, 4], [5, 2, 1], [4, 7, 2]])
decompose_matrix, pivots = lu_decomposition(matrix)
low = get_lower_triangular(decompose_matrix)
up = get_upper_triangular(decompose_matrix)
print(low)
print("----------")
print(up)
print("----------")
print(pivots)
print("----------")
print(low @ up)
