import numpy as np
from modules import svd

matrix = np.array([[3, 0], [4, 5]])
print(svd(matrix))
print(np.linalg.svd(matrix))

