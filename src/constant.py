import numpy as np

target_indices = np.array([i-1 for i in [2, 6, 10, 12, 13, 15, 18, 19, 21, 22, 25, 26, 36, 37, 39, 48]]).astype(int)

logs = [np.log2(i+2) for i in range(49)]
# logs = [np.log2(2), np.log2(3), np.log2(4)]
