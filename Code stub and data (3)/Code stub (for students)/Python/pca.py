import numpy as np
import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat("A2_data.mat")
X = mat["train_data_01"]
y = mat["train_labels_01"].flatten()

my = X.mean(axis=1, keepdims=True)
Xc = X - my

U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
PCs = U[:, :2]
Z = PCs.T @ Xc

plt.figure(figsize=(8, 6))
mask0 = y == 0
mask1 = y == 1

plt.scatter(Z[0, mask0], Z[1, mask0], c = 'red', marker = "o", label = 'Class 0') # testa alpha=0.6?
plt.scatter(Z[0, mask1], Z[1, mask1], c = 'blue', marker = "x", label = 'Class 1') # testa alpha= 0.6?

plt.xlabel('PC1', fontsize = 14)
plt.ylabel('PC2', fontsize = 14)
plt.title('PCA of MNIST Data', fontsize = 16)
plt.legend()
plt.grid()
plt.show()

print("Mean of projected data:", Z.mean(axis=1))