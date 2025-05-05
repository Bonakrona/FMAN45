import numpy as np
import scipy.io
import matplotlib.pyplot as plt 
from K_means_clustering import K_means_clustering, labeling, K_means_classifier

mat = scipy.io.loadmat("A2_data.mat")
X_train = mat["train_data_01"]
y_train = mat["train_labels_01"].flatten()

X_test = mat["test_data_01"]
y_test = mat["test_labels_01"].flatten()

my = X_train.mean(axis=1, keepdims=True)

Xc = X_train - my
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
Z2d = (U[:, :2].T @ Xc).T

# exercise 10.2 starts here
y_clusters, centroids_cls = K_means_clustering(X_train, 2)
centroid_labels = labeling(y_train=mat["train_labels_01"].flatten(), y_clusters=y_clusters, k=2)

def classify_matrix(Xmat, C, c_labels):
    return np.array([K_means_classifier(x, C, c_labels) for x in Xmat.T])

y_pred_train = classify_matrix(X_train, centroids_cls, centroid_labels)
y_pred_test  = classify_matrix(mat["test_data_01"], centroids_cls, centroid_labels)

train_errors = np.sum(y_pred_train != y_train)
test_errors  = np.sum(y_pred_test != y_test)

train_rate   = train_errors / y_train.size * 100
test_rate    = test_errors / y_test.size * 100

# print(f"TRAIN misclassified: {train_errors} / {y_train.size}  ({train_rate:.2f}% )")
# print(f"TEST  misclassified: {test_errors} / {y_test.size}   ({test_rate:.2f}% )")

# exercise 10.2 continues below


for K in (2, 5):
    labels, centroids = K_means_clustering(X_train, K)
    
    centroid_labels = labeling(y_train=mat["train_labels_01"].flatten(), y_clusters=labels, k=K)
    y_pred = np.array([K_means_classifier(x, centroids, centroid_labels) for x in X_test.T])
    K_found = labels.max() + 1
    markers = ['o', 'x', 's', 'D', 'v']
    cmap = plt.get_cmap('tab10')
    
    plt.figure(figsize=(10, 8))
    
    for k in range(K_found):
        index = labels == k
        plt.scatter(Z2d[index, 0], Z2d[index, 1], c=cmap(k), marker=markers[k % len(markers)], label=f'Cluster {k}', alpha=0.6)
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.title(f'K-means Clustering with K={K}', fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()
    
    figure, axes = plt.subplots(1, K_found, figsize=(3.2*K_found, 3.5))
    for k, ax in enumerate(axes):
        img = centroids[:, k].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Centroid {k}', fontsize=14)
        ax.axis('off')
    figure.suptitle(f'Centroids for K={K}', fontsize=16)
    plt.show()
    
# Exercise 10.2 continues here

print("\nTRAINING DATA:")
print("Cluster |  #0  |  #1  | Assigned | Misclassified")
for k in range(len(centroid_labels)):
    mask      = (labels == k)
    c0        = np.sum((y_train == 0) & mask)
    c1        = np.sum((y_train == 1) & mask)
    assigned  = centroid_labels[k]
    mis_k     = np.sum((y_train != assigned) & mask)
    print(f"{k+1:>3}      {c0:>5} {c1:>5}     {assigned:>1}         {mis_k:>5}")

print(f"\nN_train = {y_train.size}")
print(f"Sum misclassified: {train_errors}")
print(f"Misclassification rate (%): {train_rate:.2f}")

print("\nTESTING DATA:")
y_clust_test, _ = K_means_clustering(X_test, len(centroid_labels))  # Cluster the test data
print("Cluster |  #0  |  #1  | Assigned | Misclassified")
for k in range(len(centroid_labels)):
    mask      = (y_clust_test == k)
    c0        = np.sum((y_test == 0) & mask)
    c1        = np.sum((y_test == 1) & mask)
    assigned  = centroid_labels[k]
    mis_k     = np.sum((y_test != assigned) & mask)
    print(f"{k+1:>3}      {c0:>5} {c1:>5}     {assigned:>1}         {mis_k:>5}")

print(f"\nN_test = {y_test.size}")
print(f"Sum misclassified: {test_errors}")
print(f"Misclassification rate (%): {test_rate:.2f}")