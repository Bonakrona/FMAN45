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

for K in (2, 5):
    # labels, centroids = K_means_clustering(X_train, K)
    # centroid_labels = labeling(y_train, y_clusters=labels, k=K)
    # y_pred = np.array([K_means_classifier(x, centroids, centroid_labels) for x in X_test.T])
    # K_found = labels.max() + 1
    # markers = ['o', 'x', 's', 'D', 'v']
    # cmap = plt.get_cmap('tab10')
    
    # plt.figure(figsize=(10, 8))
    
    # for k in range(K_found):
    #     index = labels == k
    #     plt.scatter(Z2d[index, 0], Z2d[index, 1], c=cmap(k), marker=markers[k % len(markers)], label=f'Cluster {k}', alpha=0.6)
    # plt.xlabel('PC1', fontsize=14)
    # plt.ylabel('PC2', fontsize=14)
    # plt.title(f'K-means Clustering with K={K}', fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    # figure, axes = plt.subplots(1, K_found, figsize=(3.2*K_found, 3.5))
    # for k, ax in enumerate(axes):
    #     img = centroids[:, k].reshape(28, 28)
    #     ax.imshow(img, cmap='gray')
    #     ax.set_title(f'Centroid {k}', fontsize=14)
    #     ax.axis('off')
    # figure.suptitle(f'Centroids for K={K}', fontsize=16)
    # plt.show()
    
    # Exercise .2 continues here - comment below E10.1 and vice versa for E10.2
     
    print(f"\n RESULTS FOR K = {K}")
    train_clusters, centroids = K_means_clustering(X_train, K)
    centroid_labels = labeling(y_train, train_clusters, k=K)
    
    N_train = y_train.size
    train_errors = 0
    print("TRAINING DATA:")
    print(f"N_train = {N_train}")
    print("Cluster: #0, #1, assigned, misclassified:")
    for k in range(K):
        mask = (train_clusters == k)
        c0 = np.sum((y_train == 0) & mask)
        c1 = np.sum((y_train == 1) & mask)
        assigned = centroid_labels[k]
        mis_k = np.sum((y_train != assigned) & mask)
        train_errors += mis_k
        print(f"  {k+1}: {c0}, {c1}, {assigned}, {mis_k}")
        
    train_rate = train_errors / N_train * 100
    print(f"Sum misclassified: {train_errors}")
    print(f"Misclassification rate (%): {train_rate:.2f}")
    
    dist2_test     = np.sum((X_test[:, :, None] - centroids[:, None, :])**2, axis=0)
    test_clusters  = np.argmin(dist2_test, axis=1)
    
    N_test = y_test.size
    test_errors = 0
    print("\nTESTING DATA:")
    print(f"N_test = {N_test}")
    print("Cluster: #0, #1, assigned, misclassified:")
    for k in range(K):
        mask = (test_clusters == k)
        c0 = np.sum((y_test == 0) & mask)
        c1 = np.sum((y_test == 1) & mask)
        assigned = centroid_labels[k]
        mis_k = np.sum((y_test != assigned) & mask)
        test_errors += mis_k
        print(f"  {k+1}: {c0}, {c1}, {assigned}, {mis_k}")
    test_rate = test_errors / N_test * 100
    print(f"Sum misclassified: {test_errors}")
    print(f"Misclassification rate (%): {test_rate:.2f}")