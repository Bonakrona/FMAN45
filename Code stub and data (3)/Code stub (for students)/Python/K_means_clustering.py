import numpy as np
import scipy
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def K_means_clustering(X, K):
    """
    Perform K-means clustering on input data.

    Parameters:
    - X: numpy.ndarray
        DxN matrix of input data.
    - K: int
        Number of clusters.

    Returns:
    - y: numpy.ndarray
        Nx1 vector of cluster assignments.
    - C: numpy.ndarray
        DxK matrix of cluster centroids.
    """

    D, N = X.shape

    intermax = 50
    conv_tol = 1e-6

    # Initialize
    C = np.mean(X, axis=1).reshape(D, 1) + np.std(X, axis=1).reshape(D, 1) * np.random.randn(D, K)
    y = np.zeros(N)
    Cold = C.copy()

    for i in range(intermax):
        # Step 1: Assign to clusters
        y = step_assign_cluster(X, Cold)

        # Step 2: Assign new clusters
        C, delta = step_compute_mean(X, y, Cold)
        if delta < conv_tol:
            return y, C

        Cold = C.copy()

    return y, C

def plot_pca(X, y):
    my = X.mean(axis=1, keepdims=True)
    Xc = X - my
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    PCs = U[:, :2]
    Z = PCs.T @ Xc

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(y)
    markers = ['o', 'x', 's', 'D', '*']
    cmap = plt.get_cmap('tab10')

    for i, k in enumerate(unique_labels):
        mask = y == k
        plt.scatter(Z[0, mask], Z[1, mask], label=f'Cluster {k}', marker=markers[i % len(markers)], color=cmap(i), alpha=0.6)

    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.title('PCA of Clusters', fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()

def fxdist(x,C):
    # CHANGE
    d = C - x.reshape(-1, 1)
    d = np.sum(d**2, axis=0)
    # DO NOT CHANGE
    return d

def fcdist(C1,C2):
    # CHANGE
    # d = C1 - C2
    # d = np.sum(d**2, axis=0)
    d = np.linalg.norm(C1 - C2)
    # DO NOT CHANGE
    return d

def step_assign_cluster(X, C):
    # d = X[:, :, None] - C[:, None, :]
    # dist = np.sum(d**2, axis=0)
    # dist = fxdist(X, C)
    # return np.argmin(dist, axis=1)
    return np.array([np.argmin(fxdist(x_i, C)) for x_i in X.T])

def step_compute_mean(X, y, cold):
    k = cold.shape[1]
    cnew = np.zeros_like(cold)
    for k in range(k):
        members = X[:, y == k]
        if members.size == 0:
            cnew[:, k] = cold[:, k]
        else:
            cnew[:, k] = members.mean(axis=1)
            
    move = np.linalg.norm(cnew - cold)
    return cnew, move

def labeling(y_train, y_clusters, k=2):
    c_labels = np.zeros(k, dtype = int)
    for i in range(k):
        members = y_train[y_clusters == i]
        if members.mean() >= 0.5:
            c_labels[i] = 1
    return c_labels

def K_means_classifier(x, C, c_labels):
    dist = np.sum((C-x[:, None])**2, axis=0)
    k_near = np.argmin(dist)
    return c_labels[k_near]

def print_K_means_classifier_result(X_train, y_train, X_test, y_test, K):
    y_clusters, centroids = K_means_clustering(X_train, K)
    centroid_labels = labeling(y_train, y_clusters, k=K)

    # y_pred_train = np.array([K_means_classifier(x, centroids, centroid_labels) for x in X_train.T])
    # y_pred_test = np.array([K_means_classifier(x, centroids, centroid_labels) for x in X_test.T])

    print("TRAINING DATA:")
    print("Cluster |  #0  |  #1  | Assigned | Misclassified")
    total_misclassified_train = 0
    for k in range(K):
        cluster_mask = y_clusters == k
        c0 = np.sum((y_train == 0) & cluster_mask)
        c1 = np.sum((y_train == 1) & cluster_mask)
        assigned = centroid_labels[k]
        misclassified = np.sum(y_train[cluster_mask] != assigned)
        total_misclassified_train += misclassified
        print(f"{k+1:>3}      {c0:>5} {c1:>5}     {assigned:>1}         {misclassified:>5}")
    print(f"\nN_train = {y_train.size}")
    print(f"Sum misclassified: {total_misclassified_train}")
    print(f"Misclassification rate (%): {total_misclassified_train / y_train.size * 100:.2f}")

    print("\nTESTING DATA:")
    print("Cluster |  #0  |  #1  | Assigned | Misclassified")
    y_clusters_test = np.array([np.argmin(np.sum((centroids - x[:, None])**2, axis=0)) for x in X_test.T])
    total_misclassified_test = 0
    for k in range(K):
        cluster_mask = y_clusters_test == k
        c0 = np.sum((y_test == 0) & cluster_mask)
        c1 = np.sum((y_test == 1) & cluster_mask)
        assigned = centroid_labels[k]
        misclassified = np.sum(y_test[cluster_mask] != assigned)
        total_misclassified_test += misclassified
        print(f"{k+1:>3}      {c0:>5} {c1:>5}     {assigned:>1}         {misclassified:>5}")
    print(f"\nN_test = {y_test.size}")
    print(f"Sum misclassified: {total_misclassified_test}")
    print(f"Misclassification rate (%): {total_misclassified_test / y_test.size * 100:.2f}")

def svm_linear(X_train, y_train, X_test, C=1.0):
    linear_svm = SVC(kernel='linear', C=C)
    linear_svm.fit(X_train, y_train) # linearsvm.fit(X_train.T, y_train) ? 
    y_pred_train = linear_svm.predict(X_train) # .T ?
    y_pred_test  = linear_svm.predict(X_test) # .T ?
    return y_pred_train, y_pred_test

def svm_gausian(X_train, y_train, X_test, gamma):
    gaus_svm = SVC(kernel='rbf', gamma=gamma)
    gaus_svm.fit(X_train, y_train)
    y_pred_train = gaus_svm.predict(X_train) #.T ?
    y_pred_test  = gaus_svm.predict(X_test) # .T ?
    return y_pred_train, y_pred_test

def svm_data_print(y_true, y_pred):
    n00 = np.sum((y_true==0) & (y_pred==0))
    n01 = np.sum((y_true==1) & (y_pred==0))
    n10 = np.sum((y_true==0) & (y_pred==1))
    n11 = np.sum((y_true==1) & (y_pred==1))
    
    N = y_true.size
    sum = n01 + n10
    rate = sum / N * 100
    print(f"N = {N}")
    print(f"'0' {n00:6d} {n01:6d}")
    print(f"'1' {n10:6d} {n11:6d}")
    print(f"Sum misclassified = {sum}")
    print(f"Misclassification rate (%) = {rate:.2f}%\n")
    return n00, n01, n10, n11


def load_data():
    # Replace '/path/to/file/' with the path to your .mat file
    base_path = "C:/Users/User/OneDrive/Documents/Lund ar 5/vt/FMAN45/Assignments/1/FMAn45/Code stub and data (3)/Code stub (for students)/Python/A2_data.mat"
    mat_file_path = base_path
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"Error: File '{mat_file_path}' not found.")
        mat_data = None

    if mat_data is not None:
        # Access variables from the .mat file
        test_data = mat_data['test_data_01']
        test_labels = mat_data['test_labels_01']
        train_data = mat_data['train_data_01']
        train_labels = mat_data['train_labels_01']
        return [test_data, test_labels, train_data, train_labels]


if __name__ == "__main__":
    data = load_data()
    
    # PCA (7) -------------------------------------------------------
    # plot_pca(data[2], data[3].ravel())
    
    # nbr_clusters = 2 # Replace with you chosen int
    
    for K in (2, 5):
        # y, C = K_means_clustering(data[2], K)
        # print(f"Plotting K-means clustering with K = {K}")
        # plot_pca(data[2], y)
        
        # # Display centroids (9)
        # fig, axes = plt.subplots(1, K, figsize=(2.8 * K, 3))
        # for k in range(K):
        #     if K > 1:
        #         ax = axes[k] 
        #     else:
        #         ax = axes
        #     ax.imshow(C[:, k].reshape(28, 28), cmap='gray')
        #     ax.set_title(f'Centroid {k}', fontsize=10)
        #     ax.axis('off')
        # fig.suptitle(f'Centroids for K = {K}', fontsize=14)
        # plt.tight_layout()
        # plt.show()
        
        print_K_means_classifier_result(data[2], data[3].ravel(), data[0], data[1].ravel(), K)
        
        # Print all data result (10.2)
    
    #SVM linear (12) -------------------------------------------------------
    # print("SVM linear")
    # y_pred_train, y_pred_test = svm_linear(data[2].T, data[3].ravel(), data[0].T, C=1.0)
    # print("Train data")
    # svm_data_print(data[3].ravel(), y_pred_train)
    # print("Test data")
    # svm_data_print(data[1].ravel(),  y_pred_test)
    
    # SVM gaussian (13) -------------------------------------------------------
    # print("SVM gaussian")
    # y_pred_train1, y_pred_test1 = svm_gausian(data[2].T, data[3].ravel(), data[0].T, gamma=0.1)
    # y_pred_train2, y_pred_test2 = svm_gausian(data[2].T, data[3].ravel(), data[0].T, gamma=0.01)
    # y_pred_train3, y_pred_test3 = svm_gausian(data[2].T, data[3].ravel(), data[0].T, gamma=1)       

    # print("Train data 1 gamma 0.1")
    # svm_data_print(data[3].ravel(), y_pred_train1)
    # print("Test data 1 gamma 0.1")
    # svm_data_print(data[1].ravel(), y_pred_test1)

    # print("Train data 2 gamma 0.01")
    # svm_data_print(data[3].ravel(), y_pred_train2)
    # print("Test data 2 gamma 0.01")
    # svm_data_print(data[1].ravel(), y_pred_test2)
    
    # print("Train data 3 gamma 1")
    # svm_data_print(data[3].ravel(), y_pred_train3)
    # print("Test data 3 gamma 1")
    # svm_data_print(data[1].ravel(), y_pred_test3)