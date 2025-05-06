import numpy as np
import scipy

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


def load_data():
    # Replace '/path/to/file/' with the path to your .mat file
    base_path = "/path/to/file/"
    mat_file_path = base_path + "A2_data.mat"
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
    
def step_assign_cluster(X, C):
    d = X[:, :, None] - C[:, None, :]
    dist = np.sum(d**2, axis=0)
    return np.argmin(dist, axis=1)

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

if __name__ == "__main__":
    data = load_data()
    nbr_clusters = 4 # Replace with you chosen int
    y, C = K_means_clustering(data[2], nbr_clusters)
