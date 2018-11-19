import numpy as np
import scipy, scipy.spatial

def my_kmeans(data, k):
    # TODO:
    # The variable 'data' contains data points in its rows.
    # Initilize 'k' centers randomized.  Afterwards apply Floyd's algorithm
    # until convergence to get a solution of the k-means problem.  Utilize
    # 'scipy.spatial.cKDTree' for nearest neighbor computations.
    num_samples, dim  = data.shape
    #print(num_samples, dim)
    centers = data[np.random.choice(range(num_samples), size=k)] # each center (dimension: k x 3)
    old_distances, old_labels = scipy.spatial.cKDTree(centers).query(data)
    
    while True:
        for c in range(k):
            c_idx = np.argwhere(old_labels == c)
            centers[c] = np.mean(data[c_idx], axis=0)
        
        new_distances, new_labels = scipy.spatial.cKDTree(centers).query(data)

        if np.all(new_labels == old_labels):
            break
        
        old_labels = new_labels
                
    return old_labels, centers

