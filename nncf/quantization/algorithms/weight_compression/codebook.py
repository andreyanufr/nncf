#from sklearn.cluster import KMeans
import numpy as np
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy
from nncf.tensor import functions as fns
from nncf.tensor import Tensor

@dataclass
class CodebookWeight:
    """
    Compressed weight and decompression parameters.

    :param tensor: The tensor with compressed weight.
    :param scale: The decompression scale, in practice it is dequantization scale for the INT quantization.
    :param zero_point: The zero-point, it is the value of the compression type corresponding to the value 0
        in the non-compression realm. Applicable for INT quantization.
    """

    tensor: Tensor
    scale: Tensor
    codebook: Tensor
    zero_point: Optional[Tensor] = None


def most_common(lst):
    """
    Return the most frequently occuring element in a list.
    """
    return max(set(lst), key=lst.count)


def euclidean_(point, data):
    """
    Return euclidean distances between a point & a dataset
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))

def euclidean(point, data):
    """
    Return euclidean distances between a point & a dataset
    """
    return (point - data)**2


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train, init, fixed=[]):

        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = deepcopy(init)

        # for _ in range(self.n_clusters-1):
        #     # Calculate distances from points to the centroids
        #     dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
        #     # Normalize the distances
        #     dists /= np.sum(dists)
        #     # Choose remaining points based on their distances
        #     new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)[0]  # Indexed @ zero to get val, not array of val
        #     self.centroids += [X_train[new_centroid_idx]]
        # for idx in fixed:
        #     self.centroids[idx] = init[idx]
        # This method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = self.centroids
        while  iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            # sorted_points = [[] for _ in range(self.n_clusters)]
            # for x in X_train:
            #     dists = euclidean(x, self.centroids)
            #     centroid_idx = np.argmin(dists)
            #     sorted_points[centroid_idx].append(x)

            # # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = deepcopy(self.centroids)
            # self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            
            dists = euclidean(X_train, self.centroids)
            centroid_idxs = np.argmin(dists, axis=1)
            for i in range(self.n_clusters):
                idxs = np.where(centroid_idxs == i)
                self.centroids[:, i] = np.mean(X_train[idxs, :])
            
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            for idx in fixed:
                self.centroids[:, idx] = init[:, idx]
            iteration += 1
            if np.all(np.abs(self.centroids - prev_centroids) < 0.0001).any():
                break

    def evaluate(self, X):
        dists = euclidean(X, self.centroids)
        centroid_idxs = np.argmin(dists, axis=1)

        return deepcopy(self.centroids).flatten(), centroid_idxs



class KMeansHist:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    @staticmethod
    def get_init(values, frequencies, n_clusters):
        step = 1.0 / n_clusters
        denum = np.sum(frequencies)
        quants = [i * step for i in range(n_clusters)]
        n_frequencies = frequencies / denum
        n_frequencies = np.cumsum(n_frequencies)

        res = []
        for i in range(len(quants)):
            if i == 0:
                res.append(values[0])
            elif i == len(quants) - 1:
                res.append(values[-1])
            else:
                prev = values[np.where(n_frequencies <= quants[i])[0][-1]]
                next_ = values[np.where(n_frequencies <= quants[i + 1])[0][-1]]
                res.append((prev + next_) / 2)
        
        res = np.array(res).reshape(1, -1)
        return res

    @staticmethod
    def create_histogramm(data, data_range=(-1.0, 1.0), granularity=0.01):
        centers = []
        step = granularity
        prev = data_range[0]
        
        while prev < data_range[1]:
            centers.append(prev + step / 2)
            prev += step
        
        centers = np.array(centers).reshape(1, -1)
        
        dists = euclidean(data, centers)
        centroid_idxs = np.argmin(dists, axis=1)
        
        res = [[], [], []]
        for i in range(centers.shape[1]):
            idxs = np.where(centroid_idxs == i)
            if len(idxs[0]) == 0:
                continue
            res[0].append(centers[:, i])
            res[1].append(np.sum(data[idxs, :]))
            res[2].append(len(idxs[0]))
        
        res[0] = np.array(res[0]).reshape(-1, 1)
        res[1] = np.array(res[1])
        res[2] = np.array(res[2])
        
        return res
        
    def fit(self, X_train, init, fixed=[]):
        if self.max_iter == 1:
            self.centroids = deepcopy(init)
            return

        self.hist = self.create_histogramm(X_train)
        
        init_by_hist = self.get_init(self.hist[0], self.hist[2], self.n_clusters)
        init_by_hist[0, 0] = -1.0
        init_by_hist[0, -1] = 1.0
        zero_idx = np.argmin(np.abs(init_by_hist[0, :]))
        init_by_hist[0, zero_idx] = init[0, zero_idx]
        fixed[1] = zero_idx
        init = init_by_hist
        
        self.centroids = deepcopy(init)

        iteration = 0
        prev_centroids = self.centroids
        while  iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            # sorted_points = [[] for _ in range(self.n_clusters)]
            # for x in X_train:
            #     dists = euclidean(x, self.centroids)
            #     centroid_idx = np.argmin(dists)
            #     sorted_points[centroid_idx].append(x)

            # # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = deepcopy(self.centroids)
            # self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            
            dists = euclidean(self.hist[0], self.centroids)
            centroid_idxs = np.argmin(dists, axis=1)
            for i in range(self.n_clusters):
                idxs = np.where(centroid_idxs == i)
                self.centroids[:, i] = np.sum(self.hist[1][idxs]) / np.sum(self.hist[2][idxs])
            
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            for idx in fixed:
                self.centroids[:, idx] = init[:, idx]
            iteration += 1
            if np.all(np.abs(self.centroids - prev_centroids) < 0.00001).any():
                break
        print(self.centroids)

    def evaluate(self, X):
        dists = euclidean(X, self.centroids)
        centroid_idxs = np.argmin(dists, axis=1)

        return deepcopy(self.centroids).flatten(), centroid_idxs


def weights_clusterization_k_means(weight, n_centroids=2**4, n_init="auto"):
    weight = weight.as_numpy_tensor().data
    scale = np.max(np.abs(weight), axis=-1, keepdims=True)
    weight = weight / scale
    orig_shape = weight.shape
    weight = weight.flatten()
    n_init[0] = weight.min()
    n_init[-1] = weight.max()
    
    kmeans = KMeansHist(n_centroids, max_iter=1)
    kmeans.fit(weight.reshape(-1, 1), n_init.reshape(1, -1), fixed=[0, 7, 15])
    
    codebook, indexes = kmeans.evaluate(weight.reshape(-1, 1))
    # codebook = kmeans.cluster_centers_.flatten()
    # indexes  = kmeans.labels_
    
    indexes = np.reshape(indexes, orig_shape)

    return CodebookWeight(Tensor(indexes), Tensor(scale), Tensor(codebook))
    
    
