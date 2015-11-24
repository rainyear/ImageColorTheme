from sklearn.cluster import KMeans as KM
import numpy as np

class KMDiy(object):
    """KMDiy = KMeans DIY"""
    MAX_ITER = 300
    def __init__(self, n_clusters=8, max_iter=300):
        super(KMDiy, self).__init__()
        self.n_clusters       = n_clusters
        self.cluster_centers_ = None
        self.MAX_ITER         = max_iter

    def randCent(self, data):
        dim = data.shape[1]
        centroids = np.zeros((self.n_clusters, dim))
        for j in range(dim):
            minJ = min(data[:, j])
            maxJ = max(data[:, j])
            centroids[:, j] = minJ + float(maxJ - minJ) * np.random.rand(self.n_clusters)
        return centroids
    def fit(self, data):
        self.cluster_centers_ = self.randCent(data)
        size, dim = data.shape
        clusterAssment = np.zeros((size, 2))
        clusterChanged = True
        iters = 0
        while clusterChanged:
            clusterChanged = False
            if iters > self.MAX_ITER:
                print("Reach MAX_ITER #{0}".format(self.MAX_ITER))
                break

            for i in range(size):
                minDist = np.inf
                centIdx = -1
                for j in range(dim):
                    d = self.distMeas(self.cluster_centers_[j, :], data[i, :])
                    if d < minDist:
                        minDist = d
                        centIdx = j
                if clusterAssment[i, 0] != centIdx: clusterChanged = True
                clusterAssment[i, :] = centIdx, minDist
            for j in range(dim):
                ptsInCluster = data[np.nonzero(clusterAssment[:,0] == j)]
                self.cluster_centers_[j, :] = np.mean(ptsInCluster)#new cluster centroid
            iters += 1

    def distMeas(self, v1, v2):
        return np.sqrt(sum([pow(x-y, 2) for x in v1 for y in v2]))

class KMeans(object):
    """docstring for KMeans"""
    def __init__(self, pixData, maxColor, useSklearn=True):
        super(KMeans, self).__init__()
        h, w, d = pixData.shape
        self.pixData = np.reshape(pixData, (h * w, d))
        self.maxColor = maxColor
        if useSklearn:
            self._KMeans = KM(n_clusters = maxColor)
        else:
            self._KMeans = KMDiy(n_clusters = maxColor)

    def quantize(self):
        self._KMeans.fit(self.pixData)
        return np.array(self._KMeans.cluster_centers_, dtype=np.uint8)
