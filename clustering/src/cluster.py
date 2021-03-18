import random
import numpy as np

class KMeans(object):
    """KMeans clustering algorithm.

    Parameters
    ----------
    n_cluster: int, default: 5,
        the number of clusters, i.e., K in "KMeans".
    init: string, {'kmeans++', 'random'}
        the initialize method of the first k centers of clusters
        'random' uses random sampling, and 'kmeans++' picks dispersed set of instances.
    n_init: int, default: 10,
        the number of time the KMeans algorithms will be run with different random seeds.
    max_iter: int, default: 300,
        the maximum of iterations
    tol: float, default: 1e-4,
        the relative tolerance with regards to the difference in the cluster centers of 
        two consecutive iterations to declare convergence.
    sorted_cluster: bool, default: True,
        whether sort the clusters according to their radius in the end.
    display_log: bool, default: False,
        whether display the log on the screen.
    """
    def __init__(self, n_cluster=5, init='kmeans++', n_init=10, max_iter=300, tol=1e-4, sorted_cluster=True, display_log=False):
        if n_cluster <= 0:
            raise AttributeError('The number of clusters should be positive.')
        self._n_cluster = n_cluster
        if init not in ['kmeans++', 'random']:
            raise AttributeError('The initialization method is not available.')
        else:
            self._init = init
        self._n_init = n_init
        self._max_iter = max_iter
        self._tol = tol
        self._sorted_cluster = sorted_cluster
        self._display_log = display_log
    
    def fit(self, data):
        """Compute k-means clustering.

        Parameters
        ----------
        data: numpy.array of shape (n_samples, n_features), 
              which is the training instances to cluster.

        Returns
        -------
        self: fitted estimator.
        """
        self.interia_ = np.inf
        for epoch in range(self._n_init):
            random.seed()
            cluster_centers_ = self._init_center(data)
            labels_, interia_ = self._assign_label(data, cluster_centers_)
            last_interia = interia_
            for iter_ in range(self._max_iter):
                n_iter_ = iter_ + 1
                cluster_centers_ = self._compute_centers(data, labels_)
                labels_, interia_ = self._assign_label(data, cluster_centers_)
                if abs(interia_ - last_interia) < self._tol:
                    break
                last_interia = interia_
                if self._display_log:
                    print('Iter {}, interia = {}'.format(n_iter_, interia_))
            if self._display_log:
                print('Epoch {}, final interia = {}'.format(epoch, interia_))
            if interia_ < self.interia_:
                self.n_iter_ = n_iter_
                self.cluster_centers_ = cluster_centers_
                self.labels_ = labels_
                self.interia_ = interia_
        if self._sorted_cluster:
            self.cluster_centers_, self.labels_ = self._sort_cluster(data, self.cluster_centers_, self.labels_)
        return self
    
    def _sort_cluster(self, data, cluster_centers, labels):
        """Sort the cluster by their radius in ascending order.

        Parameters
        ----------
        data: the data used to fit the algorithm.
        cluster_centers: the centers of clusters.
        labels: the label of each instance.

        Returns
        -------
        res_cluster_centers: the new cluster centers after sorting
        res_labels: the new labels after sorting
        """
        _, n_features = data.shape
        radius = np.zeros(self._n_cluster)
        for i, pts in enumerate(data):
            k = labels[i]
            radius[k] = max(radius[k].astype(np.float64), self._get_dist(cluster_centers[k], pts))
        zipped_cluster_centers = np.concatenate([cluster_centers, radius.reshape((self._n_cluster, 1))], axis=1)
        res_cluster_centers = np.array(sorted(zipped_cluster_centers, key=lambda c: c[-1]))[:, 0:n_features]
        res_labels, _ = self._assign_label(data, res_cluster_centers)
        return res_cluster_centers, res_labels
        

    def _init_center(self, data):
        """Initialize the center of the cluster.

        Parameters
        ----------
        data: the data used to fit the algorithm.

        Returns
        -------
        cluster_centers: initialized centers of the clusters.
        """
        _, n_features = data.shape
        cluster_centers = np.zeros((self._n_cluster, n_features))
        if self._init == 'random':
            for i in range(self._n_cluster):
                center = self._random_select(data)
                while center in cluster_centers[0:i]:
                    center = self._random_select(data)
                cluster_centers[i, :] = center
        elif self._init == 'kmeans++':
            for i in range(self._n_cluster):
                cluster_centers[i, :] = self._farthest_select(data, cluster_centers[0:i])
        else:
            raise AttributeError('The initialization method is not available.')
        return cluster_centers

    def _assign_label(self, data, cluster_centers):        
        """Assign label to each instance

        Parameters
        ----------
        data: the data used to fit the algorithm.
        cluster_centers: the centers of clusters.

        Returns
        -------
        labels_: the label to each instance
        interia_: the global interia
        """
        n_samples, _ = data.shape
        labels_ = [0] * n_samples
        interia_ = 0
        for i, pts in enumerate(data):
            min_dist = np.inf
            belong_id = -1
            for k in range(self._n_cluster):
                dist = self._get_dist(pts, cluster_centers[k])
                if dist < min_dist:
                    min_dist = dist
                    belong_id = k
            assert belong_id != -1
            labels_[i] = belong_id
            interia_ += min_dist
        return labels_, interia_

    def _compute_centers(self, data, labels):      
        """Compute the centers of each cluster

        Parameters
        ----------
        data: the data used to fit the algorithm.
        labels: the label to each instance

        Returns
        -------
        cluster_centers: new centers of clusters according to the labels.
        """
        _, n_features = data.shape
        cluster_centers = np.zeros((self._n_cluster, n_features))
        cluster_counts = np.zeros(self._n_cluster)
        for i, pts in enumerate(data):
            cluster_centers[labels[i]] += pts
            cluster_counts[labels[i]] += 1
        for i in range(self._n_cluster):
            if cluster_counts[i] != 0:
                cluster_centers[i] /= cluster_counts[i]
            else:
                if self._init == 'kmeans++':
                    cluster_centers[i, :] = self._farthest_select(data, cluster_centers[0:i])
                else:
                    cluster_centers[i, :] = self._random_select(data)
        return cluster_centers

    @staticmethod
    def _get_dist(x, y):
        return np.sum((x - y) ** 2)
    
    @staticmethod
    def _random_select(data):
        n_samples, _ = data.shape
        id = random.randint(0, n_samples)
        return data[id]
    
    @staticmethod
    def _farthest_select(data, centers):
        if centers.shape[0] == 0:
            return KMeans._random_select(data)
        else:
            max_dist = -np.inf
            max_pts = np.array([])
            for pts in data:
                min_dist = np.inf
                for center in centers:
                    dist = KMeans._get_dist(center, pts)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist > max_dist:
                    max_dist = min_dist
                    max_pts = pts
            assert max_pts != np.array([])
            return max_pts

