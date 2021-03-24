import time
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
            tic = time.time()
            random.seed()
            cluster_centers_ = self._init_center(data)
            if self._display_log:
                print('Finish initialization, time cost until now:', time.time() - tic)
            labels_, interia_ = self._assign_label(data, cluster_centers_)
            if self._display_log:
                print('Finish assigning labels, time cost until now:', time.time() - tic)
            last_interia = interia_
            for iter_ in range(self._max_iter):
                n_iter_ = iter_ + 1
                cluster_centers_ = self._compute_centers(data, labels_)
                if self._display_log:
                    print('Finish computing centers, time cost until now:', time.time() - tic)
                labels_, interia_ = self._assign_label(data, cluster_centers_)
                if self._display_log:
                    print('Finish assigning labels, time cost until now:', time.time() - tic)
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
        if self._display_log:
            print('Finish sorting, time cost until now:', time.time() - tic)
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
        radius = np.zeros(self._n_cluster, dtype=np.float64)
        for i in range(self._n_cluster):
            radius[i] = self._get_dist_batch(cluster_centers[i], data[labels == i]).max(axis=0)
        index = np.argsort(radius, axis=0)
        res_cluster_centers = cluster_centers[index]
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
        cluster_centers = np.zeros((self._n_cluster, n_features), dtype = np.float64)
        if self._init == 'random':
            for i in range(self._n_cluster):
                center = self._random_select(data)
                while center in cluster_centers[0:i]:
                    center = self._random_select(data)
                cluster_centers[i, :] = center
        elif self._init == 'kmeans++':
            for i in range(self._n_cluster):
                cluster_centers[i, :] = self._kmeans_plusplus(data, cluster_centers[0:i])
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
        n_samples, n_features = data.shape
        interia_ = 0
        dist = KMeans._get_dist_batch(data.reshape(1, n_samples, n_features), cluster_centers.reshape(self._n_cluster, 1, n_features))
        labels_ = np.argmin(dist, axis=0)
        interia_ = np.sum(np.sqrt(np.min(dist, axis=0)))
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
        cluster_centers = np.zeros((self._n_cluster, n_features), dtype=np.float64)
        for i in range(self._n_cluster):
            cluster_members = data[labels == i]
            if len(cluster_members) != 0:
                cluster_centers[i] = cluster_members.mean(axis=0)
            else:
                if self._init == 'kmeans++':
                    cluster_centers[i, :] = self._kmeans_plusplus(data, cluster_centers[0:i])
                else:
                    cluster_centers[i, :] = self._random_select(data)
        return cluster_centers

    @staticmethod
    def _get_dist(x, y, squared=False):
        if squared:
            return np.sqrt((x - y).dot(x - y))
        else:
            return (x - y).dot(x - y)
    
    @staticmethod
    def _get_dist_batch(x, y, squared=False):
        if squared:
            return np.sqrt(((x - y) ** 2).sum(axis=-1))
        else:
            return ((x - y) ** 2).sum(axis=-1)
    
    @staticmethod
    def _random_select(data):
        n_samples, _ = data.shape
        id = random.randint(0, n_samples - 1)
        return data[id]
    
    @staticmethod
    def _kmeans_plusplus(data, centers):
        if centers.shape[0] == 0:
            return KMeans._random_select(data)
        else: 
            n_samples, n_features = data.shape
            n_cluster, _ = centers.shape
            max_pts = np.argmax(np.min(KMeans._get_dist_batch(data.reshape(1, n_samples, n_features), centers.reshape(n_cluster, 1, n_features)), axis=0))
            return data[max_pts]

