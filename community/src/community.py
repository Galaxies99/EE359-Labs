import random
from tqdm import tqdm
from graphx import WeightedUndirectedGraph, GraphPartition


class Louvain(object):
    '''
    Use Louvain Algorithm to detect communities in a given graph.

    References
    ----------
    1. Blondel, Vincent D., et al. "Fast unfolding of communities in large networks." Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.

    Members
    -------
    Louvain.num_clusters: the desired number of communities;
    Louvain.epsilon: the epsilon value that indicates convergence.
    '''
    def __init__(self, num_clusters, epsilon = 1e-6):
        '''
        Initialization.

        Parameters
        ----------
        num_clusters: the desired number of communities;
        epsilon: the epsilon value that indicates convergence.
        '''
        super(Louvain, self).__init__()
        self.num_clusters = num_clusters
        self.epsilon = epsilon
        
    
    def fit(self, G):
        '''
        Get the best partition of the graph using Louvain algorithm.

        Parameters
        ----------
        G: a WeightedUndirectedGraph object, the given graph.

        Returns
        -------
        A partition of the graph G.
        '''
        graph = G.copy()
        partition = GraphPartition(graph)

        partition, last_Q = self._partition(partition, graph)
        partition_list = [partition.copy()]

        while True:
            print('Modularity: ', last_Q)
            partition, graph = self._restructure(partition, graph)
            print('Modularity after restructure: ', partition.modularity())
            partition, Q = self._partition(partition, graph)
            partition_list.append(partition.copy())

            if abs(Q - last_Q) < self.epsilon:
                break

            last_Q = Q
        
        return self._combine_partition_list(partition_list, G)

    
    def _partition(self, partition, graph):
        '''
        The first phase of Louvain algorithm: optimize partition.

        Parameters
        ----------
        partition: a GraphPartition object, the initial partition;
        graph: a WeightedUndirectedGraph object, the given graph.

        Returns
        -------
        The local-optimal partition.
        '''
        Q = partition.modularity()
        no_gain = False
        while not no_gain:
            print(Q)
            no_gain = True
            # Randomize accessing nodes in G
            for x in tqdm(random.sample(graph.iter_nodes().keys(), len(graph.iter_nodes().keys()))):
                x_community = partition.get_community(x)
                max_Q = -1
                best_partition = partition.copy()
                for y in graph.iter_edges(x).keys():
                    y_community = partition.get_community(y)
                    if x_community == y_community:
                        continue
                    partition_t = partition.copy()
                    partition_t.assign_community(x, y_community)
                    # TODO: optimize for delta Q calculation
                    Q_t = partition_t.modularity()   
                    if Q_t > max_Q:
                        max_Q = Q_t
                        best_partition = partition_t.copy()
                if max_Q > Q:
                    Q = max_Q
                    partition = best_partition.copy()
                    no_gain = False
        return partition, Q

    
    def _restructure(self, partition, graph):
        '''
        Aggregate the community and restructure the graph.

        Parameters
        ----------
        partition: a GraphPartition object, the partition;
        graph: a WeightedUndirectedGraph object, the original graph.

        Returns
        -------
        new_partition: a GraphPartition object, the new partition corresponding to the new graph;
        new_graph: a WeightedUndirectedGraph object, the restructured graph.
        '''
        new_graph = WeightedUndirectedGraph()
        for x in graph.iter_nodes():
            x_community = partition.get_community(x)
            for y, w in graph.iter_edges(x).items():
                y_community = partition.get_community(y)
                if x == y:
                    new_graph.add_edge(x_community, y_community, w / 2)
                else:
                    if x_community != y_community:
                        new_graph.add_edge(x_community, y_community, w / 2)
                    else:
                        new_graph.add_edge(x_community, y_community, w)
        new_partition = GraphPartition(new_graph)
        return new_partition, new_graph

    
    def _combine_partition_list(self, partition_list, graph):
        '''
        Combine the partitions at all levels

        Parameters
        ----------
        partition_list: a list of GraphPartition object, the partition at all levels;
        graph: a WeightedUndirectedGraph object, the original graph.

        Returns
        -------
        The final combined partition of the graph.
        '''
        partition = partition_list[0].copy()
        for index in range(1, len(partition_list)):
            for x in graph.iter_nodes():
                cur_cluster = partition.get_community(x)
                partition.assign_community(x, partition_list[index].get_community(cur_cluster))
        return partition