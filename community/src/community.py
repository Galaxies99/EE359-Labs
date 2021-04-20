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
    def __init__(self, epsilon = 1e-6, minimum_dQ = 0):
        '''
        Initialization.

        Parameters
        ----------
        epsilon: float, optional, default: 1e-6, the epsilon value that indicates convergence;
        minimum_dQ: float, optional, default: 0, the minimum allowed dQ.
        '''
        super(Louvain, self).__init__()
        self.epsilon = epsilon
        self.minimum_dQ = minimum_dQ
        
    
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
            print('Modularity in Epoch:', last_Q)
            partition, graph = self._restructure(partition, graph)
            partition, Q = self._partition(partition, graph, last_Q)
            partition_list.append(partition.copy())

            if abs(Q - last_Q) < self.epsilon:
                break

            last_Q = Q
        
        return self._combine_partition_list(partition_list, G)

    
    def _partition(self, partition, graph, Q = None):
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
        if Q is None:
            Q = partition.modularity()
        
        no_gain = False
        while not no_gain:
            print('Modularity in Iteration: ', Q)
            no_gain = True
            # Randomize accessing nodes in G
            for x in random.sample(graph.iter_nodes().keys(), len(graph.iter_nodes().keys())):
                x_community = partition.get_community(x)
                max_dQ, com = self.minimum_dQ, 0
                appeared = []
                for y in graph.iter_edges(x).keys():
                    y_community = partition.get_community(y)
                    if x_community == y_community:
                        continue
                    if y_community in appeared:
                        continue
                    dQ = partition.modularity_gain(x, y_community)
                    appeared.append(y_community)
                    if dQ > max_dQ:
                        max_dQ = dQ
                        com = y_community
                        
                if max_dQ > self.minimum_dQ:
                    partition.assign_community(x, com)
                    Q = Q + max_dQ
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
        print('size = ', graph.size())
        new_graph = WeightedUndirectedGraph()
        for x in graph.iter_nodes():
            x_community = partition.get_community(x)
            for y, w in graph.iter_edges(x).items():
                y_community = partition.get_community(y)
                if x == y:
                    new_graph.add_edge(x_community, y_community, w)
                else:
                    new_graph.add_edge(x_community, y_community, w / 2)
        new_partition = GraphPartition(new_graph)
        print('new size = ', new_graph.size())
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