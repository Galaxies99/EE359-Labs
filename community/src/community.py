import math
import random
from tqdm import tqdm
from queue import Queue
from graphx import WeightedUndirectedGraph, GraphPartition


class Louvain(object):
    '''
    Use Louvain Algorithm to detect communities in a given graph.

    References
    ----------
    1. Blondel, Vincent D., et al. "Fast unfolding of communities in large networks." Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.

    Members
    -------
    Louvain.epsilon: the epsilon value that indicates convergence;
    Louvain.resolution: the resolution of the modularity;
    Louvain.minimum_dQ: the minimum allowed dQ in the first phase of Louvain algorithm.
    '''
    def __init__(self, epsilon = 1e-6, resolution = 1, minimum_dQ = 0):
        '''
        Initialization.

        Parameters
        ----------
        epsilon: float, optional, default: 1e-6, the epsilon value that indicates convergence;
        resolution: float, optional, default: 1, the resolution of the modularity;
        minimum_dQ: float, optional, default: 0, the minimum allowed dQ.
        '''
        super(Louvain, self).__init__()
        self.epsilon = epsilon
        self.minimum_dQ = minimum_dQ
        self.resolution = resolution
 
    
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
        partition = GraphPartition(graph, self.resolution)
        partition_list = []
        while True:
            partition, Q = self._partition(partition, graph)
            partition_list.append(partition.copy())
            if partition.is_singleton():
                break
            print('Modularity in Epoch:', Q)
            partition, graph = self._restructure(partition, graph)
    
        return self._combine_partition_list(partition_list, G)

    
    def _partition(self, partition, graph, Q = None):
        '''
        The first phase of Louvain algorithm: optimize partition.

        Parameters
        ----------
        partition: a GraphPartition object, the initial partition;
        graph: a WeightedUndirectedGraph object, the given graph;
        Q: float or None, optional, default: None, the precomputed modularity value, None for no precomputed value.

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
        The second phase of Louvain algorithm: ggregate the community and restructure the graph.

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
                    new_graph.add_edge(x_community, y_community, w)
                else:
                    new_graph.add_edge(x_community, y_community, w / 2)
        new_partition = GraphPartition(new_graph, self.resolution)
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


class Leiden(object):
    '''
    Use Leiden Algorithm to detect communities in a given graph.

    References
    ----------
    1. Traag, Vincent A., Ludo Waltman, and Nees Jan Van Eck. "From Louvain to Leiden: guaranteeing well-connected communities." Scientific reports 9.1 (2019): 1-12.
    2. Blondel, Vincent D., et al. "Fast unfolding of communities in large networks." Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.

    Members
    -------
    Leiden.resolution: the resolution of the modularity;
    Leiden.minimum_dQ: the minimum allowed dQ in the first phase of Leiden algorithm.
    '''
    def __init__(self, max_iter = 500, resolution = 1, minimum_dQ = 0):
        super(Leiden, self).__init__()
        self.resolution = resolution
        self.minimum_dQ = minimum_dQ
    
    def fit(self, G):
        '''
        Get the best partition of the graph using Leiden algorithm.

        Parameters
        ----------
        G: a WeightedUndirectedGraph object, the given graph.

        Returns
        -------
        A partition of the graph G.
        '''
        graph = G.copy()
        partition = GraphPartition(graph, self.resolution)

        partition_list = []

        while True:
            partition, Q = self._fast_partition(partition, graph)
            if partition.is_singleton() is True:
                break
            print('iter {} ---> Modularity in Epoch: {}'.format(iter, Q))
            refined_partition, partition, graph = self._restructure(partition, graph)
            partition_list.append(refined_partition.copy())
        
        partition_list.append(partition)
        return self._combine_partition_list(partition_list, G)

    
    def _fast_partition(self, partition, graph, Q = None):
        '''
        The first phase of Leiden algorithm: fast optimize partition.

        Parameters
        ----------
        partition: a GraphPartition object, the initial partition;
        graph: a WeightedUndirectedGraph object, the given graph;
        Q: float or None, optional, default: None, the precomputed modularity value, None for no precomputed value.

        Returns
        -------
        The local-optimal partition.
        '''
        if Q is None:
            Q = partition.modularity()
        
        node_queue = Queue(maxsize = 0)
        in_queue = {}
        for x in random.sample(graph.iter_nodes().keys(), len(graph.iter_nodes().keys())):
            node_queue.put(x)
            in_queue[x] = True
        while not node_queue.empty():
            print('Modularity in Iteration: ', Q)
            x = node_queue.get()
            in_queue[x] = False
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
                for y in graph.iter_edges(x).keys():
                    if in_queue.get(y, False) == False:
                        node_queue.put(y)
                        in_queue[y] = True
        
        partition = partition.renumber().copy()
        return partition, Q

    
    def _restructure(self, partition, graph):
        '''
        The second phase of Leiden algorithm: refine, aggregate the community and restructure the graph.

        Parameters
        ----------
        partition: a GraphPartition object, the partition;
        graph: a WeightedUndirectedGraph object, the original graph.

        Returns
        -------
        refined_partition: a GraphPartition object, the refined partition of the original graph;
        new_partition: a GraphPartition object, the new partition corresponding to the new graph;
        new_graph: a WeightedUndirectedGraph object, the restructured graph.
        '''
        refined_partition = self._get_refined_partition(partition, graph)
        new_graph = WeightedUndirectedGraph()
        for x in graph.iter_nodes():
            x_community = refined_partition.get_community(x)
            for y, w in graph.iter_edges(x).items():
                y_community = refined_partition.get_community(y)
                if x == y:
                    new_graph.add_edge(x_community, y_community, w)
                else:
                    new_graph.add_edge(x_community, y_community, w / 2)
        new_partition = GraphPartition(new_graph, self.resolution)
        return refined_partition, new_partition, new_graph


    def _get_refined_partition(self, partition, graph):
        '''
        Use the current partition to construct the refined partition.

        Parameters
        ----------
        partition: a GraphPartition object, the current partition;
        graph: a WeightedUndirectedGraph object, the original graph.

        Returns
        -------
        refined_partition: a GraphPartition object, the refined partition.
        '''
        refined_partition = GraphPartition(graph)
        for community in tqdm(partition.iter_communities()):
            community_subset = partition.get_community_members(community)
            refined_partition = self._merge_node_subset(refined_partition, graph, community_subset)
        refined_partition.renumber()
        return refined_partition


    def _merge_node_subset(self, partition, graph, community_subset):
        '''
        Refine partition on the given community.

        Parameters
        ----------
        partition: a GraphPartition object, the initial partition;
        graph: a WeightedUndirectedGraph object, the original graph;
        community_subset: list, the community members.

        Returns
        -------
        refined_partition: a GraphPartition object, the refined partition on the given community.
        '''
        refined_partition = partition.copy()
        # Find out the set of well-connected nodes, i.e., R in paper.
        well_connected = []
        for x in community_subset:
            weights = 0
            for y, w in graph.iter_edges(x).items():
                if y in community_subset and y != x:
                    weights += w
            if weights >= self.resolution * 1 * (len(community_subset) - 1):
                well_connected.append(x)
        
        # Visit node in well connected set.
        for x in random.sample(well_connected, len(well_connected)):
            x_community = refined_partition.get_community(x)
            if refined_partition.get_community_size(x_community) == 1:
                candidate_communities = []
                for y, w in graph.iter_edges(x).items():
                    if y not in community_subset:
                        continue
                    y_community = refined_partition.get_community(y)
                    if x_community != y_community and y_community not in candidate_communities:
                        candidate_communities.append(y_community)
                # Find out all well-connected communities, and calculate dQ of them
                well_connected_communities = []
                dQ_communities = []
                theta = 0
                for com in candidate_communities:
                    weights = 0
                    for _x in refined_partition.get_community_members(com):
                        for _y, w in graph.iter_edges(_x).items():
                            if _y in community_subset and refined_partition.get_community(_y) != com:
                                weights += w
                    com_size = refined_partition.get_community_size(com)
                    if weights >= self.resolution * com_size * (len(community_subset) - com_size):
                        dQ = refined_partition.modularity_gain(x, com)
                        if dQ >= 0:
                            well_connected_communities.append(com)
                            dQ_communities.append(dQ)
                            theta += math.exp(dQ)
                # Softmax probability
                rnd = random.random()
                cur_probability = 0.0
                assigned_community = -1
                for i, dQ in enumerate(dQ_communities):
                    cur_probability += math.exp(dQ) / theta
                    if rnd < cur_probability:
                        assigned_community = well_connected_communities[i]
                # Assign new community
                if assigned_community != -1:
                    refined_partition.assign_community(x, assigned_community)

        return refined_partition

    
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
