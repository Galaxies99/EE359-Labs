import copy


class WeightedUndirectedGraph(object):
    '''
    An object that records a weighted undirected graph
    
    Members
    -------
    WeightedUndirectedGraph.graph: dict of dict, the graph;
    WeightedUndirectedGraph.deg: the precomputed weighted degree of each node in the graph
    WeightedUndirectedGraph.edge_deg: the precomputed degree of each node in the graph
    '''
    def __init__(self):
        '''
        Initialize the graph as an empty graph.
        '''
        self._graph = {}
        self._degree = {}
        self._edge_degree = {}
        self._size = 0
        self._edge_size = 0
    
    def add_node(self, node):
        '''
        Add a node in the graph

        Parameters
        ----------
        node: the given index of the node
        '''
        if node in self._graph.keys():
            return
        self._graph[node] = {}
        self._degree[node] = 0
        self._edge_degree[node] = 0
    
    def add_edge(self, source, target, weight = 1):        
        '''
        Add an (weighted) undirected edge in the graph, note that multiple edges are combined into one.

        Parameters
        ----------
        source: the source node of the edge, if the node does not exist in the graph, then create a new node;
        target: the target node of the edge, if the node does not exist in the graph, then create a new node;
        weight: int, optional, default: 1, the weight of the edge.
        '''
        self.add_node(source)
        self.add_node(target)
        self._graph[source][target] = self._graph[source].get(target, 0) + weight
        self._graph[target][source] = self._graph[target].get(source, 0) + weight
        self._degree[source] += weight
        self._degree[target] += weight
        self._size += weight
        self._edge_size += 1
        self._edge_degree[source] += 1
        self._edge_degree[target] += 1
    
    def add_edges_from_list(self, edge_list):
        '''
        Add edges from edge list.

        Parameters
        ----------
        edge_list: the given edge list, which should follow the following format:
            [e1, e2, ..., en] where ei = [source_i, target_i, (weight_i)]
        '''
        for edge in edge_list:
            assert len(edge) == 2 or len(edge) == 3
            if len(edge) == 2:
                self.add_edge(edge[0], edge[1])
            else:
                self.add_edge(edge[0], edge[1], edge[2])

    def iter_nodes(self):
        '''
        Get an iterative dict of all nodes in the graph, which is used to enumerate nodes.

        Returns
        -------
        An iterative dict of all nodes.

        Usage
        -----
        graph = WeightedUndirectedGraph()
        graph.add_edge(2, 3)
        graph.add_edge(1, 4)
        for node in graph.iter_nodes():
            print(node)
        '''
        return self._graph
    
    def iter_edges(self, node):
        '''
        Get an iterative dict of all edges in the graph linking the given node, which is used to enumerate edges.

        Parameters
        ----------
        node: the target node.

        Returns
        -------
        An iterative dict of all edges.

        Usage
        -----
        graph = WeightedUndirectedGraph()
        graph.add_edge(2, 3)
        graph.add_edge(2, 4)
        for target_node, weight in graph.iter_edges(2).items():
            print(target_node, weight)        
        '''
        return self._graph[node]
    
    def degree(self, node):
        '''
        Get the weighted degree of the given node in the graph.

        Parameters
        ----------
        node: the target node.

        Returns
        -------
        The weighted degree of the node.
        '''
        return self._degree.get(node, 0)
    
    def edge_degree(self, node):
        '''
        Get the unweighted degree of the given node in the graph, i.e., the number of edges that link the node.

        Parameters
        ----------
        node: the target node.

        Returns
        -------
        The unweighted degree of the node.
        '''
        return self._edge_degree.get(node, 0)
    
    def size(self):
        '''
        Get the weighted size of the graph, i.e., the sum of edge weights.

        Returns
        -------
        The sum of edge weights in the graph.
        '''
        return self._size
    
    def edge_size(self):
        '''
        Get the unweighted size of the graph, i.e., the number of the edges.

        Returns
        -------
        The number of the edges in the graph.
        '''
        return self._edge_size
        
    def copy(self):
        '''
        Copy the current object.

        Returns
        -------
        A copied object.
        '''
        return copy.deepcopy(self)


class GraphPartition(object):
    '''
    An object that records the partition of a given graph

    Members
    -------
    GraphPartition.graph: networkx.Graph object, the given graph;
    GraphPartition.partition: dict, the partition of the graph;
    GraphPartition.cluster_size: list, the size of each cluster partitioned in the graph;
    GraphPartition.num_clusters: int, the number of clusters in the partition.
    '''
    def __init__(self, graph):
        '''
        Initialize the partition as an individual partition.

        Parameters
        ----------
        graph: a WeightedUndirectedGraph object, the graph we focus on.
        '''
        super(GraphPartition, self).__init__()
        self.graph = graph.copy()
        self.partition = {}
        self.cluster_size = []
        self.num_clusters = 0
        for x in graph.iter_nodes():
            self.partition[x] = self.num_clusters
            self.cluster_size.append(1)
            self.num_clusters += 1
    
    def get_community(self, x):
        '''
        Get the community of x in the given partition.

        Parameters
        ----------
        x: int, a given node in the graph.

        Returns
        -------
        the communities that node x lies in; if no communities is found, then return -1.
        '''
        return self.partition.get(x, -1)
    
    def _renumber(self):
        '''
        Re-number the communities
        '''
        self.num_clusters = 0
        self.cluster_size = []
        renumber_map = {}
        old_partition = self.partition.copy()
        for node, community in old_partition.items():
            if community not in renumber_map.keys():
                renumber_map[community] = self.num_clusters
                self.cluster_size.append(0)
                self.num_clusters += 1
            renumbered_community = renumber_map[community]
            self.partition[node] = renumbered_community
            self.cluster_size[renumbered_community] += 1
            
    def assign_community(self, x, com):
        '''
        Assign the community of the node.

        Parameters
        ----------
        x: int, a given node in the graph;
        com: int, the assigned community to the given node.
        '''
        old_partition = self.partition[x]
        self.partition[x] = com
        self.cluster_size[old_partition] -= 1
        self.cluster_size[com] += 1
        if self.cluster_size[old_partition] == 0:
            self._renumber()
    
    def make_individual(self, x):
        '''
        Make the node an individual community.

        Parameters
        ----------
        x: int, a given node in the graph.
        '''
        self.num_clusters += 1
        self.cluster_size.append(0)
        self.assign_community(x, self.num_clusters - 1)
    
    def iter_communities(self):
        '''
        Get an iterative list of the communities in the partition, which is used to enumerate communities.

        Returns
        -------
        An iterative list of the communities in the partition.
        '''
        return range(self.num_clusters)

    def get_partition(self):
        '''
        Get the partition dict of the graph

        Returns
        -------
        A partition dict.
        '''
        return self.partition
    
    def debug(self):
        print(self.cluster_size)
    
    def modularity(self):
        '''
        Compute the modularity of the current partition in the graph.

        Returns
        -------
        The modularity value Q of graph on the current partition.
        '''
        return modularity(self.graph, self)

    def copy(self):
        '''
        Copy the current object.

        Returns
        -------
        A copied object.
        '''
        return copy.deepcopy(self)


def modularity(graph, partition):
    '''
    Compute the modularity of the partition in the graph

    Parameters
    ----------
    graph: a WeightedUndirectedGraph object, the graph which will be decomposed;
    partition: a GraphPartition object, the partition of the given graph.

    Returns
    -------
    The modularity value of graph G on partition S, i.e., Q(G, S).
    '''
    assert type(graph) is WeightedUndirectedGraph
    assert type(partition) is GraphPartition
    
    m = graph.size()
    if m == 0:
        raise AttributeError('There should be at least one edge in the graph, otherwise the modularity value is undefined.')

    degree = {} 
    inside_weight = {}
    for x in graph.iter_nodes():
        community = partition.get_community(x)
        degree[community] = degree.get(community, 0) + graph.degree(x)
        for y, w in graph.iter_edges(x).items():
            if partition.get_community(y) == community:
                inside_weight[community] = inside_weight.get(community, 0) + w
    
    Q = 0  # The modularity value
    for community in partition.iter_communities():
        Q = Q + inside_weight.get(community, 0) / (2 * m) - (degree.get(community, 0) / (2 * m)) ** 2
    
    return Q
