import copy


class WeightedUndirectedGraph(object):
    '''
    An object that records a weighted undirected graph
    
    Members
    -------
    WeightedUndirectedGraph._graph: dict of dict, the graph;
    WeightedUndirectedGraph._degree: the precomputed weighted degree of each node in the graph;
    WeightedUndirectedGraph._node_weight: the weight of the each node in the graph;
    WeightedUndirectedGraph._node_size: the node size of the graph;
    WeightedUndirectedGraph._edge_degree: the precomputed degree of each node in the graph;
    WeightedUndirectedGraph._size: the weighted edge size of the graph;
    WeightedUndirectedGraph._edge_size: the edge size of the graph.
    '''
    def __init__(self):
        '''
        Initialize the graph as an empty graph.
        '''
        self._graph = {}
        self._degree = {}
        self._node_weight = {}
        self._node_size = 0
        self._edge_degree = {}
        self._size = 0
        self._edge_size = 0
    
    def add_node(self, node, node_weight = 1):
        '''
        Add a node in the graph

        Parameters
        ----------
        node: the given index of the node
        node_weight: int, optional, default: 1, the weight of the node
        '''
        if node in self._graph.keys():
            return
        self._graph[node] = {}
        self._node_weight[node] = node_weight
        self._degree[node] = 0
        self._edge_degree[node] = 0
        self._node_size += 1
    
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
        if source != target:
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
    
    def node_weight(self, node):
        '''
        Get the weight of the given node in the graph.

        Parameters
        ----------
        node: the target node.

        Returns
        -------
        The node weight.
        '''
        return self._node_weight[node]

    def size(self):
        '''
        Get the weighted size of the graph, i.e., the sum of edge weights.

        Returns
        -------
        The sum of edge weights in the graph.
        '''
        return self._size
    
    def node_size(self):
        '''
        Get the node size of the graph

        Returns
        -------
        The node size of the graph.
        '''
        return self._node_size

    def edge_size(self):
        '''
        Get the unweighted size of the graph, i.e., the number of the edges.

        Returns
        -------
        The number of the edges in the graph.
        '''
        return self._edge_size

    def get_selfcycle(self, node):
        '''
        Get the weight of self-cycle node-node.

        Parameters
        ----------
        node: int, the start node and end node of the self-cycle.

        Returns
        -------
        The weight of the self-cycle started and ended at node
        '''
        return self._graph[node].get(node, 0)
        
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
    GraphPartition.resolution: float, optional, default: 1.0, the resolution of modularity;
    GraphPartition.m2: int, the total degree of the graph;
    GraphPartition.partition: dict, the partition of the graph;
    GraphPartition.nodes: list of list, the nodes contained in each community;
    GraphPartition.nodes_weight: list, the summation of node weights in each community;
    GraphPartition.degree: list, the degree of nodes in each community, used for modularity calculation;
    GraphPartition.inside_weight: list, the inside edge weight in each community, used for modularity calculation;
    GraphPartition.cluster_size: list, the size of each cluster partitioned in the graph;
    GraphPartition.num_clusters: int, the number of clusters in the partition.
    '''
    def __init__(self, graph, resolution = 1.0, initialize_singleton = True):
        '''
        Initialize the partition as an individual partition.

        Parameters
        ----------
        graph: a WeightedUndirectedGraph object, the graph we focus on;
        resolution: float, optional, default: 1.0, the resolution of modularity;
        initialize_singleton: bool, optional, defualt: True, whether to initialize the object as a singleton partition of the graph.
        '''
        super(GraphPartition, self).__init__()
        self.graph = graph.copy()
        self.resolution = resolution
        self.m2 = 2 * self.graph.size()
        self.partition = {}
        self.nodes = []
        self.nodes_weight = []
        self.degree = []
        self.inside_weight = []
        self.cluster_size = []
        self.num_clusters = 0
        if initialize_singleton:
            for x in graph.iter_nodes():
                self.partition[x] = self.num_clusters
                self.nodes.append([x])
                self.nodes_weight.append(self.graph.node_weight(x))
                self.degree.append(graph.degree(x))
                self.inside_weight.append(self.graph.get_selfcycle(x) * 2)
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

    def assign_community(self, x, com):
        '''
        Assign the community of the node.

        Parameters
        ----------
        x: int, a given node in the graph;
        com: int, the assigned community to the given node.
        '''
        if self.partition[x] == com:
            return
        if com >= self.num_clusters:
            raise ValueError('community number is not valid!')
        # Remove from old community, maintain self.nodes, self.degree, self.inside_weight, self.cluster_size
        old_com = self.partition[x]
        self.nodes[old_com].remove(x)
        self.degree[old_com] -= self.graph.degree(x)
        for y, w in self.graph.iter_edges(x).items():
            if self.partition[y] == old_com:
                self.inside_weight[old_com] -= w + w
        self.cluster_size[old_com] -= 1
        self.nodes_weight[old_com] -= self.graph.node_weight(x)

        # Add into new community, maintain self.nodes, self.degree, self.inside_weight, self.cluster_size
        self.partition[x] = com
        self.nodes[com].append(x)
        self.degree[com] += self.graph.degree(x)
        for y, w in self.graph.iter_edges(x).items():
            if self.partition[y] == com:
                self.inside_weight[com] += w + w
        self.cluster_size[com] += 1
        self.nodes_weight[com] += self.graph.node_weight(x)
    
    def insert_community(self, num = 1):
        '''
        Insert new empty communities.

        Parameters
        ----------
        num: int, optional, default: 1, the number of communities you want to insert.
        '''
        if type(num) is not int or num < 0:
            raise ValueError('num should be non-negative integers.')
        for _ in range(num):
            self.nodes.append([])
            self.nodes_weight.append(0)
            self.degree.append(0)
            self.inside_weight.append(0)
            self.cluster_size.append(0)
            self.num_clusters += 1

    def iter_communities(self):
        '''
        Get an iterative list of the communities in the partition, which is used to enumerate communities.

        Returns
        -------
        An iterative list of the communities in the partition.
        '''
        res = []
        for i in range(self.num_clusters):
            if self.cluster_size[i] > 0:
                res.append(i)
        return res

    def get_partition(self):
        '''
        Get the partition dict of the graph.

        Returns
        -------
        A partition dict.
        '''
        return self.partition

    def get_community_size(self, com):
        '''
        Get the size of the community.

        Parameters
        ----------
        com: int, the community.

        Returns
        -------
        The size of the community, 0 if the community not exists.
        '''
        return self.cluster_size[com] if com < self.num_clusters else 0
    
    def get_community_members(self, com):
        '''
        Get the members of the community.

        Parameters
        ----------
        com: int, the community.

        Returns
        -------
        The members of the community.
        '''
        return self.nodes[com] if com < self.num_clusters else []
    
    def get_community_nodes_weight(self, com):
        '''
        Get the weights of all nodes of the community.

        Parameters
        ----------
        com: int, the community.

        Returns
        -------
        The weight of all nodes of the community.
        '''
        return self.nodes_weight[com] if com < self.num_clusters else 0

    def is_singleton(self):
        '''
        Check whether the partition is a singleton partition.

        Returns
        -------
        True if the partition is a singleton partition, False otherwise.
        '''
        for i in range(self.num_clusters):
            if self.cluster_size[i] != 1:
                return False
        return True
    
    def get_degree(self, com):
        '''
        Get the degree of the community.

        Parameters
        ----------
        com: int, the community.

        Returns
        -------
        The degree of the community.
        '''
        return self.degree[com] if com < self.num_clusters else 0

    def renumber(self):
        '''
        Renumber the partitions.

        Returns
        -------
        The renumbered partition. 
        '''
        res = GraphPartition(self.graph, self.resolution, initialize_singleton=False)
        renumber_mapping = {}
        for com in self.iter_communities():
            new_com = res.num_clusters
            renumber_mapping[com] = new_com
            res.nodes.append(self.nodes[com].copy())
            res.nodes_weight.append(self.nodes_weight[com])
            res.degree.append(self.degree[com])
            res.inside_weight.append(self.inside_weight[com])
            res.cluster_size.append(self.cluster_size[com])
            res.num_clusters += 1
        for x in self.graph.iter_nodes().keys():
            res.partition[x] = renumber_mapping[self.partition[x]]
        return res
        

    def modularity(self, optimized = True):
        '''
        Compute the modularity of the current partition in the graph.

        Parameters
        ----------
        optimized: bool, optional, default: True, whether use the precomputed value to optimize the modularity calculation.

        Returns
        -------
        The modularity value Q of graph on the current partition.
        '''
        if optimized:
            Q = 0
            for community in self.iter_communities():
                Q += self.inside_weight[community] / self.m2 - self.resolution * (self.degree[community] / self.m2) ** 2
            return Q
        else:
            return modularity(self.graph, self, self.resolution)

    def modularity_gain(self, x, com):
        '''
        Calculate the modularity gain if we assign com to the community of x.

        Parameters
        ----------
        x: int, a given node in the graph;
        com: int, the assigned community to the node x.

        Returns
        -------
        dQ, the modularity gain.
        '''
        old_com = self.partition[x]
        dQ = (self.degree[com] / self.m2) ** 2 + (self.degree[old_com] / self.m2) ** 2
        dQ -= ((self.degree[com] + self.graph.degree(x)) / self.m2) ** 2 + ((self.degree[old_com] - self.graph.degree(x)) / self.m2) ** 2
        dQ = dQ * self.resolution
        for y, w in self.graph.iter_edges(x).items():
            if x == y:
                continue
            if self.partition[y] == com:
                dQ += (w + w) / self.m2
            elif self.partition[y] == old_com:
                dQ -= (w + w) / self.m2
        return dQ

    def copy(self):
        '''
        Copy the current object.

        Returns
        -------
        A copied object.
        '''
        return copy.deepcopy(self)


def modularity(graph, partition, resolution = 1.0):
    '''
    Compute the modularity of the partition in the graph

    Parameters
    ----------
    graph: a WeightedUndirectedGraph object, the graph which will be decomposed;
    partition: a GraphPartition object, the partition of the given graph;
    resolution: float, optional, default: 1.0, the resolution of the modularity.

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
                if x == y:  # self-cycle
                    inside_weight[community] = inside_weight.get(community, 0) + w
    
    Q = 0  # The modularity value
    for community in partition.iter_communities():
        Q = Q + inside_weight.get(community, 0) / (2 * m) - resolution * (degree.get(community, 0) / (2 * m)) ** 2
    
    return Q
