import copy


class Graph(object):
    '''
    An object that records a (weighted, undirected) graph
    
    Members
    -------
    Graph._graph: dict of dict, the graph;
    Graph._degree: dict, the degree of each node in the graph.
    '''
    def __init__(self, node_num: int = 0):
        '''
        Initialize the graph as an empty graph.

        Parameters
        ----------
        node_num: int, the number of node. 
        '''
        super(Graph, self).__init__()
        self._graph = {}
        self._degree = {}
        assert node_num >= 0
        for i in range(node_num):
            self.add_node(i)
    
    def add_node(self, node: int):
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
    
    def add_edge(self, source: int, target: int, weight: int = 1):        
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
        self._degree[source] += 1
        self._degree[target] += 1
    
    def add_edges_from_list(self, edge_list: list):
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
        Get an iterative list of all nodes in the graph, which is used to enumerate nodes.

        Returns
        -------
        An iterative list of all nodes.

        Usage
        -----
        graph = Graph()
        graph.add_edge(2, 3)
        graph.add_edge(1, 4)
        for node in graph.iter_nodes():
            print(node)
        '''
        return list(self._graph.keys())
    
    def iter_edges(self, node: int):
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
        return self._graph.get(node, {})
    
    def neighbors(self, node: int):
        '''
        Get the neighbors of the given node in the graph.

        Parameters
        ----------
        node: the target node

        Returns
        -------
        An list of all neighbors of the given node.
        '''
        return list(self._graph.get(node, {}).keys())
    
    def get_weight(self, source: int, target: int):
        '''
        Get the weight of edge between source and target.

        Paramters
        ---------
        source: int, the start node of the edge;
        target: int, the target node of the edge.

        Returns
        -------
        weight: int, the weight of edge between source and target.
        '''
        return self._graph.get(source, {}).get(target, 0)
    
    def get_selfcycle(self, node: int):
        '''
        Get the weight of self-cycle node-node.

        Parameters
        ----------
        node: int, the start node and end node of the self-cycle.

        Returns
        -------
        The weight of the self-cycle started and ended at node
        '''
        return self.get_weight(node, node)
    
    def get_degree(self, node: int):
        '''
        Get the degree of the node.

        Parameters
        ----------
        node: int, the target node.

        Returns
        -------
        The degree of the node; if the node does not exist, return -1.
        '''
        return self._degree.get(node, -1)

    def copy(self):
        '''
        Copy the current object.

        Returns
        -------
        A copied object.
        '''
        return copy.deepcopy(self)
    
    def state_dict(self):
        '''
        Get the state dict of the graph.

        Returns
        -------
        A state dict.
        '''
        return {'graph': self._graph, 'degree': self._degree}

    def load_state_dict(self, state_dict):
        '''
        Load the state dict of the graph.

        Parameters
        ----------
        state_dict: dict, the state dict.
        '''
        self._graph = state_dict.get('graph', {})
        self._degree = state_dict.get('degree', {})