import random
import numpy as np
from tqdm import tqdm
from graphx import Graph
from sampling import alias_init, alias_sampling


class n2vGraph(object):
    '''
    A graph that supports node2vec random walking.

    Members
    -------
    n2vGraph.graph: graphx.Graph object, the graph;
    n2vGraph.nodes: list, the node list of the graph;
    n2vGraph.p, n2vGraph.q: float, the parameters of probability in node2vec walking process;
    n2vGraph.node_sampling: dict, the alias parameters of node sampling;
    n2vGraph.edge_sampling: dict, the alias parameters of edge sampling;
    n2vGraph.negative_sampling_params: tuple, the alias parameters of negative sampling;
    n2vGraph.walking_pool: dict, the walking pool of each node;
    n2vGraph.walking_pool_size: int, the size of walking pool of each node; non-positive value means no walking pool;
    n2vGraph.walk_length: int, the walking length of the walking pool;
    n2vGraph.id: dict, the mapping of the node to its ID number.

    Reference
    ---------
    Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." 
      Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.
    '''
    def __init__(self, graph: Graph, walking_pool_size: int, walk_length: int, p: float, q: float):
        '''
        Initialize the node2vec graph, along with alias sampling initializations.

        Parameters
        ----------
        graph: graphx.Graph object, the graph;
        walking_pool_size: int, the size of walking pool of each node; non-positive value means no walking pool;
        walk_length: int, the walking length of the walking pool;
        p, q: float, the parameters of probability in node2vec walking process.
        '''
        super(n2vGraph, self).__init__()
        self.graph = graph
        self.nodes = graph.iter_nodes()
        self.walking_pool_size = walking_pool_size
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.id = {}
        self.initialization()
        if self.walking_pool_size > 0:
            self.initialization_walking_pool()
    
    def initialization(self):
        '''
        The initialization process of alias sampling and preprocess the id mapping, along with walking pool initialization.
        '''
        self.node_sampling = {}
        for x in self.nodes:
            prob = [self.graph.get_weight(x, y) for y in self.graph.neighbors(x)]
            if len(prob) > 0:
                self.node_sampling[x] = alias_init(prob, normalized = False)
        
        self.edge_sampling = {}
        for x in self.nodes:
            for y in self.graph.iter_edges(x):
                # x -> y
                prob = []
                for z, weight in self.graph.iter_edges(y).items():
                    # y -> z
                    if z == x:
                        prob.append(weight / self.p)
                    elif self.graph.get_weight(z, x) != 0:
                        prob.append(weight)
                    else:
                        prob.append(weight / self.q)
                if len(prob) > 0:
                    self.edge_sampling[(x, y)] = alias_init(prob)
        
        prob = [self.graph.get_degree(x) for x in self.nodes]
        self.negative_sampling_params = alias_init(prob, normalized = False)

        idx = 0
        for x in self.nodes:
            self.id[x] = idx
            idx += 1
    
    def initialization_walking_pool(self):
        '''
        Initialize the walking pool.
        '''
        self.walking_pool = {}
        for node in tqdm(self.nodes):
            self.walking_pool[node] = self.walk(self.walking_pool_size, self.walk_length, nodes = [node], create_pool = True)
                        
    def random_walk(self, length: int, node: int):
        '''
        Perform a random walk of certain length starting from the given node.

        Parameters
        ----------
        length: int, the length of the random walk
        node: int, the starting node.

        Returns
        -------
        A walk list.
        '''
        walk = [node]
        
        if length != 1:
            neighbors = self.graph.neighbors(node)
            if len(neighbors) > 0:
                walk.append(neighbors[alias_sampling(self.node_sampling[node])])
            else:
                walk.append(node)

        while len(walk) < length:
            y = walk[-1]
            neighbors = self.graph.neighbors(y)
            if len(neighbors) > 0:
                x = walk[-2]
                z = neighbors[alias_sampling(self.edge_sampling[(x, y)])]
                walk.append(z)
            else:
                walk.append(y)
        
        id_walk = []
        for item in walk:
            id_walk.append(self.id[item])
        return id_walk

    def walk(self, num_walks: int, length: int, nodes: list = [], create_pool: bool = False):
        '''
        Perform random walks of certain length starting from each node for certain times.

        Parameters
        ----------
        num_walks: int, the times of random walk starting from each node;
        length: int, the length of the random walk;
        nodes: list, optional, default: [], the list of starting nodes. 
               If nodes is [], then we use all nodes in the graph to walk.
        create_pool: bool, optional, default: False, whether we are creating the walking pool.
        
        Returns
        -------
        The result of random walks.
        '''
        walks = []
        if nodes == []:
            nodes = self.nodes
        if not create_pool and num_walks <= self.walking_pool_size and length == self.walk_length:
            for node in random.sample(nodes, len(nodes)):
                walks = walks + random.sample(self.walking_pool[node], num_walks)
        else:
            for _ in range(num_walks):
                for node in random.sample(nodes, len(nodes)):
                    walks.append(self.random_walk(length, node))
        return walks
    
    def negative_sampling(self, k: int):
        '''
        Perform negative sampling.

        Parameters
        ----------
        k: int, the times of negative sampling.

        Returns
        -------
        A list consists of nodes, which is the result of negative sampling.
        '''
        res = []
        for _ in range(k):
            res.append(alias_sampling(self.negative_sampling_params))
        return res
    
    def get_node_list(self):
        '''
        Get the node list of the graph.

        Returns
        -------
        The node list of the graph.
        '''
        return self.nodes
    