import torch
import torch.nn as nn
import torch.nn.functional as F
from graphwalk import n2vGraph
from graphx import Graph


class node2vec(nn.Module):
    '''
    The main framework of node2vec.

    Members
    -------
    node2vec.graph: n2vGraph object, the graph;
    node2vec.num_walks: int, the number of walks;
    node2vec.walk_length: int, the walking length;
    node2vec.embedding_dim: int, the embedding dim of each node;
    node2vec.embeddings: nn.Embedding, the embedding of each node

    Reference
    ---------
    Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." 
      Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.
    '''
    def __init__(self, graph: Graph, num_walks: int, walking_pool_size: int, walk_length: int, embedding_dim: int, p: float, q: float, k: int):
        '''
        Initialize the node2vec framework.

        Parameters
        ----------
        graph: Graph object, the original graph;
        num_walks, int, the number of walks;
        walking_pool_size: int, the size of walking pool of each node; non-positive value means no walking pool;
        walk_length: int, the walking length;
        embedding_dim: int, the embedding dim of each node;
        p, q: float, the parameters of probability in node2vec walking process;
        k: int, the times of negative sampling.
        '''
        super(node2vec, self).__init__()
        self.graph = n2vGraph(graph, walking_pool_size, walk_length, p, q)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.embedding_dim = embedding_dim
        self.k = k
        num_nodes = len(self.graph.get_node_list())
        self.emb = nn.Embedding(num_embeddings = num_nodes, embedding_dim = self.embedding_dim)

    def sample(self, nodes, device):
        '''
        Perform random walk from given set of starting nodes.

        Parameters
        ----------
        nodes: list, the given set of starting nodes;
        device: the device of the network.

        Returns
        -------
        The sampling random walks.
        '''
        return torch.LongTensor(self.graph.walk(num_walks = self.num_walks, length = self.walk_length, nodes = nodes)).to(device)

    def loss(self, nodes, walks, device):
        '''
        Compute the loss function of the node2vec framework, using negative sampling method.

        Parameters
        ----------
        nodes: list, the given set of starting nodes;
        walks: torch.LongTensor, the sampling random walks;
        device: the device of the network.

        Returns
        -------
        The loss of node2vec framework.
        '''
        node_num = len(nodes)
        start_node = walks[:, [0]]
        walk_node = walks[:, 1:]
        emb_start =  self.emb(start_node)
        emb_walk = self.emb(walk_node)
        loss = - torch.log(torch.sigmoid((emb_start * emb_walk).sum(dim = -1).view(-1))).sum()
        all_node = torch.unique(start_node).view(-1, 1)
        samples = []
        for _ in all_node:
            samples.append(self.graph.negative_sampling(self.k))
        sample_node = torch.LongTensor(samples).to(device)
        emb_node = self.emb(all_node)
        emb_sample = self.emb(sample_node)
        loss = loss + torch.log(torch.sigmoid(emb_node * emb_sample).sum(dim = -1).view(-1)).sum()
        loss = loss / node_num / self.num_walks / (self.walk_length - 1)
        return loss
    
    def link_prediction(self, source, target):
        '''
        Perform link prediction based on node2vec embeddings.

        Parameters
        ----------
        source, target: the predicting edges.

        Returns
        -------
        The prediction of the given edges.
        '''
        for i, src in enumerate(source):
            source[i] = self.graph.id[src.item()]
        for i, tgt in enumerate(target):
            target[i] = self.graph.id[tgt.item()]
        emb_source = self.emb(source)
        emb_target = self.emb(target)
        pred = torch.sigmoid((emb_source * emb_target).sum(dim = -1))
        return pred