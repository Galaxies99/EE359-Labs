import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from graphx import Graph
from torch.utils.data import Dataset


class n2vDataset(Dataset):
    '''
    The training dataset of node2vec framework.

    Members
    -------
    self.graph: graphx.Graph object, the graph.
    '''
    def __init__(self, graph: Graph):
        super(n2vDataset, self).__init__()
        self.graph = graph
        self.nodes = self.graph.iter_nodes()
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, index):
        return self.nodes[index]


class n2vValDataset(Dataset):
    '''
    The validating dataset of node2vec framework.

    Members
    -------
    self.edges: the validating edges of the graph;
    self.labels: the labels of the validating edges.
    '''
    def __init__(self, edges: list, labels: list):
        super(n2vValDataset, self).__init__()
        self.edges = edges
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.edges[index][0], self.edges[index][1], self.labels[index]


def load_graph(file_path: str):
    df = pd.read_csv(file_path)
    data = np.array(df)
    graph = Graph()
    graph.add_edges_from_list(data)

    edge_num = data.shape[0]
    training_edge_num = int(edge_num * 0.8)
    testing_edge_num = edge_num - training_edge_num
    np.random.shuffle(data)
    training_edges = data[:training_edge_num, :]
    testing_edges = data[training_edge_num:, :]
    
    nodes = graph.iter_nodes()
    subgraph = Graph()
    for node in nodes:
        subgraph.add_node(node)
    subgraph.add_edges_from_list(training_edges)

    testing_edges = testing_edges.tolist()
    testing_labels = []
    for _ in range(testing_edge_num):
        testing_labels.append(1)

    for _ in range(testing_edge_num):
        continue_sample = True
        while continue_sample:
            source = nodes[np.random.randint(0, len(nodes))]
            target = nodes[np.random.randint(0, len(nodes))]
            if source != target and graph.get_weight(source, target) == 0:
                testing_edges.append([source, target])
                testing_labels.append(0)
                continue_sample = False
    
    return subgraph, testing_edges, testing_labels