import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from graphx import Graph
from torch.utils.data import Dataset


class n2vDataset(Dataset):
    '''
    The dataloader of node2vec framework.

    Members
    -------
    self.graph: graphx.Graph object, the graph.
    '''
    def __init__(self, graph: Graph):
        self.graph = graph
        self.nodes = self.graph.iter_nodes()
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, index):
        return self.nodes[index]


def load_graph(file_path):
    df = pd.read_csv(file_path)
    data = np.array(df)
    graph = Graph()
    graph.add_edges_from_list(data)
    return graph, []