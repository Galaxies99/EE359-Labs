import pandas as pd
import numpy as np
from graphx import WeightedUndirectedGraph
from community import Louvain


df = pd.read_csv('data/edges.csv')
data = np.array(df)
print('Read data end ...')

graph = WeightedUndirectedGraph()
graph.add_edges_from_list(data)
print('Add edges end ...')

louvain = Louvain(5)
louvain.fit(graph)

print('Louvain end ...')
