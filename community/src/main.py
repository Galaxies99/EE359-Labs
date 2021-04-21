import pandas as pd
import numpy as np
from graphx import WeightedUndirectedGraph
from community import Louvain, Leiden
from utils import generate_labels


EDGE_FILE = 'data/edges.csv'
GT_FILE = 'data/ground_truth.csv'

df = pd.read_csv(EDGE_FILE)
data = np.array(df)

graph = WeightedUndirectedGraph()
graph.add_edges_from_list(data)

louvain = Louvain()
partition = louvain.fit(graph)

gt_df = pd.read_csv(GT_FILE)
gt_data = np.array(gt_df)
labels, criterion = generate_labels(partition, gt_data)

LABEL_FILE = 'data/label_{}.csv'.format(criterion)

out_df = pd.DataFrame(labels, columns=['category'])
out_df.index.name = 'id'
out_df.to_csv(LABEL_FILE)
