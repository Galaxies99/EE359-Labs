import pandas as pd
import numpy as np
from graphx import WeightedUndirectedGraph
from community import Louvain
from utils import find_most_occurence


df = pd.read_csv('data/edges.csv')
data = np.array(df)
print('Read data end ...')

graph = WeightedUndirectedGraph()
graph.add_edges_from_list(data)
print('Add edges end ...')

louvain = Louvain()
partition = louvain.fit(graph)

dict = partition.get_partition()

pre_clusters = len(set(dict.values()))
print(set(dict.values()))
gt = []
gt_mapping = []
for i in range(pre_clusters):
    gt.append([])

df = pd.read_csv('data/ground_truth.csv')
gt_data = np.array(df)
for item in gt_data:
    node, gt_label = item[0], item[1]
    gt[dict[node]].append(gt_label)

for i in range(pre_clusters):
    gt_mapping.append(find_most_occurence(gt[i], range(5)))

id = 0
labels = []
while id in dict.keys():
    labels.append(gt_mapping[dict[id]])
    id += 1

out_df = pd.DataFrame(labels, columns=['category'])
out_df.index.name = 'id'
out_df.to_csv('data/labels.csv')
