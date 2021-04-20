from graphx import WeightedUndirectedGraph, GraphPartition
from community import Louvain, Leiden

graph = WeightedUndirectedGraph()
edge_list = \
[
    [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 2], [1, 4], [1, 7],
    [2, 4], [2, 5], [2, 6],
    [3, 7],
    [4, 10],
    [5, 7], [5, 11],
    [6, 7], [6, 11],
    [8, 9], [8, 10], [8, 11], [8, 14], [8, 15],
    [9, 12], [9, 14],
    [10, 11], [10, 12], [10, 13], [10, 14],
    [11, 13]
]

graph.add_edges_from_list(edge_list)

louvain = Leiden()

partition = louvain.fit(graph)

print(partition.get_partition())