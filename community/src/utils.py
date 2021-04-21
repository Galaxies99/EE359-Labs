import random
import numpy as np
from tqdm import tqdm
from graphx import WeightedUndirectedGraph, GraphPartition


def find_most_occurence(num_list, candidates):
    '''
    Find the most occurence of candidates in num_list

    Parameters
    ----------
    num_list: list, the list to be checked;
    candidates: list, the candidate list.

    Returns
    -------
    The list of the most occurence of candidates.
    '''
    occurence = {}
    for item in num_list:
        assert item in candidates
        occurence[item] = occurence.get(item, 0) + 1
    maximum = 0
    res = []
    for item in candidates:
        if occurence.get(item, 0) > maximum:
            maximum = occurence[item]
            res = [item]
        elif occurence.get(item, 0) == maximum:
            res.append(item)
    return res


def generate_labels(partition, gt):
    partition = partition.renumber().copy()
    n = partition.num_clusters
    gt_record = []
    cluster_candidates = []
    for _ in range(n):
        gt_record.append([])
    for item in gt:
        node, label = item[0], item[1]
        gt_record[partition.get_community(node)].append(label)
    for i in range(n):
        cluster_candidates.append(find_most_occurence(gt_record[i], range(5)))
    num = []
    criterion = 0
    for i in range(n):
        num.append(len(cluster_candidates[i]))
        if num[i] == 1:
            criterion += partition.get_community_size(i) / num[i]
    criterion = criterion / partition.graph.node_size()
    print('total: ', criterion)
    num = np.array(num)
    id = np.argsort(num)

    '''
    for index in tqdm(range(n)):
        com = id[index]
        partition_res = None
        max_dQ = -1e18
        for new_com in cluster_candidates[com]:
            partition_t = partition.copy()
            partition_t.insert_community(5)
            dQ = 0
            for x in partition.get_community_members(com):
                dQ += partition_t.modularity_gain(x, new_com + n)
                partition_t.assign_community(x, new_com + n)
            if dQ > max_dQ:
                partition_res = partition_t.copy()
                max_dQ = dQ
        partition = partition_res.copy()
    '''
    partition.insert_community(5)
    for i in range(n):
        com = cluster_candidates[i][0] + n
        com_members = partition.get_community_members(i).copy()
        for x in com_members:
            partition.assign_community(x, com)
    
    partition = partition.renumber()
    label_gen = partition.get_partition()
    labels = []
    id = 0
    while id in label_gen.keys():
        labels.append(partition.get_community(id))
        id += 1
    return labels, criterion
