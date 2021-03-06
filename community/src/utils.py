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


def extend_gt(graph, gt):
    '''
    Extend the ground truth nodes to their neighbors.

    Parameters
    ----------
    gt: the ground-truth labels.

    Returns
    -------
    gt_res: the reconstructed ground-truth labels.
    '''
    gt_res = []
    record = {}
    for item in gt:
        node, label = item[0], item[1]
        for x in graph.iter_edges(node).keys():
            if x not in gt[:, 0]:
                record[x] = record.get(x, []) + [label]
        gt_res.append([node, label])
    for node, gt_rec in record.items():
        candidates = list(set(gt_rec))
        if len(candidates) == 1 and len(gt_rec) == 2:
            gt_res.append([node, candidates[0]])
    gt_res = np.array(gt_res)
    # print(gt.shape, gt_res.shape)
    return gt_res


def generate_labels(partition, gt, extended_gt = True, assign_method = 'simple'):
    '''
    Generate labels according to the ground-truth file and the partition.

    Parameters
    ----------
    partition: a GraphPartition object, the partition;
    gt: a np.array object, the ground-truth labels;
    extended_gt: bool, optional, default: True, extend the ground-truth labels;
    assign_method: str, {'simple', 'modularity'}, optional, default: 'simple': the label assigning method.
        - 'simple': the simple assigning method;
        - 'modularity': assign the labels of the community based on the minimum loss of modularity.

    Returns
    -------
    The generated labels, along with the criterion, which is defined as:
       criterion = [the nodes whose labels can be determined with almost 100% confidence] / [the number of nodes]
    '''
    partition = partition.renumber().copy()
    if extended_gt:
        gt = extend_gt(partition.graph, gt)
    n = partition.num_clusters
    gt_record = []
    gt_dict = {}
    cluster_candidates = []
    for _ in range(n):
        gt_record.append([])
    for item in gt:
        node, label = item[0], item[1]
        gt_record[partition.get_community(node)].append(label)
        gt_dict[node] = label
    for i in range(n):
        cluster_candidates.append(find_most_occurence(gt_record[i], range(5)))
    num = []
    criterion = 0
    for i in range(n):
        num.append(len(cluster_candidates[i]))
        if num[i] == 1:
            criterion += partition.get_community_size(i)
    criterion = criterion / partition.graph.node_size()
    print('total: ', criterion)

    if assign_method == 'modularity':
        num = np.array(num)
        id = np.argsort(num)
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
    elif assign_method == 'simple':
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
        if id not in gt_dict.keys():
            labels.append(partition.get_community(id))
        else:
            labels.append(gt_dict[id])
        id += 1
    return labels, criterion
