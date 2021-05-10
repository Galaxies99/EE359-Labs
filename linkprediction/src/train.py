import os
import yaml
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.optim import AdamW
from node2vec import node2vec
from dataset import n2vDataset, load_graph
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'node2vec.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg
with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

GRAPH_FILE = cfg_dict.get('graph_file', 'data/course3_edge.csv')
TEST_FILE = cfg_dict.get('test_file', 'data/course3_test.csv')
NUM_WALKS = cfg_dict.get('num_walks', 40)
WALK_LENGTH = cfg_dict.get('walk_length', 20)
EMBEDDING_DIM = cfg_dict.get('embedding_dim', 256)
EPOCH_NUM = cfg_dict.get('epoch_num', 60)
BATCH_SIZE = cfg_dict.get('batch_size', 128)
P = cfg_dict.get('p', 1)
Q = cfg_dict.get('q', 1)
K = cfg_dict.get('k', 20)

graph, testing_edges = load_graph(GRAPH_FILE)
model = node2vec(graph, NUM_WALKS, WALK_LENGTH, EMBEDDING_DIM, P, Q, K)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
dataset = n2vDataset(graph)
dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)
optimizer = AdamW(model.parameters(), lr = 0.01)


def train_one_epoch(model, optimizer, epoch_num, display = True):
    print('*** Epoch: {}'.format(epoch_num + 1))
    model.train()
    for idx, nodes in enumerate(dataloader):
        optimizer.zero_grad()
        sample = model.sample(list(nodes.detach().numpy()), device)
        loss = model.loss(nodes, sample, device)
        loss.backward()
        optimizer.step()
        if display:
            print('Batch: {}, Loss: {}'.format(idx + 1, loss.item()))


def train(model, optimizer, epochs, display = True):
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, epoch, display = display)


if __name__ == '__main__':
    train(model, optimizer, 60, display = True)