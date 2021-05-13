import os
import yaml
import torch
import argparse
from graphx import Graph
from node2vec import node2vec
from dataset import n2vTestDataset, load_testing_set
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'node2vec.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg
with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

GRAPH_FILE = cfg_dict.get('graph_file', 'data/course3_edge.csv')
TEST_FILE = cfg_dict.get('test_file', 'data/course3_test.csv')
DUMP_FILE = cfg_dict.get('test_result_file', 'submission.csv')
CHECKPOINT_DIR = cfg_dict.get('checkpoint_dir', 'checkpoint')
NODE_NUMBER = cfg_dict.get('node_number', 0)
NUM_WALKS = cfg_dict.get('num_walks', 40)
WALKING_POOL_SIZE = cfg_dict.get('walking_pool_size', 50)
WALK_LENGTH = cfg_dict.get('walk_length', 20)
EMBEDDING_DIM = cfg_dict.get('embedding_dim', 256)
EPOCH_NUM = cfg_dict.get('epoch_num', 300)
BATCH_SIZE = cfg_dict.get('batch_size', 128)
LEARNING_RATE = cfg_dict.get('learning_rate', 0.01)
MILESTONES = cfg_dict.get('milestones', [100])
GAMMA = cfg_dict.get('gamma', 0.1)
P = cfg_dict.get('p', 0.5)
Q = cfg_dict.get('q', 2)
K = cfg_dict.get('k', 30)

test_edges = load_testing_set(TEST_FILE)
dataset = n2vTestDataset(test_edges)
# For testing/inference, just use batch size of 1 for convenience.
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

start_epoch = 0
if os.path.exists(CHECKPOINT_DIR) == False:
    raise AttributeError('No checkpoint file!')
checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):    
    checkpoint = model = torch.load(checkpoint_file, map_location=device)
    graph = Graph(NODE_NUMBER)
    graph.load_state_dict(checkpoint['graph'])
    # For testing, no need to create pool.
    model = node2vec(graph, NUM_WALKS, -1, WALK_LENGTH, EMBEDDING_DIM, P, Q, K)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print('Checkpoint file loaded (epoch {})'.format(start_epoch))
else:
    raise AttributeError('No checkpoint file!')

model.to(device)


def test(model, epoch):
    print('*** Inference on model of epoch {}'.format(epoch))
    model.eval()
    res_dict = {}
    for idx, data in enumerate(dataloader):
        id, source, target = data
        source = source.to(device)
        target = target.to(device)
        with torch.no_grad():
            pred = model.link_prediction(source, target)
        res_dict[id.item()] = pred.item()
    return res_dict


def dump_result(res, file):
    id = 0
    res_list = []
    while id in res.keys():
        res_list.append(res[id])
        id += 1
    with open(file, 'w') as f:
        f.write('id, label\n')
        for i, res in enumerate(res_list):
            f.write('{}, '.format(i))
            f.write('%.4f' % res)
            f.write('\n')


if __name__ == '__main__':
    res = test(model, start_epoch)
    dump_result(res, DUMP_FILE)