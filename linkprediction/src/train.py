import os
import yaml
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from node2vec import node2vec
from dataset import n2vDataset, n2vValDataset, load_graph
from torch.utils.data import DataLoader
from criterion import calc_auc

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'node2vec.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg
with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

GRAPH_FILE = cfg_dict.get('graph_file', 'data/course3_edge.csv')
TEST_FILE = cfg_dict.get('test_file', 'data/course3_test.csv')
CHECKPOINT_DIR = cfg_dict.get('checkpoint_dir', 'checkpoint')
NUM_WALKS = cfg_dict.get('num_walks', 40)
WALK_LENGTH = cfg_dict.get('walk_length', 20)
EMBEDDING_DIM = cfg_dict.get('embedding_dim', 256)
EPOCH_NUM = cfg_dict.get('epoch_num', 60)
BATCH_SIZE = cfg_dict.get('batch_size', 128)
LEARNING_RATE = cfg_dict.get('learning_rate', 0.01)
MILESTONES = cfg_dict.get('milestones', [30, 60, 90, 120, 150])
GAMMA = cfg_dict.get('gamma', 0.1)
P = cfg_dict.get('p', 1)
Q = cfg_dict.get('q', 1)
K = cfg_dict.get('k', 20)

graph, val_edges, val_labels = load_graph(GRAPH_FILE)
model = node2vec(graph, NUM_WALKS, WALK_LENGTH, EMBEDDING_DIM, P, Q, K)
dataset = n2vDataset(graph)
val_dataset = n2vValDataset(val_edges, val_labels)
dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = 128, shuffle = True)
optimizer = AdamW(model.parameters(), lr = 0.01)
lr_scheduler = MultiStepLR(optimizer, milestones = MILESTONES, gamma = GAMMA)

start_epoch = 0
if os.path.exists(CHECKPOINT_DIR) == False:
    os.mkdir(CHECKPOINT_DIR)
checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


def train_one_epoch(model, optimizer, lr_scheduler, epoch_num, display = True):
    print('*** Epoch: {}, Current learning rate: {}'.format(epoch_num + 1, lr_scheduler.get_last_lr()[0]))
    model.train()
    for idx, nodes in enumerate(dataloader):
        optimizer.zero_grad()
        sample = model.sample(list(nodes.detach().numpy()), device)
        loss = model.loss(nodes, sample, device)
        loss.backward()
        optimizer.step()
        if display:
            print('Batch: {}, Loss: {}'.format(idx + 1, loss.item()))


def eval_one_epoch(model, epoch_num, display = True):
    print('*** Epoch {} Testing'.format(epoch_num))
    model.eval()
    pred_list, label_list = [], []
    for idx, data in enumerate(val_dataloader):
        source, target, label = data
        source = source.to(device)
        target = target.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model.link_prediction(source, target)
        pred_list.append(pred)
        label_list.append(label)
    preds = torch.cat(pred_list, dim = 0)
    labels = torch.cat(label_list, dim = 0)
    print('AUC: ', calc_auc(preds.cpu().numpy(), labels.cpu().numpy()))


def train(model, optimizer, lr_scheduler, epochs, start_epoch, display = True):
    for epoch in range(start_epoch, epochs):
        train_one_epoch(model, optimizer, lr_scheduler, epoch, display = display)
        eval_one_epoch(model, epoch, display = display)
        lr_scheduler.step()
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict()
        }
        torch.save(save_dict, checkpoint_file)


if __name__ == '__main__':
    train(model, optimizer, lr_scheduler, EPOCH_NUM, start_epoch, display = True)