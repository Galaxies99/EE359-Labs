import torch
import torch.nn as nn
from torch.optim import SGD
from node2vec import node2vec
from dataset import n2vDataset, load_graph
from torch.utils.data import DataLoader


graph, testing_edges = load_graph('data/course3_edge.csv')
model = node2vec(graph, 20, 20, 128, 1, 0.5, 10)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
dataset = n2vDataset(graph)
dataloader = DataLoader(dataset, batch_size = 500, shuffle = True)
optimizer = SGD(model.parameters(), lr = 0.001)


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