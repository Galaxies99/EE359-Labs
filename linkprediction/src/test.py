import torch
import torch.nn as nn 
weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)

input = torch.IntTensor([[0], [1], [0]])
print(input)
print(torch.unique(input).view(-1, 1))