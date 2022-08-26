# 111
import io

import torch
import torch.nn as nn

Relu = nn.ReLU()
Sigmoid = nn.Sigmoid()
input = torch.rand((3, 3))
print(input)
a = torch.tensor([[1, 3, -1],
                  [4, 1, -4],
                  [3, 5, 3]])
print(Relu(a))
print(Sigmoid(Relu(a)))

m = nn.Linear(2, 3, bias=True)
input = torch.randn(3, 3, 2)
print('in:', input)
output = m(input)
print('out:', output)

# dim=0 跨行球softmax
soft = nn.Softmax(dim=0)
input = torch.randn(3, 3)
output = soft(input)
print('----------------------------')
print(input)
print(output)

a = torch.ones(2, 3)
print(a)

