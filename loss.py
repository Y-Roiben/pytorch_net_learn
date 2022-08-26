# 111
import torch
from torch import nn

input = torch.tensor([1, 2, 3], dtype=torch.float).reshape(1, 3)
targets = torch.tensor([1, 2, 5], dtype=torch.float).reshape(1, 3)
print('-----------l = |x-y|-----------------')
loss1 = nn.L1Loss(reduction='mean')
result1 = loss1(input, targets)
print('LOSS mean:', result1)

loss2 = nn.L1Loss(reduction='sum')
result2 = loss2(input, targets)
print('LOSS sum:', result2)

print('-----------l = (x-y)^2-----------------')
loss3 = nn.MSELoss(reduction='mean')
result3 = loss3(input, targets)
print('MES LOSS mean:', result3)

loss4 = nn.MSELoss(reduction='sum')
result4 = loss4(input, targets)
print('MES LOSS sum:', result4)

"""交叉熵损失"""
print('-----------交叉熵---------------')
x = torch.tensor([0.1, 0.2, 0.3]).reshape((1, 3))
target = torch.tensor([2])
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, target)
print(result_cross)
