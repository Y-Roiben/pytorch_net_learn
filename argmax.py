# 111
import torch

output = torch.rand((4, 4))
print(output)
print(output.argmax(1))  # tensor([1, 1]) 第一行第二个数最大，第二行第二个数最大

perds = output.argmax(1)
targets = torch.tensor([0, 1, 2, 3])
print(perds == targets)  # tensor([False,  True]) 第一个预测错，第二个预测对

print((perds == targets).sum())  # tensor(1) 一次预测正确

print((perds == targets).sum().item())
