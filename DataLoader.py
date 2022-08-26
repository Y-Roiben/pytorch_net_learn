from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

test_data = datasets.CIFAR10(root="./dataset",
                             train=False,
                             download=True,
                             transform=ToTensor())

test_loader = DataLoader(dataset=test_data,
                         batch_size=100,
                         shuffle=False,
                         num_workers=0,
                         drop_last=False)


# 测试数据集中第一个样本及target
img, target = test_data[0]
print(img.shape)
print(target)

img, target = test_data[2]
print(img.shape)
print(target)


for data in test_loader:
    img, target = data
    print(img.shape)
    print(target)

