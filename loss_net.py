# 111
import torch
from torch.nn import Module, Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class Roiben(Module):
    def __init__(self):
        super(Roiben, self).__init__()
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        input = self.model(input)
        return input


test_data = datasets.CIFAR10(root="./dataset",
                             train=False,
                             download=True,
                             transform=ToTensor())

test_loader = DataLoader(dataset=test_data,
                         batch_size=1,
                         shuffle=True)

net = Roiben()
i = 1
loss_cross = CrossEntropyLoss(reduction='mean')
for data in test_loader:
    img_tensor, target = data
    out = net(img_tensor)
    loss = loss_cross(out, target)
    if i == 1:
        print('target:', target)
        print('out:', out)
        print('loss:', loss)
        i = i - 1
