# 111

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from tensorboardX import SummaryWriter


class MAX(nn.Module):
    def __init__(self):
        super(MAX, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        y = self.max_pool(x)
        return y


input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float)

input = input.reshape((-1, 1, 5, 5))
print(input.size())

max1 = MAX()
output = max1.forward(input)

print(output)

Test_data = datasets.CIFAR10(root="./dataset",
                             train=False,
                             download=False,
                             transform=ToTensor())

datas = DataLoader(Test_data, batch_size=64, shuffle=True)
i = 1
writer = SummaryWriter('max_pool')
for data in datas:
    img, target = data
    print(img.size())
    img_max = max1.forward(img)
    print(img_max.size())
    '''
    if i > 0:
        toPIL = ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        pic1 = toPIL(img_max)
        plt.imshow(pic1)
        plt.axis('off')
        plt.show()
        pic = toPIL(img)
        plt.imshow(pic)
        plt.axis('off')
        plt.show()
        i = i-1
    '''