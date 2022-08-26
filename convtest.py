# 111
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

Test_data = datasets.CIFAR10(root="./dataset",
                             train=False,
                             download=False,
                             transform=ToTensor())

# datas = DataLoader(Test_data, batch_size=64)


class CONV(nn.Module):
    def __init__(self):
        super(CONV, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x):
        return self.conv(x)


test = CONV()
i = 1

for data in Test_data:
    img_ten, target = data
    output = test.forward(img_ten)
    if i == 1:
        toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        pic1 = toPIL(output)
        plt.imshow(pic1)
        plt.axis('off')
        plt.show()
        pic = toPIL(img_ten)
        plt.imshow(pic)
        plt.axis('off')
        plt.show()
        i = 0


'''
toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
pic = toPIL(output)
plt.imshow(pic)
plt.show()

'''

