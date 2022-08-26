from tensorboardX import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.CIFAR10(root="./dataset",
                                 train=True,
                                 download=True,
                                 transform=ToTensor())

Test_data = datasets.CIFAR10(root="./dataset",
                             train=False,
                             download=True,
                             transform=ToTensor())

"""
print(Test_data.classes)
print(training_data.classes)
img, target = training_data[0]
print(target)
print(img)
"""

writer = SummaryWriter('CIFAR_10')
for i in range(10):
    img, target = Test_data[i]
    writer.add_image('test_data', img, i)
writer.close()
