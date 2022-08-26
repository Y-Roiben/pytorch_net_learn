# 111
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_path[index]
        imd_item_path = os.path.join(self.path, img_name)
        img = Image.open(imd_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


ants_data = MyData('dataset//train', 'ant')
bees_data = MyData('dataset//train', 'bee')
train_datas = ants_data + bees_data

a = 0
for i in range(int(len(ants_data) / 10) + 1):
    plt.figure(figsize=(10, 4))
    for j in range(10):
        if a < int(len(ants_data) / 10):
            index = 10 * a + j
            img, label = ants_data[index]
            plt.subplot(2, 5, j + 1)
            plt.imshow(img)
            plt.axis('off')
        else:
            for k in range(len(ants_data) % 10):
                index = 120 + k
                img, label = ants_data[index]
                plt.imshow(img)
                plt.show()
            break
    plt.show()
    a = a + 1
