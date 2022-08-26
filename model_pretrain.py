# 111

from torchvision import models


vgg16_model_none = models.vgg16()
vgg16_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

print(vgg16_model)
