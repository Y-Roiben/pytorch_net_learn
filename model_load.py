# 111
import torch
from torchvision import models

# 打开方式1 --> 保存方式1
model1 = torch.load('vgg16_method1.pth')
# print(model1)


# 打开方式2 --> 保存方式2
model2 = torch.load('model_weights.pth')
# 输出为字典形式
print(model2)

# 打开方式2 --> 保存方式2
# 恢复成网络模型
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
print(model)
