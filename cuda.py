from __future__ import print_function
import torch
import pandas
x = torch.rand(5, 3)
print(x)
rand_x = torch.rand(5, 3)
print(rand_x)
zero_x = torch.zeros(5, 3, dtype=torch.long)
print(zero_x)
ones_x = torch.ones(5, 3, dtype=torch.int)
print(ones_x)
tensor1 = torch.tensor([5.5, 3, 2])
print(tensor1)
x1 = ones_x + tensor1
print(x1)
print('----------------CUDA 张量--------------')

if torch.cuda.is_available():
    device = torch.device("cuda")  # 定义一个 CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 显示创建在 GPU 上的一个 tensor
    x = x.to(device)  # 也可以采用 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # .to() 方法也可以改变数值类型

a = torch.ones(3, 3, 3)
b = torch.rand(3, 3, 3)
print('--------------------连接张量-----------------------')
t1 = torch.cat([a, b], dim=0)
print(t1)
print(t1.size())


print("--------------------------transform-----------------------------------------")





class person:
    def __call__(self, name):
        print('__call__'+'Hello'+name)

    @staticmethod
    def hello(name):
        print('Hello' + name)


p = person()

p('zhangsan')
p.hello('sdf')
