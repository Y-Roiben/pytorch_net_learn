# 111
import math

import matplotlib.pyplot as plt
import torch

print(torch.cuda.is_available())

a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
print(a)

b = torch.sin(a)
plt.plot(a.detach(), b.detach())
# plt.show()
c = a.detach()
print(c.requires_grad)

print("----------backward-------------")
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print('z=', z)
print('out=', out)
out.backward()
# 输出梯度 d(out)/dx
print(x.grad)
print('--------------------------------------')
x = torch.ones(5, requires_grad=True)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
print(z)
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss.backward()
print(w.grad)
print(x.grad)
