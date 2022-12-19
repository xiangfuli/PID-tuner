import torch
from torch.autograd import grad

a = torch.tensor([1., 2.], requires_grad=True)

b = torch.matmul(torch.tensor([a[0], 1]), (a ** 2))

b.sum().backward()

print(a.grad)

a1 = torch.tensor([1.], requires_grad=True)
a2 = torch.tensor([2.], requires_grad=True)

v1 = torch.tensor([a1, 1], requires_grad=True)

b = torch.matmul(v1, (torch.tensor([a1, a2], requires_grad=True) ** 2))

b.sum().backward()

print(a1.grad)


aa1 = torch.tensor([1.], requires_grad=True)

aa1_res = aa1 * 1 + (1 - torch.sin(aa1)) * aa1

aa1_res = (1 - aa1_res) * torch.cos(aa1_res) * aa1_res + aa1

aa1_res.sum().backward()

print(aa1.grad)

mod = torch.tensor([2.], requires_grad=True)

# res = torch.atan2(torch.sin(mod), torch.cos(mod))
# res = torch.remainder(mod, torch.pi)

# res = torch.max(torch.tensor(1.), res)

# res.sum().backward()

# print(mod.grad)





# pp = torch.tensor([3.], requires_grad=True)

# p2 = 1 - pp

# p3 = p2 ** 2

# p3.backward()

# print(pp.grad)


tem = torch.tensor([1.], requires_grad=True)
tem1 = torch.tensor([1.], requires_grad=True)

aaaa = torch.where(tem < 0, tem, tem1)


loss = aaaa * tem

loss.backward()

print(tem.grad)