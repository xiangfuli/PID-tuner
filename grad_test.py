import torch
from torch.autograd import grad

a = torch.tensor([[1.], [2.], [3.]], requires_grad=True)
a_skew = torch.tensor(
  [
    [0, -a[2], a[1]],
    [a[2], 0, -a[0]],
    [-a[1], a[0], 0]
  ]
)
b = torch.mm(a_skew, a)
b.backward()

print(a.grad)