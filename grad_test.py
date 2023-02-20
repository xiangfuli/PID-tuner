import torch
from torch.autograd import grad
from utils.commons import quaternion_2_rotation_matrix
from torch.autograd.functional import jacobian

a = torch.tensor([[0.9346231 ], [ 0.0853563], [0.0512138], [0.3414252]], requires_grad=True)

grad = jacobian(quaternion_2_rotation_matrix, a)

# print(quaternion_2_rotation_matrix(a))

# print(grad)

def func(input0, input1):
  a = input0 ** 2

  c = input1[0]
  d = input1[1]

  c_pow = c ** 2
  d_pow = d ** 2

  # return torch.tensor(
  #     a
  # )
  return torch.stack((a, a))
  # return (a, c_pow, d_pow)

grad = jacobian(func, (torch.tensor([2, 3]).double(), torch.tensor([5,6]).double()))

print(grad)


