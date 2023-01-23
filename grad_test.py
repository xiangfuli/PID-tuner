import torch
from torch.autograd import grad
from utils.commons import quaternion_2_rotation_matrix
from torch.autograd.functional import jacobian

a = torch.tensor([[0.9346231 ], [ 0.0853563], [0.0512138], [0.3414252]], requires_grad=True)

grad = jacobian(quaternion_2_rotation_matrix, a)

# print(quaternion_2_rotation_matrix(a))

# print(grad)

def func(input0, input1):
  a = input0[0]
  b = input0[1]

  c = input1[0]
  d = input1[1]

  a_pow = a ** 2
  b_pow = b ** 2
  c_pow = c ** 2
  d_pow = d ** 2

  # return torch.tensor(
  #   [
  #     a_pow,
  #     b_pow,
  #     c_pow,
  #     d_pow,
  #   ]
  # )
  return (a_pow, b_pow, c_pow, d_pow)

grad = jacobian(func, (torch.tensor([2,3]).double(), torch.tensor([5,6]).double()))

print(grad)


