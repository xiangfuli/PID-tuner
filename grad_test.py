import torch
from torch.autograd import grad
from utils.commons import quaternion_2_rotation_matrix
from torch.autograd.functional import jacobian

a = torch.tensor([[0.9346231 ], [ 0.0853563], [0.0512138], [0.3414252]], requires_grad=True)


grad = jacobian(quaternion_2_rotation_matrix, a)

print(quaternion_2_rotation_matrix(a))

# print(grad)