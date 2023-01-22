import torch

import sys
sys.path.append(".")

from utils.commons import quaternion_2_rotation_matrix, rotation_matrix_2_quaternion, vee

q0 = torch.tensor([1., 0., 0., 0.]).reshape([4, 1]).double()
# ZXY - 0.01, 0.02, -0.01
q1 = torch.tensor([ 0.9999253, 0.0050497, 0.0099746, -0.0049497 ]).reshape([4, 1]).double() 

time_interval = 0.01

# calculate angular velocity using rotation matrix differention
R0 = quaternion_2_rotation_matrix(q0)
R1 = quaternion_2_rotation_matrix(q1)

pose_error = R1 - R0
w = vee(torch.mm(pose_error / time_interval, R0)).double()

print("angular velocity calculation using rotation matrix differention: %s" % w.reshape([1, 3]))


# calculate angular velocity using axis translation
R_err = torch.mm(torch.t(R0), R1)
q_err = rotation_matrix_2_quaternion(R_err)
q_err = q_err / torch.norm(q_err)
axis = torch.tensor([q_err[1], q_err[2], q_err[3]]).reshape([3, 1])
axis = axis / torch.norm(axis)
angle = 2 * torch.acos(q_err[0])
print("angular velocity calculation using axis translation: %s" % (angle / time_interval * axis).reshape([1, 3]))