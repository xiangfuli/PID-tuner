import torch

def get_tensor_item(tensor):
  if torch.is_tensor(tensor):
    return tensor.item()
  else:
    return 0.

def hat(vector):
  return torch.tensor([
    [0, -vector[2], vector[1]],
    [vector[2], 0, -vector[0]],
    [-vector[1], vector[0], 0]
  ])

def vee(skew_symmetric_matrix):
  return torch.tensor(
    [
      -skew_symmetric_matrix[1][2],
      skew_symmetric_matrix[0][2],
      -skew_symmetric_matrix[0][1],
    ]
  ).reshape([3, 1])

def quaternion_2_rotation_matrix(q):
  q = q / torch.norm(q)
  qahat = hat(q[:][1:4])
  return (torch.eye(3) + 2 * torch.mm(qahat, qahat) + 2 * q[0] * qahat).double()

def rotation_matrix_2_quaternion(R):
  tr = R[0,0] + R[1,1] + R[2,2];

  if tr > 0:
    S = torch.sqrt(tr+1.0) * 2
    qw = 0.25 * S;
    qx = (R[2,1] - R[1,2]) / S
    qy = (R[0,2] - R[2,0]) / S
    qz = (R[1,0] - R[0,1]) / S
  elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
    S = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
    qw = (R[2,1] - R[1,2]) / S
    qx = 0.25 * S
    qy = (R[0,1] + R[1,0]) / S
    qz = (R[0,2] + R[2,0]) / S
  elif R[1,1] > R[2,2]:
    S = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
    qw = (R[0,2] - R[2,0]) / S
    qx = (R[0,1] + R[1,0]) / S
    qy = 0.25 * S
    qz = (R[1,2] + R[2,1]) / S
  else:
    S = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
    qw = (R[1,0] - R[0,1]) / S
    qx = (R[0,2] + R[2,0]) / S
    qy = (R[1,2] + R[2,1]) / S
    qz = 0.25 * S

  q = torch.tensor([[qw,qx,qy,qz]]).reshape([4,1])
  q = q * ((qw+0.00000001) / torch.abs(qw + 0.00000001))
  q = q / torch.norm(q)

  return q

def get_shortest_path_between_angles(original_ori, des_ori):
  e_ori = des_ori - original_ori
  if abs(e_ori) > torch.pi:
    if des_ori > original_ori:
      e_ori = - (original_ori + 2 * torch.pi - des_ori)
    else:
      e_ori = des_ori + 2 * torch.pi - original_ori
  return e_ori

def get_desired_angular_speed(original_ori, des_ori, dt):
  return get_shortest_path_between_angles(original_ori, des_ori) / dt

def angular_vel_2_quaternion_dot(quaternion, w):
  qW, qX, qY, qZ = quaternion
  p, q, r = w

  K_quat = 2; #this enforces the magnitude 1 constraint for the quaternion
  quaterror = 1 - (qW**2 + qX**2 + qY**2 + qZ**2)
  
  return -0.5 * torch.mm(torch.tensor(
    [
      [0, p, q, r],
      [-p, 0, -r, q],
      [-q, r, 0, -p],
      [-r, -q, p, 0]
    ], dtype=torch.float64), quaternion)