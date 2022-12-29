import torch

from commons import hat, vee

g = 9.81
e3 = torch.tensor([0., 0., 1.]).reshape([3, 1]).float()

class Quadrotor:
  def __init__(self, init_params_kx = 1., init_params_kv = 1., init_params_kr = 1., init_params_ko = 1.,
               position= [0., 0., 0.], vel = [0., 0., 0.], pose = [0, 0, 0], angular_vel = [0., 0., 0.], 
               m = 0.18, J = [0.00025, 0.000232, 0.0003738], d = 0.315, cf_motor = 8.004 * 0.0001):
    self.position = torch.tensor([position]).T;
    self.vel = torch.tensor([vel]).T;
    self.pose = torch.eye(3)
    self.angular_vel = torch.tensor([angular_vel]).T;
    self.params_kx = torch.tensor([init_params_kx]).T;
    self.params_kv = torch.tensor([init_params_kv]).T;
    self.params_kr = torch.tensor([init_params_kr]).T;
    self.params_ko = torch.tensor([init_params_ko]).T;

    self.m = m
    self.J = torch.tensor([
      [J[0], 0., 0.], [0., J[1], 0.], [0., 0., J[2]]
    ])
    self.J_inv = torch.inverse(self.J)
    self.d = d
    self.cf_motor = cf_motor

    self.max_f = torch.tensor([self.m * 2 * g])
    self.min_f = 0
    self.arm_length = 0.086

  def set_params(self, params_kx, params_kv, params_kr, params_ko):
    self.params_kx = torch.tensor([params_kx]).T;
    self.params_kv = torch.tensor([params_kv]).T;
    self.params_kr = torch.tensor([params_kr]).T;
    self.params_ko = torch.tensor([params_ko]).T;

  def get_controller_output(self, desired_position, desired_velocity, desired_acceleration, desired_pose, desired_angular_vel, desired_angular_acc):
    err_position = self.position - desired_position
    err_vel = self.vel - desired_velocity

    b3_des = - self.params_kx * err_position - self.params_kv * err_vel - self.m * g * e3 + self.m * desired_acceleration
    b3_des = b3_des.float()
    f = -torch.mm(b3_des.T, torch.mm(self.pose, e3))

    err_pose = 0.5 * (torch.mm(desired_pose.T, self.pose) - torch.mm(self.pose.T, desired_pose))
    err_pose = vee(err_pose)

    err_angular_vel = self.angular_vel - torch.mm(self.pose.T, torch.mm(desired_pose, desired_angular_vel))

    M = -self.params_kr * err_pose - self.params_ko * err_angular_vel + torch.cross(self.angular_vel, torch.mm(self.J, self.angular_vel))
    temp_M = torch.mm(hat(self.angular_vel), torch.mm(self.pose.T, torch.mm(desired_pose, desired_angular_vel))) - torch.mm(self.pose.T, torch.mm(desired_pose, desired_angular_acc))
    M = M - torch.mm(self.J, temp_M)

    # limit motion
    A = torch.tensor(
      [
        [0.25, 0., -0.5/self.arm_length],
        [0.25, 0.5/self.arm_length, 0],
        [0.25, 0, 0.5/self.arm_length],
        [0.25, -0.5/self.arm_length, 0]
      ]
    )
    prop_thrust = torch.mm(A, torch.tensor([f, M[0], M[1]]).reshape([3,1]))
    prop_thrusts_clamped = torch.max(torch.min(torch.tensor(self.max_f/4), prop_thrust), torch.tensor(self.min_f/4))

    B = torch.tensor([[1, 1, 1, 1],
                      [0, self.arm_length, 0, -self.arm_length],
                      [-self.arm_length, 0, self.arm_length, 0],
                      [-self.arm_length, self.arm_length, -self.arm_length, self.arm_length]
                      ]);
    f = torch.mm(B, prop_thrusts_clamped)[0]
    M = torch.mm(B, prop_thrusts_clamped)[1:].reshape([3, 1])

    print(f)
    return (f, M)

  def state_update(self, f, M, dt):
    f = torch.mm(self.pose.T, f * e3)
    f += self.m * g * e3

    acc = f / self.m
    vel_end = self.vel + acc * dt
    position_end = self.position + self.vel * dt

    angular_vel_asymmetric_matrix = hat(torch.mm(self.pose, self.angular_vel))
    pose_end = torch.mm(angular_vel_asymmetric_matrix * dt + torch.eye(3), self.pose)
    pose_end[:, 0] = pose_end[:, 0] / torch.norm(pose_end[:, 0])
    pose_end[:, 1] = pose_end[:, 1] / torch.norm(pose_end[:, 1])
    pose_end[:, 2] = pose_end[:, 2] / torch.norm(pose_end[:, 2])

    pqrdot = torch.mm(self.J_inv, M - torch.cross(self.angular_vel, torch.mm(self.J, self.angular_vel)))
    angular_end = pqrdot * dt + self.angular_vel

    return (
        position_end,
        pose_end,
        vel_end,
        angular_end
    )

  def set_state(self, state):
    self.position = state[0]
    self.pose = state[1]
    self.vel = state[2]
    self.angular_vel = state[3]

  def state_update_in_place(self, f, M, dt):
    self.set_state(
        self.state_update(
            f, M, dt
        )
    )
  