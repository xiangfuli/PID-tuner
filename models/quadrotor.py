import torch

from utils.commons import hat, vee, quaternion_2_rotation_matrix, angular_vel_2_quaternion_dot

g = 9.81
e3 = torch.tensor([0., 0., 1.]).reshape([3, 1]).double()

class Quadrotor:
  def __init__(self, dt, m = 0.18, J = [0.0829, 0.0845, 0.1377], d = 0.315, cf_motor = 8.004 * 0.0001):
    self.dt = dt
    
    self.m = m
    self.J = torch.tensor([
      [J[0], 0., 0.], [0., J[1], 0.], [0., 0., J[2]]
    ]).double()
    self.J_inv = torch.inverse(self.J)
    self.d = d
    self.cf_motor = cf_motor

    self.max_f = torch.tensor([self.m * 2 * g]).double()
    self.min_f = 0
    self.arm_length = 0.086

    self.desired_state = ()

  def h(self, k_states, parameters, measurement_noise):
    desired_position, desired_velocity, desired_acceleration, desired_pose, desired_angular_vel, desired_angular_acc = self.desired_state
    
    k_states = k_states + measurement_noise

    position = k_states[0:3]
    pose = k_states[3:7]
    vel = k_states[7:10]
    angular_vel = k_states[10:13]
    kp, kv, kori, kw = parameters

    Rwb = quaternion_2_rotation_matrix(pose)

    err_position = position - desired_position
    err_vel = vel - desired_velocity
    
    b3_des = - kp * err_position - kv * err_vel - self.m * g * e3 + self.m * desired_acceleration
    f = -torch.mm(b3_des.T, torch.mm(Rwb, e3))

    err_pose = (torch.mm(desired_pose.T, Rwb) - torch.mm(Rwb.T, desired_pose))
    err_pose = 0.5 * vee(err_pose)

    err_angular_vel = angular_vel - torch.mm(Rwb.T, torch.mm(desired_pose, desired_angular_vel))

    M = - kori * err_pose - kw * err_angular_vel + torch.cross(angular_vel, torch.mm(self.J, angular_vel))
    temp_M = torch.mm(hat(angular_vel), torch.mm(Rwb.T, torch.mm(desired_pose, desired_angular_vel))) - torch.mm(Rwb.T, torch.mm(desired_pose, desired_angular_acc))
    M = M - torch.mm(self.J, temp_M)

    # # limit motion
    # A = torch.tensor(
    #   [
    #     [0.25, 0., -0.5/self.arm_length],
    #     [0.25, 0.5/self.arm_length, 0],
    #     [0.25, 0, 0.5/self.arm_length],
    #     [0.25, -0.5/self.arm_length, 0]
    #   ]
    # ).double()
    # prop_thrust = torch.mm(A, torch.tensor([f, M[0], M[1]]).reshape([3,1]).double())
    # prop_thrusts_clamped = torch.max(torch.min(torch.tensor(self.max_f/4), prop_thrust), torch.tensor(self.min_f/4))

    # B = torch.tensor([[1, 1, 1, 1],
    #                   [0, self.arm_length, 0, -self.arm_length],
    #                   [-self.arm_length, 0, self.arm_length, 0],
    #                   [-self.arm_length, self.arm_length, -self.arm_length, self.arm_length]
    #                   ]).double()
    # f = torch.mm(B, prop_thrusts_clamped)[0]
    # M = torch.mm(B, prop_thrusts_clamped)[1:].reshape([3, 1])

    return torch.stack(
      [torch.max(torch.tensor([0.]), f[0]), M[0], M[1], M[2]]
    ).reshape([4, 1])

  # def f(self, k_states, inputs):
  #   position, pose, vel, angular_vel = k_states[0:3], k_states[3:7], k_states[7:10], k_states[10:13]
    
  #   f = inputs[0]
  #   M = inputs[1:4]
  #   M = M.reshape([3, 1])

  #   pose_R = quaternion_2_rotation_matrix(pose)

  #   f = torch.mm(pose_R, -f * e3)
  #   f += self.m * g * e3

  #   acc = f / self.m
  #   vel_end = vel + acc * self.dt
  #   position_end = position + vel * self.dt

  #   pose_end = pose + angular_vel_2_quaternion_dot(pose, angular_vel) * self.dt
  #   pose_end = pose_end / torch.norm(pose_end)
  #   pqrdot = torch.mm(self.J_inv, M - torch.cross(angular_vel, torch.mm(self.J, angular_vel)))
    
  #   angular_end = pqrdot * self.dt + angular_vel

  #   return (
  #       position_end,
  #       pose_end,
  #       vel_end,
  #       angular_end
  #   )

  def f(self, k_states, inputs, system_plant_noise):
    k1 = self.derivative(k_states, inputs + system_plant_noise)
    k2 = self.derivative(self.euler_update(k_states, k1, self.dt / 2), inputs + system_plant_noise)
    k3 = self.derivative(self.euler_update(k_states, k2, self.dt / 2), inputs + system_plant_noise)
    k4 = self.derivative(self.euler_update(k_states, k3, self.dt), inputs + system_plant_noise)

    updated_state = k_states + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * self.dt
    # normalize quaternion

    return updated_state
  
  def set_desired_state(self, desired_state):
    self.desired_state = desired_state

  def derivative(self, state, input):
    position, pose, vel, angular_speed = state[0:3], state[3:7], state[7:10], state[10:13]
    thrust, M = input[0], input[1:4].reshape([3, 1])
    pose_in_R = quaternion_2_rotation_matrix(pose)

    acceleration = (torch.mm(pose_in_R, -thrust * e3) + self.m * g * e3) / self.m
    w_dot = torch.mm(self.J_inv, M - torch.cross(angular_speed, torch.mm(self.J, angular_speed)))
    
    return torch.concat(
      [
        vel,
        angular_vel_2_quaternion_dot(pose, angular_speed),
        acceleration,
        w_dot
      ]
    )

  def euler_update(self, state, derivative, dt):
    position, pose, vel, angular_speed = state[0:3], state[3:7], state[7:10], state[10:13]
    vel, angular_derivative, acceleration, w_dot = derivative[0:3], derivative[3:7], derivative[7:10], derivative[10:13]

    position_updated = position + vel * dt
    pose_updated = pose + angular_derivative * dt
    pose_updated = pose_updated / torch.norm(pose_updated)
    vel_updated = vel + acceleration * dt
    angular_speed_updated = angular_speed + w_dot * dt

    return torch.concat(
      [
        position_updated,
        pose_updated,
        vel_updated,
        angular_speed_updated
      ]
    )