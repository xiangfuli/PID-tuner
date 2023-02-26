from models.dynamic_system import DynamicSystem
import torch
import math

from utils.commons import get_tensor_item, get_shortest_path_between_angles, get_desired_angular_speed

class Car(DynamicSystem):
  def __init__(self, mass, inertial, dt):
    self.m = mass
    self.I = inertial
    self.dtype = torch.float64
    self.dt = dt
    self.desired_state = (0, 0, 0, 0, 0, 0, 0, 0, 0)

  def f(self, k_state, inputs):
    # # use RK4 to get the updated states
    # k1_state = self.euler_state_update(k_state, inputs, self.dt/6)
    # k2_state = self.euler_state_update(k1_state, inputs, self.dt/3)
    # k3_state = self.euler_state_update(k2_state, inputs, self.dt/3)
    # k4_state = self.euler_state_update(k3_state, inputs, self.dt/6)

    # k1_px, k1_py, k1_ori, k1_vel, k1_w = k1_state
    # k2_px, k2_py, k2_ori, k2_vel, k2_w = k2_state
    # k3_px, k3_py, k3_ori, k3_vel, k3_w = k3_state
    # k4_px, k4_py, k4_ori, k4_vel, k4_w = k4_state

    # kp1_px = (k1_px + 2 * k2_px + 2 * k3_px + k4_px) / 6
    # kp1_py = (k1_py + 2 * k2_py + 2 * k3_py + k4_py) / 6
    # kp1_ori = (k1_ori + 2 * k2_ori + 2 * k3_ori + k4_ori) / 6
    # kp1_vel = (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6
    # kp1_w = (k1_w + 2 * k2_w + 2 * k3_w + k4_w) / 6
    
    # return (kp1_px, kp1_py, kp1_ori, kp1_vel, kp1_w)

    # px, py, orientation, vel, w = k_state
    # acceleration, orientation_ddot = inputs
    
    # orientation_cos = torch.cos(orientation)
    # orientation_sin = torch.sin(orientation)
    
    # new_position_x_tensor = px + vel * orientation_cos * self.dt
    # new_position_y_tensor = py + vel * orientation_sin * self.dt
    # new_orientation_tensor = orientation + w * self.dt
    # new_velocity_tensor = torch.max(torch.tensor(0.), vel + acceleration * self.dt)
    # new_orientation_dot_tensor = w + orientation_ddot * self.dt

    # return (new_position_x_tensor, \
    #   new_position_y_tensor, \
    #   new_orientation_tensor, \
    #   new_velocity_tensor, \
    #   new_orientation_dot_tensor)

    k1 = self.derivative(k_state, inputs)
    k2 = self.derivative(self.euler_update(k_state, k1, self.dt / 2), inputs)
    k3 = self.derivative(self.euler_update(k_state, k2, self.dt / 2), inputs)
    k4 = self.derivative(self.euler_update(k_state, k3, self.dt), inputs)

    new_position_x_tensor, \
      new_position_y_tensor, \
      new_orientation_tensor, \
      new_velocity_tensor, \
      new_orientation_dot_tensor = self.euler_update(k_state, (k1 + 2 * k2 + 2 * k3 + k4) / 6, self.dt)
    
    return (new_position_x_tensor, \
      new_position_y_tensor, \
      new_orientation_tensor, \
      new_velocity_tensor, \
      new_orientation_dot_tensor)

  def euler_update(self, state, de, dt):
    pos_x, pos_y, orientation, vel, w = state
    vx, vy, angular, acc, w_dot = de
    return torch.stack(
      [
        pos_x + vx * dt,
        pos_y + vy * dt,
        orientation + angular * dt,
        vel + acc * dt,
        w + w_dot * dt
      ]
    )
    
  def derivative(self, state, input):
        pos_x, pos_y, orientation, vel, w = state
        # acceleration and angular acceleration
        acc, w_dot = input

        return torch.stack(
            [
                vel * torch.cos(orientation),
                vel * torch.sin(orientation),
                w,
                acc,
                w_dot
            ]
        )

  def h(self, k_state, parameters):
    x_desired, y_desired, vx_desired, vy_desired, accx_desired, accy_desired, angle_desired, angle_dot_desired, angle_ddot_desired = self.desired_state
    px, py, orientation, vel, w = k_state
    kp, kv, kori, kw = parameters

    orientation_cos = torch.cos(orientation)
    orientation_sin = torch.sin(orientation)

    # acceleration output
    acceleration = kp * (orientation_cos * (x_desired - px) + orientation_sin * (y_desired - py)) \
      + kv * (orientation_cos * (vx_desired - vel * orientation_cos) + orientation_sin * (vy_desired - vel * orientation_sin)) \
      + accx_desired * orientation_cos + accy_desired * orientation_sin
    
    err_angle = angle_desired - orientation
    orientation_ddot = kori * err_angle + kw * (angle_dot_desired - w) + angle_ddot_desired

    return (acceleration, orientation_ddot)

  def set_parameters(self, parameters):
    self.initial_parameters = parameters

    self.kp = torch.tensor(get_tensor_item(self.initial_parameters[0][0]), dtype=self.dtype, requires_grad=True)
    self.kv = torch.tensor(get_tensor_item(self.initial_parameters[1][0]), dtype=self.dtype, requires_grad=True)
    self.kori = torch.tensor(get_tensor_item(self.initial_parameters[2][0]), dtype=self.dtype, requires_grad=True)
    self.koridot = torch.tensor(get_tensor_item(self.initial_parameters[3][0]), dtype=self.dtype, requires_grad=True)
    
    self.parameters = torch.tensor([
      self.kp,
      self.kv,
      self.kori,
      self.koridot
    ]).reshape([4, 1])
  
  def set_desired_state(self, desired_state):
    self.desired_state = desired_state

  def euler_state_update(self, k_state, inputs, dt):
    px, py, orientation, vel, w = k_state
    acceleration, orientation_ddot = inputs

    orientation_cos = torch.cos(orientation)
    orientation_sin = torch.sin(orientation)

    new_position_x_tensor = px + vel * orientation_cos * dt
    new_position_y_tensor = py + vel * orientation_sin * dt
    new_orientation_tensor = orientation + w * dt
    new_velocity_tensor = vel + acceleration * dt
    new_orientation_dot_tensor = w + orientation_ddot * dt

    return (new_position_x_tensor, \
      new_position_y_tensor, \
      new_orientation_tensor, \
      new_velocity_tensor, \
      new_orientation_dot_tensor)