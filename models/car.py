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
    px, py, orientation, vel, w = k_state
    acceleration, orientation_ddot = inputs

    orientation_cos = torch.cos(orientation)
    orientation_sin = torch.sin(orientation)
    
    new_position_x_tensor = px + vel * orientation_cos * self.dt
    new_position_y_tensor = py + vel * orientation_sin * self.dt
    new_orientation_tensor = orientation + w * self.dt
    new_velocity_tensor = vel + acceleration * self.dt
    new_orientation_dot_tensor = w + orientation_ddot * self.dt

    return (new_position_x_tensor, \
      new_position_y_tensor, \
      new_orientation_tensor, \
      new_velocity_tensor, \
      new_orientation_dot_tensor)
    
  
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