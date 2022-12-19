from dynamic_system import DynamicSystem
import torch
import math

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

class Car(DynamicSystem):
  def __init__(self, mass, inertial, initial_state, initial_parameters, dt):
    self.m = mass
    self.I = inertial
    self.dtype = torch.float64
    self.initial_state = initial_state
    self.initial_parameters = initial_parameters
    self.position_x = torch.tensor([self.initial_state[0]], dtype=self.dtype, requires_grad=True)
    self.position_y = torch.tensor([self.initial_state[1]], dtype=self.dtype, requires_grad=True)
    self.velocity = torch.tensor([self.initial_state[2]], dtype=self.dtype, requires_grad=True)
    self.orientation = torch.tensor([self.initial_state[3]], dtype=self.dtype, requires_grad=True)
    self.orientation_dot = torch.tensor([self.initial_state[4]], dtype=self.dtype, requires_grad=True)
    # inputs
    self.input_acc = torch.tensor([0.], requires_grad=True)
    self.input_orientation_ddot = torch.tensor([0.], requires_grad=True)

    self.kp = torch.tensor([self.initial_parameters[0]], dtype=self.dtype, requires_grad=True)
    self.kv = torch.tensor([self.initial_parameters[1]], dtype=self.dtype, requires_grad=True)
    self.kori = torch.tensor([self.initial_parameters[2]], dtype=self.dtype, requires_grad=True)
    self.koridot = torch.tensor([self.initial_parameters[3]], dtype=self.dtype, requires_grad=True)

    self.dt = dt

    self.states = [
      self.position_x,
      self.position_y,
      self.orientation,
      self.velocity,
      self.orientation_dot
    ]

    self.inputs = [
      self.input_acc, 
      self.input_orientation_ddot
    ]

    self.parameters = [
      self.kp,
      self.kv,
      self.kori,
      self.koridot
    ]

    self.max_v = torch.tensor([3.])
    self.max_orientation_dot = torch.tensor([torch.pi]) * 2

  def state_transition(self, desired_state):
    acceleration, orientation_ddot = self.pid_controller_output(desired_state)

    new_orientation_dot_tensor = self.orientation_dot + orientation_ddot * self.dt
    new_orientation_tensor = self.orientation + (self.orientation_dot + new_orientation_dot_tensor) / 2 * self.dt

    orientation_cos = torch.cos(new_orientation_tensor)
    orientation_sin = torch.sin(new_orientation_tensor)

    new_velocity_tensor = self.velocity + acceleration * self.dt
    new_position_x_tensor = self.position_x + (self.velocity + new_velocity_tensor) / 2 * orientation_cos * self.dt
    new_position_y_tensor = self.position_y + (self.velocity + new_velocity_tensor) / 2 * orientation_sin * self.dt
    
    self.position_x = new_position_x_tensor
    self.position_y = new_position_y_tensor
    self.velocity = torch.max(torch.tensor([0.]), torch.min(new_velocity_tensor, self.max_v))
    self.orientation = torch.atan2(torch.sin(new_orientation_tensor), torch.cos(new_orientation_tensor))
    self.orientation_dot = torch.max(-self.max_orientation_dot, torch.min(new_orientation_dot_tensor, self.max_orientation_dot))

    self.states = [
      self.position_x,
      self.position_y,
      self.orientation,
      self.velocity,
      self.orientation_dot
    ]
  
  def pid_controller_output(self, desired_state):
    x_desired, y_desired, vx_desired, vy_desired, accx_desired, accy_desired, angle_desired, angle_dot_desired, angle_ddot_desired = desired_state

    orientation_cos = torch.cos(self.orientation)
    orientation_sin = torch.sin(self.orientation)

    # acceleration output
    acceleration = self.kp * (orientation_cos * (x_desired - self.position_x) + orientation_sin * (y_desired - self.position_y)) \
      + self.kv * (orientation_cos * (vx_desired - self.velocity * orientation_cos) + orientation_sin * (vy_desired - self.velocity * orientation_sin)) \
      + accx_desired * orientation_cos + accy_desired * orientation_sin
    
    err_angle = angle_desired - self.orientation
    if torch.abs(angle_desired - self.orientation) > torch.pi:
      err_angle = torch.where(torch.abs(angle_desired - self.orientation) > torch.pi, - (self.orientation + 2 * torch.pi - angle_desired), angle_desired + 2 * torch.pi - self.orientation)
    orientation_ddot = self.kori * err_angle + self.koridot * (angle_dot_desired - self.orientation_dot) + angle_ddot_desired

    self.input_acc = acceleration
    self.input_orientation_ddot = orientation_ddot

    self.inputs = [
      self.input_acc,
      self.input_orientation_ddot
    ]

    return acceleration, orientation_ddot

  def get_desired_state(self, waypoint, last_desired_state):
    last_desired_orientation, \
    last_desired_orientation_dot, \
    last_desired_orientation_ddot = last_desired_state[-3:]
    current_orientation = math.atan2(waypoint.vel.y, waypoint.vel.x)
    current_ori_dot = get_desired_angular_speed(last_desired_orientation, current_orientation, self.dt)
    return (waypoint.position.x, \
            waypoint.position.y, \
            waypoint.vel.x, \
            waypoint.vel.y, \
            waypoint.acc.x, \
            waypoint.acc.y, \
            current_orientation, \
            current_ori_dot, \
            (current_ori_dot - last_desired_orientation_dot) / self.dt)
  
  def set_parameters(self, parameters):
    self.initial_parameters = parameters

    self.kp = torch.tensor([self.initial_parameters[0]], dtype=self.dtype, requires_grad=True)
    self.kv = torch.tensor([self.initial_parameters[1]], dtype=self.dtype, requires_grad=True)
    self.kori = torch.tensor([self.initial_parameters[2]], dtype=self.dtype, requires_grad=True)
    self.koridot = torch.tensor([self.initial_parameters[3]], dtype=self.dtype, requires_grad=True)
    
    self.parameters = [
      self.kp,
      self.kv,
      self.kori,
      self.koridot
    ]
  
  def reset(self):
    for param in self.parameters:
      del param

    for state in self.states:
      del state

    self.kp = torch.tensor([self.initial_parameters[0]], dtype=self.dtype, requires_grad=True)
    self.kv = torch.tensor([self.initial_parameters[1]], dtype=self.dtype, requires_grad=True)
    self.kori = torch.tensor([self.initial_parameters[2]], dtype=self.dtype, requires_grad=True)
    self.koridot = torch.tensor([self.initial_parameters[3]], dtype=self.dtype, requires_grad=True)

    self.parameters = [
      self.kp,
      self.kv,
      self.kori,
      self.koridot
    ]

    self.position_x = torch.tensor([self.initial_state[0]], dtype=self.dtype, requires_grad=True)
    self.position_y = torch.tensor([self.initial_state[1]], dtype=self.dtype, requires_grad=True)
    self.velocity = torch.tensor([self.initial_state[2]], dtype=self.dtype, requires_grad=True)
    self.orientation = torch.tensor([self.initial_state[3]], dtype=self.dtype, requires_grad=True)
    self.orientation_dot = torch.tensor([self.initial_state[4]], dtype=self.dtype, requires_grad=True)
    self.states = [
      self.position_x,
      self.position_y,
      self.orientation,
      self.velocity,
      self.orientation_dot
    ]
    

