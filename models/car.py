from models.dynamic_system import DynamicSystem
import torch
import math

from utils.commons import get_tensor_item, get_shortest_path_between_angles, get_desired_angular_speed

class Car(DynamicSystem):
  def __init__(self, mass, inertial, initial_state, initial_parameters, dt):
    self.m = mass
    self.I = inertial
    self.dtype = torch.float64
    self.initial_state = torch.tensor(initial_state).reshape([5, 1]).double()
    self.initial_parameters = torch.tensor(initial_parameters).reshape([4, 1]).double()

    self.position_x = torch.tensor(self.initial_state[0][0], dtype=self.dtype, requires_grad=True)
    self.position_y = torch.tensor(self.initial_state[1][0], dtype=self.dtype, requires_grad=True)
    self.velocity = torch.tensor(self.initial_state[2][0], dtype=self.dtype, requires_grad=True)
    self.orientation = torch.tensor(self.initial_state[3][0], dtype=self.dtype, requires_grad=True)
    self.orientation_dot = torch.tensor(self.initial_state[4][0], dtype=self.dtype, requires_grad=True)
   
    # inputs
    self.input_acc = torch.tensor(0., requires_grad=True)
    self.input_orientation_ddot = torch.tensor(0., requires_grad=True)

    self.kp = torch.tensor(self.initial_parameters[0][0], dtype=self.dtype, requires_grad=True)
    self.kv = torch.tensor(self.initial_parameters[1][0], dtype=self.dtype, requires_grad=True)
    self.kori = torch.tensor(self.initial_parameters[2][0], dtype=self.dtype, requires_grad=True)
    self.koridot = torch.tensor(self.initial_parameters[3][0], dtype=self.dtype, requires_grad=True)

    self.dt = dt

    self.states = torch.tensor([
      self.position_x,
      self.position_y,
      self.orientation,
      self.velocity,
      self.orientation_dot
    ]).reshape([5, 1])

    self.inputs = torch.tensor([
      self.input_acc, 
      self.input_orientation_ddot
    ]).reshape([2, 1])

    self.parameters = torch.tensor([
      self.kp,
      self.kv,
      self.kori,
      self.koridot
    ]).reshape([4, 1])

    self.max_v = torch.tensor([3.])
    self.max_orientation_dot = torch.tensor([torch.pi])

  def state_transition(self, desired_state):
    inputs = self.pid_controller_output(desired_state)
    return self.state_update(inputs)
  
  def state_update(self, inputs):
    acceleration = inputs[0][0]
    orientation_ddot = inputs[1][0]

    orientation_cos = torch.cos(self.orientation)
    orientation_sin = torch.sin(self.orientation)
    
    new_position_x_tensor = self.position_x + self.velocity * orientation_cos * self.dt
    new_position_y_tensor = self.position_y + self.velocity * orientation_sin * self.dt
    new_orientation_tensor = self.orientation + self.orientation_dot * self.dt
    new_velocity_tensor = self.velocity + acceleration * self.dt
    new_orientation_dot_tensor = self.orientation_dot + orientation_ddot * self.dt

    self.position_x = new_position_x_tensor
    self.position_y = new_position_y_tensor
    # self.velocity = torch.max(torch.tensor([0.]), new_velocity_tensor)
    # self.orientation = torch.atan2(torch.sin(new_orientation_tensor), torch.cos(new_orientation_tensor))
    # self.orientation_dot = torch.max(-self.max_orientation_dot, torch.min(new_orientation_dot_tensor, self.max_orientation_dot))
    self.velocity = new_velocity_tensor
    self.orientation = new_orientation_tensor
    self.orientation_dot = new_orientation_dot_tensor

    return torch.tensor([
      self.position_x,
      self.position_y,
      self.orientation,
      self.velocity,
      self.orientation_dot
    ]).reshape([5, 1])
  
  def pid_controller_output(self, desired_state):
    x_desired, y_desired, vx_desired, vy_desired, accx_desired, accy_desired, angle_desired, angle_dot_desired, angle_ddot_desired = desired_state

    orientation_cos = torch.cos(self.orientation)
    orientation_sin = torch.sin(self.orientation)

    # acceleration output
    acceleration = self.kp * (orientation_cos * (x_desired - self.position_x) + orientation_sin * (y_desired - self.position_y)) \
      + self.kv * (orientation_cos * (vx_desired - self.velocity * orientation_cos) + orientation_sin * (vy_desired - self.velocity * orientation_sin)) \
      + accx_desired * orientation_cos + accy_desired * orientation_sin
    
    err_angle = angle_desired - self.orientation
    orientation_ddot = self.kori * err_angle + self.koridot * (angle_dot_desired - self.orientation_dot) + angle_ddot_desired

    self.input_acc = acceleration
    self.input_orientation_ddot = orientation_ddot

    return torch.tensor([acceleration, orientation_ddot]).reshape([2, 1])

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
  
  def set_states(self, states):
    self.position_x = torch.tensor(get_tensor_item(states[0][0]), requires_grad=True)
    self.position_y = torch.tensor(get_tensor_item(states[1][0]), requires_grad=True)
    self.orientation = torch.tensor(get_tensor_item(states[2][0]), requires_grad=True)
    self.velocity = torch.tensor(get_tensor_item(states[3][0]), requires_grad=True)
    self.orientation_dot = torch.tensor(get_tensor_item(states[4][0]), requires_grad=True)

    self.states = states
  
  def reset(self):
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

    self.position_x = torch.tensor(get_tensor_item(self.initial_state[0][0]), dtype=self.dtype, requires_grad=True)
    self.position_y = torch.tensor(get_tensor_item(self.initial_state[1][0]), dtype=self.dtype, requires_grad=True)
    self.velocity = torch.tensor(get_tensor_item(self.initial_state[2][0]), dtype=self.dtype, requires_grad=True)
    self.orientation = torch.tensor(get_tensor_item(self.initial_state[3][0]), dtype=self.dtype, requires_grad=True)
    self.orientation_dot = torch.tensor(get_tensor_item(self.initial_state[4][0]), dtype=self.dtype, requires_grad=True)

    self.states = torch.tensor([
      self.position_x,
      self.position_y,
      self.orientation,
      self.velocity,
      self.orientation_dot
    ]).reshape([5, 1])

    self.inputs = torch.tensor([
      torch.tensor(self.input_acc),
      torch.tensor(self.input_orientation_ddot)
    ]).reshape([2, 1])
  
  def reinit_states_and_params(self):
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

    self.position_x = torch.tensor(get_tensor_item(self.states[0][0]), dtype=self.dtype, requires_grad=True)
    self.position_y = torch.tensor(get_tensor_item(self.states[1][0]), dtype=self.dtype, requires_grad=True)
    self.velocity = torch.tensor(get_tensor_item(self.states[2][0]), dtype=self.dtype, requires_grad=True)
    self.orientation = torch.tensor(get_tensor_item(self.states[3][0]), dtype=self.dtype, requires_grad=True)
    self.orientation_dot = torch.tensor(get_tensor_item(self.states[4][0]), dtype=self.dtype, requires_grad=True)
    
    self.states = torch.tensor([
      self.position_x,
      self.position_y,
      self.orientation,
      self.velocity,
      self.orientation_dot
    ]).reshape([5, 1])

    self.inputs = torch.tensor([
      torch.tensor(0.),
      torch.tensor(0.),
    ]).reshape([2, 1])

