from dynamic_system import DynamicSystem
import torch
import math

from commons import get_tensor_item, get_shortest_path_between_angles, get_desired_angular_speed

class Car(DynamicSystem):
  def __init__(self, mass, inertial, initial_state, initial_parameters, dt):
    self.m = mass
    self.I = inertial
    self.dtype = torch.float64
    self.initial_state = torch.tensor(initial_state).double()
    self.initial_parameters = torch.tensor(initial_parameters).double()
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
    self.max_orientation_dot = torch.tensor([torch.pi])

  def state_transition(self, desired_state):
    inputs = self.pid_controller_output(desired_state)
    self.state_update(inputs)
  
  def state_update(self, inputs):
    acceleration, orientation_ddot = inputs

    orientation_cos = torch.cos(self.orientation)
    orientation_sin = torch.sin(self.orientation)
    
    new_position_x_tensor = self.position_x + self.velocity * orientation_cos * self.dt
    new_position_y_tensor = self.position_y + self.velocity * orientation_sin * self.dt
    new_orientation_tensor = self.orientation + self.orientation_dot * self.dt
    new_velocity_tensor = self.velocity + acceleration * self.dt
    new_orientation_dot_tensor = self.orientation_dot + orientation_ddot * self.dt

    self.position_x = new_position_x_tensor
    self.position_y = new_position_y_tensor
    self.velocity = torch.max(torch.tensor([0.]), new_velocity_tensor)
    # self.orientation = torch.atan2(torch.sin(new_orientation_tensor), torch.cos(new_orientation_tensor))
    # self.orientation_dot = torch.max(-self.max_orientation_dot, torch.min(new_orientation_dot_tensor, self.max_orientation_dot))
    # self.velocity = new_velocity_tensor
    self.orientation = new_orientation_tensor
    self.orientation_dot = new_orientation_dot_tensor

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
    orientation_ddot = self.kori * err_angle + self.koridot * (angle_dot_desired - self.orientation_dot) + angle_ddot_desired

    acceleration.retain_grad()
    orientation_ddot.retain_grad()

    self.input_acc = acceleration
    self.input_orientation_ddot = orientation_ddot

    self.inputs = [
      self.input_acc,
      self.input_orientation_ddot
    ]

    return acceleration, orientation_ddot

  def set_parameters(self, parameters):
    self.initial_parameters = parameters

    self.kp = torch.tensor([get_tensor_item(self.initial_parameters[0])], dtype=self.dtype, requires_grad=True)
    self.kv = torch.tensor([get_tensor_item(self.initial_parameters[1])], dtype=self.dtype, requires_grad=True)
    self.kori = torch.tensor([get_tensor_item(self.initial_parameters[2])], dtype=self.dtype, requires_grad=True)
    self.koridot = torch.tensor([get_tensor_item(self.initial_parameters[3])], dtype=self.dtype, requires_grad=True)
    
    self.parameters = [
      self.kp,
      self.kv,
      self.kori,
      self.koridot
    ]
  
  def set_states(self, states):
    self.position_x = torch.tensor(get_tensor_item(states[0]), requires_grad=True)
    self.position_y = torch.tensor(get_tensor_item(states[1]), requires_grad=True)
    self.velocity = torch.tensor(get_tensor_item(states[2]), requires_grad=True)
    self.orientation = torch.tensor(get_tensor_item(states[3]), requires_grad=True)
    self.orientation_dot = torch.tensor(get_tensor_item(states[4]), requires_grad=True)

    self.states = [
      self.position_x,
      self.position_y,
      self.orientation,
      self.velocity,
      self.orientation_dot
    ]
  
  def reset(self):
    self.kp = torch.tensor([get_tensor_item(self.initial_parameters[0])], dtype=self.dtype, requires_grad=True)
    self.kv = torch.tensor([get_tensor_item(self.initial_parameters[1])], dtype=self.dtype, requires_grad=True)
    self.kori = torch.tensor([get_tensor_item(self.initial_parameters[2])], dtype=self.dtype, requires_grad=True)
    self.koridot = torch.tensor([get_tensor_item(self.initial_parameters[3])], dtype=self.dtype, requires_grad=True)

    self.parameters = [
      self.kp,
      self.kv,
      self.kori,
      self.koridot
    ]

    self.position_x = torch.tensor([get_tensor_item(self.initial_state[0])], dtype=self.dtype, requires_grad=True)
    self.position_y = torch.tensor([get_tensor_item(self.initial_state[1])], dtype=self.dtype, requires_grad=True)
    self.velocity = torch.tensor([get_tensor_item(self.initial_state[2])], dtype=self.dtype, requires_grad=True)
    self.orientation = torch.tensor([get_tensor_item(self.initial_state[3])], dtype=self.dtype, requires_grad=True)
    self.orientation_dot = torch.tensor([get_tensor_item(self.initial_state[4])], dtype=self.dtype, requires_grad=True)
    self.states = [
      self.position_x,
      self.position_y,
      self.orientation,
      self.velocity,
      self.orientation_dot
    ]

    self.inputs = [
      torch.tensor(self.input_acc),
      torch.tensor(self.input_orientation_ddot)
    ]
  
  def reinit_states_and_params(self):
    self.kp = torch.tensor([get_tensor_item(self.initial_parameters[0])], dtype=self.dtype, requires_grad=True)
    self.kv = torch.tensor([get_tensor_item(self.initial_parameters[1])], dtype=self.dtype, requires_grad=True)
    self.kori = torch.tensor([get_tensor_item(self.initial_parameters[2])], dtype=self.dtype, requires_grad=True)
    self.koridot = torch.tensor([get_tensor_item(self.initial_parameters[3])], dtype=self.dtype, requires_grad=True)

    self.parameters = [
      self.kp,
      self.kv,
      self.kori,
      self.koridot
    ]

    self.position_x = torch.tensor([get_tensor_item(self.states[0])], dtype=self.dtype, requires_grad=True)
    self.position_y = torch.tensor([get_tensor_item(self.states[1])], dtype=self.dtype, requires_grad=True)
    self.velocity = torch.tensor([get_tensor_item(self.states[2])], dtype=self.dtype, requires_grad=True)
    self.orientation = torch.tensor([get_tensor_item(self.states[3])], dtype=self.dtype, requires_grad=True)
    self.orientation_dot = torch.tensor([get_tensor_item(self.states[4])], dtype=self.dtype, requires_grad=True)
    self.states = [
      self.position_x,
      self.position_y,
      self.orientation,
      self.velocity,
      self.orientation_dot
    ]

    self.inputs = [
      torch.tensor([0]),
      torch.tensor([0]),
    ]

