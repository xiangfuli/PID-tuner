import torch
import numpy as np
from utils.commons import get_tensor_item, rotation_matrix_2_quaternion
from torch.autograd.functional import jacobian
import math

class PIDAutoTunerUsingSensituvityPropagation:
  def __init__(self, dynamic_system):
    self.dynamic_system = dynamic_system
  
  def train(self, desired_states, initial_states, parameters, learning_rate):
    dxdparam_gradients = []
    dudparam_gradients = []
    inputs_at_k = []
    states_at_k = []
    dxdparam_gradients.append(torch.zeros([13, 4]).double())
    states = initial_states
    states_at_k.append(states)
    for index, desired_state in enumerate(desired_states):
      self.dynamic_system.set_desired_state(desired_state)
      inputs = self.dynamic_system.h(states, parameters)
      f, M = inputs
      inputs_tensor = torch.concat((f.reshape([1, 1])[0], M.reshape([1, 3])[0]), dim=0)
      xkp1_states = self.dynamic_system.f(states, inputs_tensor)

      position, pose, velocity, w = xkp1_states

      # dh_gradient = jacobian(self.dynamic_system.h, (states, parameters))

      dhdx_func = lambda states: self.dynamic_system.h(states, parameters = parameters)
      dhdxk_grad = jacobian(dhdx_func, states)

      dhdparam_func = lambda params: self.dynamic_system.h(states, params)
      dhdparam_grad = jacobian(dhdparam_func, parameters)

      dfdxk_func = lambda states: self.dynamic_system.f(states, inputs_tensor)
      dfdxk_grad = jacobian(dfdxk_func, states)

      dfduk_func = lambda inputs: self.dynamic_system.f(states, inputs)
      dfduk_grad = jacobian(dfduk_func, inputs_tensor)

      states = torch.concat(
        [
          position, pose, velocity, w
        ]
      )
      states_at_k.append(states)

      dhdxk_tensor = torch.concat(
        [
          torch.t(dhdxk_grad[0][0][0]),
          torch.t(dhdxk_grad[1][0][0]),
          torch.t(dhdxk_grad[1][1][0]),
          torch.t(dhdxk_grad[1][2][0])
        ], 
        dim=0
      )

      dhdparam_tensor = torch.concat(
        [
          torch.t(dhdparam_grad[0][0][0]),
          torch.t(dhdparam_grad[1][0][0]),
          torch.t(dhdparam_grad[1][1][0]),
          torch.t(dhdparam_grad[1][2][0])
        ],
        dim=0
      )

      dfdxk_tensor = torch.concat(
        [
          torch.t(dfdxk_grad[0][0][0]),
          torch.t(dfdxk_grad[0][1][0]),
          torch.t(dfdxk_grad[0][2][0]),
          torch.t(dfdxk_grad[1][0][0]),
          torch.t(dfdxk_grad[1][1][0]),
          torch.t(dfdxk_grad[1][2][0]),
          torch.t(dfdxk_grad[1][3][0]),
          torch.t(dfdxk_grad[2][0][0]),
          torch.t(dfdxk_grad[2][1][0]),
          torch.t(dfdxk_grad[2][2][0]),
          torch.t(dfdxk_grad[3][0][0]),
          torch.t(dfdxk_grad[3][1][0]),
          torch.t(dfdxk_grad[3][2][0])
        ],
        dim=0
      )

      dfduk_tensor = torch.concat(
        [
          dfduk_grad[0][0],
          dfduk_grad[0][1],
          dfduk_grad[0][2],
          dfduk_grad[1][0],
          dfduk_grad[1][1],
          dfduk_grad[1][2],
          dfduk_grad[1][3],
          dfduk_grad[2][0],
          dfduk_grad[2][1],
          dfduk_grad[2][2],
          dfduk_grad[3][0],
          dfduk_grad[3][1],
          dfduk_grad[3][2],
        ],
        dim=0
      )

      dxdparam_gradients.append(
        torch.mm(dfdxk_tensor + torch.mm(dfduk_tensor, dhdxk_tensor), dxdparam_gradients[index]) + torch.mm(dfduk_tensor, dhdparam_tensor)
      )

    # accumulate the gradients
    state_chooser = torch.zeros([13, 13]).double()
    state_chooser[0, 0] = 1.
    state_chooser[1, 1] = 1.
    state_chooser[2, 2] = 1.
    gradient_sum = torch.zeros([4, 1]).double()
    for index in range(0, len(desired_states)):
      position = states_at_k[index][0:3]
      pose = states_at_k[index][3:7]
      vel = states_at_k[index][7:10]
      angular_vel = states_at_k[index][10:13]
      desired_position, desired_velocity, desired_acceleration, desired_pose, desired_angular_vel, desired_angular_acc = desired_states[index]

      desired_pose_q = rotation_matrix_2_quaternion(desired_pose)

      state_error = -torch.concat(
        [
          desired_position - position,
          desired_pose_q - pose,
          desired_velocity - vel,
          desired_angular_vel - angular_vel
        ],
        dim = 0
      )
      state_error = state_error.reshape([1, 13]).double()
      state_error = torch.mm(state_error, state_chooser)
      gradient_sum += 2 * torch.mm(state_error, dxdparam_gradients[index]).reshape([4, 1])

      # inputs = inputs_at_k[index - 1]
      # gradient_sum += 0.2 * 2 * inputs[0] * (dudparam_gradients[index - 1][0, :]).reshape([4, 1])
    
    gradient_sum = learning_rate * gradient_sum
    print("Gradient %s" % torch.t(gradient_sum))

    max_num = 0.01
    updated_parameters_tensor = torch.max(torch.ones([4, 1]) * max_num, parameters - gradient_sum)

    loss = 0
    states = initial_states
    for index, desired_state in enumerate(desired_states):
      desired_position, desired_velocity, desired_acceleration, desired_pose, desired_angular_vel, desired_angular_acc = desired_states[index]
      self.dynamic_system.set_desired_state(desired_state)
      inputs = self.dynamic_system.h(states, updated_parameters_tensor)
      f, M = inputs
      inputs_tensor = torch.concat((f.reshape([1, 1])[0], M.reshape([1, 3])[0]), dim=0)
      xkp1_states = self.dynamic_system.f(states, inputs_tensor)
      
      position = xkp1_states[0]
      pose = xkp1_states[1]
      vel = xkp1_states[2]
      angular_vel = xkp1_states[3]

      states = torch.concat(
        [
          position, pose, vel, angular_vel
        ]
      )

      loss += torch.norm(
        desired_position - position
      )
    print("Loss : %s" % (loss/len(desired_states)))
    return updated_parameters_tensor