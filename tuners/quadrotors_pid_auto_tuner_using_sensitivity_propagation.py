import torch
import numpy as np
from utils.commons import get_tensor_item, rotation_matrix_2_quaternion
from torch.autograd.functional import jacobian
import math
from collections import deque

class PIDAutoTunerUsingSensituvityPropagation:
  def __init__(self, dynamic_system):
    self.dynamic_system = dynamic_system
  
  def train(self, desired_states, initial_states, parameters, learning_rate):
    dxdparam_gradients = []
    dudparam_gradients = []
    inputs_at_k = []
    states_at_k = []
    states = torch.clone(initial_states)
    states.detach_()
    collect_gradients_num = 0
    
    gradients = []
    gradients.append(torch.zeros([13, 4]).double())

    for index, desired_state in enumerate(desired_states):
      self.dynamic_system.set_desired_state(desired_state)
      measurement_noise = torch.randn(13, 1)*(0.01**0.5)
      inputs_tensor = self.dynamic_system.h(states, parameters, measurement_noise)
      plant_noise = torch.randn(4, 1)*(0.01**0.5)
      xkp1_states = self.dynamic_system.f(states, inputs_tensor, plant_noise)

      position, pose, velocity, w = xkp1_states[0:3], xkp1_states[3:7], xkp1_states[7:10], xkp1_states[10:13]
      # dh_gradient = jacobian(self.dynamic_system.h, (states, parameters))

      dhdx_func = lambda states: self.dynamic_system.h(states, parameters = parameters, measurement_noise = torch.randn(13, 1)*(0.0000000**0.5))
      dhdxk_grad = jacobian(dhdx_func, states)

      dhdparam_func = lambda params: self.dynamic_system.h(states, params, measurement_noise=torch.randn(13, 1)*(0.0000000**0.5))
      dhdparam_grad = jacobian(dhdparam_func, parameters)

      dfdxk_func = lambda states: self.dynamic_system.f(states, inputs_tensor, plant_noise)
      dfdxk_grad = jacobian(dfdxk_func, states)

      dfduk_func = lambda inputs: self.dynamic_system.f(states, inputs, plant_noise)
      dfduk_grad = jacobian(dfduk_func, inputs_tensor)

      states = torch.tensor(xkp1_states)
      states_at_k.append(states + measurement_noise)

      dhdxk_tensor = torch.concat(
        [
          torch.t(dhdxk_grad[0][0]),
          torch.t(dhdxk_grad[1][0]),
          torch.t(dhdxk_grad[2][0]),
          torch.t(dhdxk_grad[3][0])
        ], 
        dim=0
      )

      dhdparam_tensor = torch.concat(
        [
          torch.t(dhdparam_grad[0][0]),
          torch.t(dhdparam_grad[1][0]),
          torch.t(dhdparam_grad[2][0]),
          torch.t(dhdparam_grad[3][0])
        ],
        dim=0
      )

      dfdxk_tensor = torch.concat(
        [
          torch.t(dfdxk_grad[0][0]),
          torch.t(dfdxk_grad[1][0]),
          torch.t(dfdxk_grad[2][0]),
          torch.t(dfdxk_grad[3][0]),
          torch.t(dfdxk_grad[4][0]),
          torch.t(dfdxk_grad[5][0]),
          torch.t(dfdxk_grad[6][0]),
          torch.t(dfdxk_grad[7][0]),
          torch.t(dfdxk_grad[8][0]),
          torch.t(dfdxk_grad[9][0]),
          torch.t(dfdxk_grad[10][0]),
          torch.t(dfdxk_grad[11][0]),
          torch.t(dfdxk_grad[12][0])
        ],
        dim=0
      )

      dfduk_tensor = torch.concat(
        [
          dfduk_grad[0][0].reshape([1, 4]),
          dfduk_grad[1][0].reshape([1, 4]),
          dfduk_grad[2][0].reshape([1, 4]),
          dfduk_grad[3][0].reshape([1, 4]),
          dfduk_grad[4][0].reshape([1, 4]),
          dfduk_grad[5][0].reshape([1, 4]),
          dfduk_grad[6][0].reshape([1, 4]),
          dfduk_grad[7][0].reshape([1, 4]),
          dfduk_grad[8][0].reshape([1, 4]),
          dfduk_grad[9][0].reshape([1, 4]),
          dfduk_grad[10][0].reshape([1, 4]),
          dfduk_grad[11][0].reshape([1, 4]),
          dfduk_grad[12][0].reshape([1, 4]),
        ],
        dim=0
      )

      last_gradient = gradients[-1]
      gradients.append(
        torch.mm(dfdxk_tensor + 0.01 * torch.mm(dfduk_tensor, dhdxk_tensor), last_gradient) +  0.01 * torch.mm(dfduk_tensor, dhdparam_tensor)
      )

    # accumulate the gradients
    state_chooser = torch.zeros([13, 13]).double()
    state_chooser[0, 0] = 1.
    state_chooser[1, 1] = 1.
    state_chooser[2, 2] = 1.
    gradient_sum = torch.zeros([4, 1]).double()
    for des_state_index in range(0, len(desired_states)):
      position = states_at_k[des_state_index][0:3]
      pose = states_at_k[des_state_index][3:7]
      vel = states_at_k[des_state_index][7:10]
      angular_vel = states_at_k[des_state_index][10:13]
      desired_position, desired_velocity, desired_acceleration, desired_pose, desired_angular_vel, desired_angular_acc = desired_states[des_state_index]

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
      gradient_sum += 2 * torch.mm(state_error, gradients[des_state_index]).reshape([4, 1])

        # inputs = inputs_at_k[index - 1]
        # gradient_sum += 0.2 * 2 * inputs[0] * (dudparam_gradients[index - 1][0, :]).reshape([4, 1])
      
    gradient_sum = learning_rate * gradient_sum
    print("Gradient %s" % torch.t(gradient_sum))

    max_num = 0.01
    updated_parameters_tensor = torch.max(torch.ones([4, 1]) * max_num, parameters - gradient_sum)
    parameters = updated_parameters_tensor
    print("Parameters %s" % torch.t(parameters))

    # clean the gradients
    gradients.clear()
    gradients.append(torch.zeros([13, 4]).double())
    collect_gradients_num = 0

    loss = 0
    states = initial_states
    for index, desired_state in enumerate(desired_states):
      desired_position, desired_velocity, desired_acceleration, desired_pose, desired_angular_vel, desired_angular_acc = desired_states[index]
      self.dynamic_system.set_desired_state(desired_state)
      inputs_tensor = self.dynamic_system.h(states, updated_parameters_tensor, torch.zeros([13, 1]))
      xkp1_states = self.dynamic_system.f(states, inputs_tensor, torch.zeros([4, 1]))
      
      position, pose, vel, angular_vel = xkp1_states[0:3], xkp1_states[3:7], xkp1_states[7:10], xkp1_states[10:13]

      states = xkp1_states

      loss += torch.norm(
        desired_position - position
      )
    print("Loss : %s" % (loss/len(desired_states)))
    return updated_parameters_tensor