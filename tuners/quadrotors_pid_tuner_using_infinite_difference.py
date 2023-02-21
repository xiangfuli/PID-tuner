import torch
import numpy as np
from utils.commons import get_tensor_item, rotation_matrix_2_quaternion
from torch.autograd.functional import jacobian
import math

class PIDAutoTunerUsingInfiniteDifference:
  def __init__(self, dynamic_system):
    self.dynamic_system = dynamic_system
    # infinite difference interval
    self.ind = 0.0000001
  
  def train(self, desired_states, initial_states, parameters, learning_rate):
    dxdparam_gradients = []
    states_at_k = []
    dxdparam_gradients.append(torch.zeros([13, 4]).double())
    states = initial_states
    states_at_k.append(states)
    for index, desired_state in enumerate(desired_states):
      self.dynamic_system.set_desired_state(desired_state)
      inputs_tensor = self.dynamic_system.h(states, parameters)
      xkp1_states = self.dynamic_system.f(states, inputs_tensor)

      position, pose, velocity, w = xkp1_states

      # dh_gradient = jacobian(self.dynamic_system.h, (states, parameters))

      dhdx_func = lambda states: self.dynamic_system.h(states, parameters = parameters)
      dhdxk_grad_0 = torch.tensor([])
      dhdxk_grad_1 = torch.tensor([])
      dhdxk_grad_2 = torch.tensor([])
      dhdxk_grad_3 = torch.tensor([])
      for state_index, state in enumerate(states):
        ori_state = torch.clone(state)
        state_m = state - self.ind
        state_p = state + self.ind

        states[state_index] = state_m
        f_state_m_output = dhdx_func(states)

        states[state_index] = state_p
        f_state_p_output = dhdx_func(states)

        dhdxk_grad_0 = torch.concat([dhdxk_grad_0,
          torch.tensor([(f_state_p_output[0] - f_state_m_output[0]) / (2 * self.ind)])])
        dhdxk_grad_1 = torch.concat([dhdxk_grad_1,
          torch.tensor([(f_state_p_output[1] - f_state_m_output[1]) / (2 * self.ind)])])
        dhdxk_grad_2 = torch.concat([dhdxk_grad_2,
          torch.tensor([(f_state_p_output[2] - f_state_m_output[2]) / (2 * self.ind)])])
        dhdxk_grad_3 = torch.concat([dhdxk_grad_3,
          torch.tensor([(f_state_p_output[3] - f_state_m_output[3]) / (2 * self.ind)])])

        states[state_index] = ori_state
        
      dhdxk_tensor = torch.concat(
        [
          dhdxk_grad_0, dhdxk_grad_1, dhdxk_grad_2, dhdxk_grad_3
        ]
      ).reshape([4, 13])

      dhdparam_func = lambda params: self.dynamic_system.h(states, params)
      dhdparam_grad_0 = torch.tensor([])
      dhdparam_grad_1 = torch.tensor([])
      dhdparam_grad_2 = torch.tensor([])
      dhdparam_grad_3 = torch.tensor([])
      for param_index, param in enumerate(parameters):
        ori_param = torch.clone(param)
        param_m = param - self.ind
        param_p = param + self.ind

        parameters[param_index] = param_m
        h_m_output = dhdparam_func(parameters)

        parameters[param_index] = param_p
        h_p_output = dhdparam_func(parameters)

        dhdparam_grad_0 = torch.concat([dhdparam_grad_0,
          torch.tensor([(h_p_output[0] - h_m_output[0]) / (2 * self.ind)])])
        dhdparam_grad_1 = torch.concat([dhdparam_grad_1,
          torch.tensor([(h_p_output[1] - h_m_output[1]) / (2 * self.ind)])])
        dhdparam_grad_2 = torch.concat([dhdparam_grad_2,
          torch.tensor([(h_p_output[2] - h_m_output[2]) / (2 * self.ind)])])
        dhdparam_grad_3 = torch.concat([dhdparam_grad_3,
          torch.tensor([(h_p_output[3] - h_m_output[3]) / (2 * self.ind)])])

        parameters[param_index] = ori_param
        
      dhdparam_tensor = torch.concat(
        [dhdparam_grad_0, dhdparam_grad_1, dhdparam_grad_2, dhdparam_grad_3]
      ).reshape([4, 4])

      dfdxk_func = lambda states: self.dynamic_system.f(states, inputs_tensor)
      dfdxk_grad_0 = torch.tensor([])
      dfdxk_grad_1 = torch.tensor([])
      dfdxk_grad_2 = torch.tensor([])
      dfdxk_grad_3 = torch.tensor([])
      dfdxk_grad_4 = torch.tensor([])
      dfdxk_grad_5 = torch.tensor([])
      dfdxk_grad_6 = torch.tensor([])
      dfdxk_grad_7 = torch.tensor([])
      dfdxk_grad_8 = torch.tensor([])
      dfdxk_grad_9 = torch.tensor([])
      dfdxk_grad_10 = torch.tensor([])
      dfdxk_grad_11 = torch.tensor([])
      dfdxk_grad_12 = torch.tensor([])
      for state_index, state in enumerate(states):
        ori_state = torch.clone(state)
        state_m = state - self.ind
        state_p = state + self.ind

        states[state_index] = state_m
        f_m_output = dfdxk_func(states)

        states[state_index] = state_p
        f_p_output = dfdxk_func(states)

        dfdxk_grad_0 = torch.concat([dfdxk_grad_0,
          torch.tensor([(f_p_output[0][0] - f_m_output[0][0]) / (2 * self.ind)])])
        dfdxk_grad_1 = torch.concat([dfdxk_grad_1,
          torch.tensor([(f_p_output[0][1] - f_m_output[0][1]) / (2 * self.ind)])])
        dfdxk_grad_2 = torch.concat([dfdxk_grad_2,
          torch.tensor([(f_p_output[0][2] - f_m_output[0][2]) / (2 * self.ind)])])
        dfdxk_grad_3 = torch.concat([dfdxk_grad_3,
          torch.tensor([(f_p_output[1][0] - f_m_output[1][0]) / (2 * self.ind)])])
        dfdxk_grad_4 = torch.concat([dfdxk_grad_4,
          torch.tensor([(f_p_output[1][1] - f_m_output[1][1]) / (2 * self.ind)])])
        dfdxk_grad_5 = torch.concat([dfdxk_grad_5,
          torch.tensor([(f_p_output[1][2] - f_m_output[1][2]) / (2 * self.ind)])])
        dfdxk_grad_6 = torch.concat([dfdxk_grad_6,
          torch.tensor([(f_p_output[1][3] - f_m_output[1][3]) / (2 * self.ind)])])
        dfdxk_grad_7 = torch.concat([dfdxk_grad_7,
          torch.tensor([(f_p_output[2][0] - f_m_output[2][0]) / (2 * self.ind)])])
        dfdxk_grad_8 = torch.concat([dfdxk_grad_8,
          torch.tensor([(f_p_output[2][1] - f_m_output[2][1]) / (2 * self.ind)])])
        dfdxk_grad_9 = torch.concat([dfdxk_grad_9,
          torch.tensor([(f_p_output[2][2] - f_m_output[2][2]) / (2 * self.ind)])])
        dfdxk_grad_10 = torch.concat([dfdxk_grad_10,
          torch.tensor([(f_p_output[3][0] - f_m_output[3][0]) / (2 * self.ind)])])
        dfdxk_grad_11 = torch.concat([dfdxk_grad_11,
          torch.tensor([(f_p_output[3][1] - f_m_output[3][1]) / (2 * self.ind)])])
        dfdxk_grad_12 = torch.concat([dfdxk_grad_12,
          torch.tensor([(f_p_output[3][2] - f_m_output[3][2]) / (2 * self.ind)])])

        states[state_index] = ori_state
      
      dfdxk_tensor = torch.concat(
        [
          dfdxk_grad_0, dfdxk_grad_1, dfdxk_grad_2, dfdxk_grad_3, dfdxk_grad_4, dfdxk_grad_5, dfdxk_grad_6, dfdxk_grad_7, dfdxk_grad_8, dfdxk_grad_9,dfdxk_grad_10,dfdxk_grad_11,dfdxk_grad_12
        ]
      ).reshape([13, 13])

      dfduk_func = lambda inputs: self.dynamic_system.f(states, inputs)
      dfduk_grad_0 = torch.tensor([])
      dfduk_grad_1 = torch.tensor([])
      dfduk_grad_2 = torch.tensor([])
      dfduk_grad_3 = torch.tensor([])
      dfduk_grad_4 = torch.tensor([])
      dfduk_grad_5 = torch.tensor([])
      dfduk_grad_6 = torch.tensor([])
      dfduk_grad_7 = torch.tensor([])
      dfduk_grad_8 = torch.tensor([])
      dfduk_grad_9 = torch.tensor([])
      dfduk_grad_10 = torch.tensor([])
      dfduk_grad_11 = torch.tensor([])
      dfduk_grad_12 = torch.tensor([])
      for uk_index, uk in enumerate(inputs_tensor):
        ori_uk = torch.clone(uk)
        uk_m = uk - self.ind
        uk_p = uk + self.ind

        inputs_tensor[uk_index] = uk_m
        f_uk_m_output = dfduk_func(inputs_tensor)

        inputs_tensor[uk_index] = uk_p
        f_uk_p_output = dfduk_func(inputs_tensor)

        dfduk_grad_0 = torch.concat([dfduk_grad_0,
          torch.tensor([(f_uk_p_output[0][0] - f_uk_m_output[0][0]) / (2 * self.ind)])])
        dfduk_grad_1 = torch.concat([dfduk_grad_1,
          torch.tensor([(f_uk_p_output[0][1] - f_uk_m_output[0][1]) / (2 * self.ind)])])
        dfduk_grad_2 = torch.concat([dfduk_grad_2,
          torch.tensor([(f_uk_p_output[0][2] - f_uk_m_output[0][2]) / (2 * self.ind)])])
        dfduk_grad_3 = torch.concat([dfduk_grad_3,
          torch.tensor([(f_uk_p_output[1][0] - f_uk_m_output[1][0]) / (2 * self.ind)])])
        dfduk_grad_4 = torch.concat([dfduk_grad_4,
          torch.tensor([(f_uk_p_output[1][1] - f_uk_m_output[1][1]) / (2 * self.ind)])])
        dfduk_grad_5 = torch.concat([dfduk_grad_5,
          torch.tensor([(f_uk_p_output[1][2] - f_uk_m_output[1][2]) / (2 * self.ind)])])
        dfduk_grad_6 = torch.concat([dfduk_grad_6,
          torch.tensor([(f_uk_p_output[1][3] - f_uk_m_output[1][3]) / (2 * self.ind)])])
        dfduk_grad_7 = torch.concat([dfduk_grad_7,
          torch.tensor([(f_uk_p_output[2][0] - f_uk_m_output[2][0]) / (2 * self.ind)])])
        dfduk_grad_8 = torch.concat([dfduk_grad_8,
          torch.tensor([(f_uk_p_output[2][1] - f_uk_m_output[2][1]) / (2 * self.ind)])])
        dfduk_grad_9 = torch.concat([dfduk_grad_9,
          torch.tensor([(f_uk_p_output[2][2] - f_uk_m_output[2][2]) / (2 * self.ind)])])
        dfduk_grad_10 = torch.concat([dfduk_grad_10,
          torch.tensor([(f_uk_p_output[3][0] - f_uk_m_output[3][0]) / (2 * self.ind)])])
        dfduk_grad_11 = torch.concat([dfduk_grad_11,
          torch.tensor([(f_uk_p_output[3][1] - f_uk_m_output[3][1]) / (2 * self.ind)])])
        dfduk_grad_12 = torch.concat([dfduk_grad_12,
          torch.tensor([(f_uk_p_output[3][2] - f_uk_m_output[3][2]) / (2 * self.ind)])])

        inputs_tensor[uk_index] = ori_uk
      
      dfduk_tensor = torch.concat([
        dfduk_grad_0, dfduk_grad_1, dfduk_grad_2, dfduk_grad_3, dfduk_grad_4, dfduk_grad_5, dfduk_grad_6, dfduk_grad_7, dfduk_grad_8, dfduk_grad_9, dfduk_grad_10, dfduk_grad_11, dfduk_grad_12
      ]).reshape([13, 4])

      states = torch.concat(
        [
          position, pose, velocity, w
        ]
      )
      states_at_k.append(states)

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
          desired_pose_q - pose, # wrong error for now
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
      inputs_tensor = self.dynamic_system.h(states, updated_parameters_tensor)
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