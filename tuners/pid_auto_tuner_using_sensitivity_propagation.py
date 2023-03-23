import torch
import numpy as np
import sys
sys.path.append(".")
from utils.commons import get_tensor_item
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
    dxdparam_gradients.append(torch.zeros([5, 4]).double())
    states = torch.clone(initial_states)
    states_at_k.append(states)
    for index, desired_state in enumerate(desired_states[1:]):
      self.dynamic_system.set_desired_state(desired_state)
      inputs = self.dynamic_system.h(states, parameters, torch.zeros([5, 1]))
      xkp1_states = self.dynamic_system.f(states, inputs, torch.zeros([5, 1]))

      inputs_var_number = len(inputs)
      states_var_number = len(states)

      states_at_k.append(torch.t(torch.tensor([xkp1_states])))

      dh_gradient = jacobian(self.dynamic_system.h, (states, parameters))
      df_gradient = jacobian(self.dynamic_system.f, (states, torch.tensor([inputs]).reshape([inputs_var_number, 1])))

      states = torch.tensor(xkp1_states).reshape([5, 1])

      dhdxk_tensor = torch.tensor([], dtype=torch.float64)
      dhdparam_tensor = torch.tensor([], dtype=torch.float64)
      for input_index in range(inputs_var_number):
        dhdxk_tensor = torch.cat([dhdxk_tensor, torch.t(dh_gradient[input_index][0][0])], dim=0)
        dhdparam_tensor = torch.cat([dhdparam_tensor, torch.t(dh_gradient[input_index][1][0])], dim=0)
      
      dfdxk_tensor = torch.tensor([], dtype=torch.float64)
      dfduk_tensor = torch.tensor([], dtype=torch.float64)
      for state_index in range(states_var_number):
        dfdxk_tensor = torch.cat([dfdxk_tensor, torch.t(df_gradient[state_index][0][0])], dim=0)
        dfduk_tensor = torch.cat([dfduk_tensor, torch.t(df_gradient[state_index][1][0])], dim=0)

      dxdparam_gradients.append(
        torch.mm(dfdxk_tensor + torch.mm(dfduk_tensor, dhdxk_tensor), dxdparam_gradients[index]) + torch.mm(dfduk_tensor, dhdparam_tensor)
      )

    # accumulate the gradients
    state_chooser = torch.eye(5).double()
    state_chooser[2, 2] = 0.
    state_chooser[3, 3] = 0.
    state_chooser[4, 4] = 0.
    gradient_sum = torch.zeros([4, 1]).double()
    for index in range(0, len(desired_states)):
      px, py, theta, v, w = states_at_k[index]
      x_desired, y_desired, vx_desired, vy_desired, accx_desired, accy_desired, angle_desired, angle_dot_desired, angle_ddot_desired = desired_states[index]

      state_error = states_at_k[index] - torch.tensor([
        x_desired, y_desired, angle_desired, math.sqrt(vx_desired ** 2 + vy_desired ** 2), angle_dot_desired
      ]).reshape(5, 1)
      state_error = state_error.reshape([1, 5]).double()
      state_error = torch.mm(state_error, state_chooser)
      gradient_sum += 2 * torch.mm(state_error, dxdparam_gradients[index]).reshape([4, 1])

      # inputs = inputs_at_k[index - 1]
      # gradient_sum += 0.2 * 2 * inputs[0] * (dudparam_gradients[index - 1][0, :]).reshape([4, 1])
    
    gradient_sum = learning_rate * gradient_sum
    print("Gradient %s" % torch.t(gradient_sum))

    max_num = 0.01
    updated_parameters_tensor = torch.max(torch.ones([4, 1]) * max_num, parameters - gradient_sum)

    loss = 0
    states = torch.clone(initial_states)
    for index, desired_state in enumerate(desired_states[1:]):
      x_desired, y_desired, vx_desired, vy_desired, accx_desired, accy_desired, angle_desired, angle_dot_desired, angle_ddot_desired = desired_state
      self.dynamic_system.set_desired_state(desired_state)
      inputs = self.dynamic_system.h(states, updated_parameters_tensor)
      xkp1_states = self.dynamic_system.f(states, inputs)
      states = xkp1_states

      loss += torch.norm(
        torch.tensor(
          [
            x_desired - states[0],
            y_desired - states[1],
            # get_shortest_path_between_angles(desired_states[index][6], self.dynamic_system.states[2]),
            # desired_states[index][3] -self.dynamic_system.states[3],
          ]
        )
      )
    print("Loss : %s" % (loss/(len(desired_states) - 1)))
    return updated_parameters_tensor
    
