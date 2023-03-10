import torch
import numpy as np
from utils.commons import get_tensor_item, get_shortest_path_between_angles
class PIDAutoTunerUsingBackwardPropagation:
  def __init__(self, dynamic_system):
    self.dynamic_system = dynamic_system

    self.state_at_k = []
    self.des_state_at_k = []
    self.parameter_gradients_at_k = []
    self.state_loss_at_k = []
    self.state_gradient_at_k = []
    self.input_gradient_at_k = []

  def train(self, desired_states, initial_states, parameters, learning_rate):
    for i in range(1, len(desired_states)):
      px_grads = []
      py_grads = []
      ori_grads = []
      v_grads = []
      self.state_at_k = []
      self.des_state_at_k = []
      states = torch.clone(initial_states)
      for index, desired_state in enumerate(desired_states[0:i]):
        self.des_state_at_k.append(desired_state)
        self.dynamic_system.set_desired_state(desired_state)
        inputs = self.dynamic_system.h(states, parameters)
        xkp1_states = self.dynamic_system.f(states, inputs)
        states = xkp1_states
        self.state_at_k.append(states)
        
      # position and orientation loss
      px_loss = states[0] - desired_states[i][0]
      px_loss_square = px_loss ** 2
      px_loss_square.backward(retain_graph=True)

      px_grads.append([get_tensor_item(parameters[0].grad), \
        get_tensor_item(parameters[1].grad), \
        get_tensor_item(parameters[2].grad), \
        get_tensor_item(parameters[3].grad)])
      
      # position and orientation loss
      py_loss = states[1] - desired_states[i][1]
      py_loss_square = py_loss ** 2
      py_loss_square.backward(retain_graph=True)

      py_grads.append([get_tensor_item(parameters[0].grad), \
        get_tensor_item(parameters[1].grad), \
        get_tensor_item(parameters[2].grad), \
        get_tensor_item(parameters[3].grad)])
      
      ori_loss = get_shortest_path_between_angles(desired_states[i][6], states[2])
      ori_loss_square = ori_loss ** 2
      ori_loss_square.backward(retain_graph=True)

      ori_grads.append([get_tensor_item(parameters[0].grad), \
        get_tensor_item(parameters[1].grad), \
        get_tensor_item(parameters[2].grad), \
        get_tensor_item(parameters[3].grad)])

      
      # v_loss = desired_states[i][3] - self.dynamic_system.states[3]
      # v_loss_square = v_loss ** 2
      # v_loss_square.backward(retain_graph=True)

      # v_grads.append([get_tensor_item(self.dynamic_system.parameters[0].grad), \
      #   get_tensor_item(self.dynamic_system.parameters[1].grad), \
      #   get_tensor_item(self.dynamic_system.parameters[2].grad), \
      #   get_tensor_item(self.dynamic_system.parameters[1].grad)])

      dpxdparam = torch.zeros([4, 1])
      dpydparam = torch.zeros([4, 1])
      doridparam = torch.zeros([4, 1])
      dvdparam = torch.zeros([4, 1])
      dpxdparam += torch.tensor(px_grads[0]).reshape([4, 1])
      dpydparam += torch.tensor(py_grads[0]).reshape([4, 1])
      # doridparam += torch.tensor(ori_grads[0]).reshape([4, 1])
      # dvdparam += torch.tensor(v_grads[0]).reshape([4, 1])

      # gradient of inputs
      # acc = self.dynamic_system.inputs[0]
      # acc_loss = acc ** 2
      # acc_loss.backward(retain_graph=True)
      # acc_grad = []
      # acc_grad.append(
      #   [
      #     get_tensor_item(parameters[0].grad),
      #     get_tensor_item(parameters[1].grad),
      #     get_tensor_item(parameters[2].grad),
      #     get_tensor_item(parameters[3].grad)
      #   ]
      # )
      # daccdparam = torch.zeros([4, 1])
      # daccdparam += torch.tensor(acc_grad[0]).reshape([4, 1])

      # ori_ddot = self.dynamic_system.inputs[1]
      # ori_ddot_loss = ori_ddot ** 2
      # ori_ddot_loss.backward(retain_graph=True)
      # ori_ddot_grad = []
      # ori_ddot_grad.append(
      #   [
      #     get_tensor_item(self.dynamic_system.parameters[0].grad),
      #     get_tensor_item(self.dynamic_system.parameters[1].grad),
      #     get_tensor_item(self.dynamic_system.parameters[2].grad),
      #     get_tensor_item(self.dynamic_system.parameters[3].grad)
      #   ]
      # )
      # doriddotdparam = torch.zeros([4, 1])
      # doriddotdparam += torch.tensor(ori_ddot_grad[0]).reshape([4, 1])

      # angleddot = self.dynamic_system.inputs[1]
      # angleddot_loss = angleddot ** 2
      # angleddot_loss.backward()
      # dangleddotdparam = torch.tensor(
      #   [
      #     self.dynamic_system.parameters[0].grad,
      #     self.dynamic_system.parameters[1].grad,
      #     self.dynamic_system.parameters[2].grad,
      #     self.dynamic_system.parameters[3].grad
      #   ]
      # )

      grad = dpxdparam + dpydparam #+ 0.2*daccdparam #+ 0.01 * doriddotdparam  # + doridparam 
      grad = grad * learning_rate
      max_num = 0.01
      parameters = torch.max(torch.ones([4, 1]) * max_num, parameters - grad)

    loss = 0
    states = torch.clone(initial_states)
    for index, desired_state in enumerate(desired_states):
      self.dynamic_system.set_desired_state(desired_state)
      inputs = self.dynamic_system.h(states, parameters)
      xkp1_states = self.dynamic_system.f(states, inputs)
      states = xkp1_states

      loss += torch.norm(
        torch.tensor(
          [
            desired_state[0] - states[0],
            desired_state[1] - states[1],
            # get_shortest_path_between_angles(desired_states[index][6], self.dynamic_system.states[2]),
            # desired_states[index][3] -self.dynamic_system.states[3],
          ]
        )
      )
    print("Loss : %s" % (loss/(len(desired_states) - 1)))

    return parameters