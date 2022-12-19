import torch
import numpy as np
class PIDAutoTuner:
  def __init__(self, dynamic_system):
    self.states = dynamic_system.states
    self.parameters = dynamic_system.parameters
    self.dynamic_system = dynamic_system

    self.state_at_k = []
    self.des_state_at_k = []
    self.parameter_gradients_at_k = []
    self.state_loss_at_k = []
    self.state_gradient_at_k = []
    self.input_gradient_at_k = []

  def train(self, desired_states, learning_rate):
    for i in range(1, len(desired_states)):
      px_grads = []
      py_grads = []
      ori_grads = []
      v_grads = []
      self.state_at_k = []
      self.des_state_at_k = []
      self.dynamic_system.reset()
      for index, desired_state in enumerate(desired_states[0:i]):
        self.dynamic_system.state_transition(desired_state)
        self.state_at_k.append(self.dynamic_system.states)
        self.des_state_at_k.append(desired_state)
        
      # position and orientation loss
      px_loss = desired_states[i][0] - self.dynamic_system.states[0]
      px_loss_square = px_loss ** 2
      px_loss_square.backward(retain_graph=True)

      px_grads.append([self.dynamic_system.parameters[0].grad, \
        self.dynamic_system.parameters[1].grad, \
        self.dynamic_system.parameters[2].grad, \
        self.dynamic_system.parameters[3].grad])
      
      # position and orientation loss
      py_loss = desired_states[i][1] - self.dynamic_system.states[1]
      py_loss_square = py_loss ** 2
      py_loss_square.backward(retain_graph=True)

      py_grads.append([self.dynamic_system.parameters[0].grad, \
        self.dynamic_system.parameters[1].grad, \
        self.dynamic_system.parameters[2].grad, \
        self.dynamic_system.parameters[3].grad])
      

      ori_loss = desired_states[i][2] - self.dynamic_system.states[2]
      ori_loss_square = ori_loss ** 2
      ori_loss_square.backward(retain_graph=True)

      ori_grads.append([torch.tensor([0.]), \
        torch.tensor([0.]), \
        self.dynamic_system.parameters[2].grad, \
        self.dynamic_system.parameters[3].grad])

      
      # v_loss = desired_states[i][3] - self.dynamic_system.states[3]
      # v_loss_square = v_loss ** 2
      # v_loss_square.backward(retain_graph=True)

      # v_grads.append([self.dynamic_system.parameters[0].grad, \
      #   self.dynamic_system.parameters[1].grad, \
      #   torch.tensor([0.]), \
      #   torch.tensor([0.])])

      dpxdparam = torch.zeros([4, 1])
      dpydparam = torch.zeros([4, 1])
      doridparam = torch.zeros([4, 1])
      # dvdparam = torch.zeros([4, 1])
      dpxdparam += torch.tensor(px_grads[0]).reshape([4, 1])
      dpydparam += torch.tensor(py_grads[0]).reshape([4, 1])
      doridparam += torch.tensor(ori_grads[0]).reshape([4, 1])
      # dvdparam += torch.tensor(v_grads[0]).reshape([4, 1])
      
      grad = dpxdparam + dpydparam + doridparam # + dvdparam
      grad = grad * learning_rate
      min_num = -100
      self.dynamic_system.set_parameters(
        [
          max(min_num, self.dynamic_system.parameters[0].item() - grad[0].item()),
          max(min_num, self.dynamic_system.parameters[1].item() - grad[1].item()),
          max(min_num, self.dynamic_system.parameters[2].item() - grad[2].item()),
          max(min_num, self.dynamic_system.parameters[3].item() - grad[3].item()),
        ]
      )

    loss = 0
    self.dynamic_system.reset()
    for index, desired_state in enumerate(desired_states[0:i]):
      self.dynamic_system.state_transition(desired_state)
      loss += torch.norm(
        torch.tensor(
          [
            desired_states[index][0] -self.dynamic_system.states[0],
            desired_states[index][1] -self.dynamic_system.states[1],
            desired_states[index][2] -self.dynamic_system.states[2],
            # desired_states[index][3] -self.dynamic_system.states[3],
          ]
        )
      )
    print(loss)