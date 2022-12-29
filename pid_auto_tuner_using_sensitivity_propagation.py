import torch
import numpy as np
from commons import get_tensor_item
class PIDAutoTunerUsingSensituvityPropagation:
  def __init__(self, dynamic_system):
    self.states = dynamic_system.states
    self.parameters = dynamic_system.parameters
    self.dynamic_system = dynamic_system

    self.state_at_k = []
    self.input_at_k = []
    self.des_state_at_k = []
    self.parameter_gradients_at_k = []
    self.state_loss_at_k = []
    self.state_gradient_at_k = []
    self.input_gradient_at_k = []

  def train(self, desired_states, learning_rate):
    self.state_gradient_at_k = []
    self.input_gradient_at_k = []
    self.state_at_k.append(self.dynamic_system.states)
    self.state_gradient_at_k.append(torch.zeros([5, 4]))
    for index, desired_state in enumerate(desired_states):
      self.dynamic_system.reset()
      k_states = self.dynamic_system.states
      self.dynamic_system.state_transition(desired_state)
      kp1_states = self.dynamic_system.states
      dxkp1dxk_all_k_states_grads = []
      dxkp1dparam_all_params_grads = []
      dxkp1duk_all_uk_grads = []
      for kp1_state in kp1_states:
        # dxkp1dxk: 5 by 5 matrix
        dxkp1dxk_single_k_state_grad = []
        dxkp1duk_single_uk_grads = []
        dxkp1dparam_single_params_grads = []
        kp1_state.backward(retain_graph = True)
        for k_state in k_states:
          dxkp1dxk_single_k_state_grad.append(
            get_tensor_item(k_state.grad)
          )
        for input in self.dynamic_system.inputs:
          dxkp1duk_single_uk_grads.append(
            get_tensor_item(input.grad)
          )
        for param in self.dynamic_system.parameters:
          dxkp1dparam_single_params_grads.append(
            get_tensor_item(param.grad)
          )
        
        dxkp1dxk_all_k_states_grads.append(dxkp1dxk_single_k_state_grad)
        dxkp1duk_all_uk_grads.append(dxkp1duk_single_uk_grads)
        dxkp1dparam_all_params_grads.append(dxkp1dparam_single_params_grads)
      dukdxk_all_uk_grads = []
      # dukdxk: 2 by 5 matrix
      dukdparam_all_params_grads = []
      # dukdparam_all_params_grads : 2 by 4 matrix
      for input in self.dynamic_system.inputs:
        dukdxk_single_uk_grads = []
        dukdparam_single_params_grads = []
        input.backward(retain_graph=True)
        for k_state in k_states:
          dukdxk_single_uk_grads.append(
            get_tensor_item(k_state.grad)
          )
        for parameter in self.dynamic_system.parameters:
          dukdparam_single_params_grads.append(
            get_tensor_item(parameter.grad)
          )
        dukdxk_all_uk_grads.append(dukdxk_single_uk_grads)
        dukdparam_all_params_grads.append(dukdparam_single_params_grads)
        
      dxkp1dxk_all_k_states_grads_tensor = torch.tensor(dxkp1dxk_all_k_states_grads)
      dxkp1duk_all_uk_grads_tensor = torch.tensor(dxkp1duk_all_uk_grads)
      dukdxk_all_uk_grads_tensor = torch.tensor(dukdxk_all_uk_grads)
      dukdparam_all_params_grads_tensor = torch.tensor(dukdparam_all_params_grads)
      dxkp1dparam_all_params_grads_tensor = torch.tensor(dxkp1dparam_all_params_grads)

      # print(dxkp1duk_all_uk_grads_tensor)
      # print(dukdxk_all_uk_grads_tensor) # 这个好像有点大
      # print(dukdparam_all_params_grads_tensor)
      # print(torch.mm(dxkp1duk_all_uk_grads_tensor, dukdparam_all_params_grads_tensor))
      # print(dxkp1dparam_all_params_grads_tensor)
      # print(self.state_gradient_at_k[index])
      # print(torch.mm(dxkp1duk_all_uk_grads_tensor, dukdxk_all_uk_grads_tensor))
      self.state_gradient_at_k.append(
        dxkp1dparam_all_params_grads_tensor #+ torch.mm(dxkp1duk_all_uk_grads_tensor, dukdparam_all_params_grads_tensor)
      )
      self.input_gradient_at_k.append(
        torch.mm(dukdxk_all_uk_grads_tensor, self.state_gradient_at_k[index]) + dukdparam_all_params_grads_tensor
      )

      self.input_at_k.append(self.dynamic_system.inputs)
      self.state_at_k.append(self.dynamic_system.states)

    loss_grad = torch.zeros([1, 4])
    for index, desired_state in enumerate(desired_states):
      desired_state_tensor = torch.tensor([[desired_state[0]],[desired_state[1]]])
      state_tensor = torch.tensor(
        [
          [self.state_at_k[index][0]],
          [self.state_at_k[index][1]]
        ]
      )
      self.state_gradient_at_k[index] = self.state_gradient_at_k[index].double()
      # print(desired_state_tensor)
      # print(state_tensor)
      # print(self.state_gradient_at_k[index][0:2].dtype)
      # print((desired_state_tensor - state_tensor))
      # print(self.state_gradient_at_k[index][0:2])
      #print(2 * torch.mm((desired_state_tensor - state_tensor).T, self.state_gradient_at_k[index][0:2]) / len(desired_states))

      loss_grad += 2 * torch.mm((desired_state_tensor - state_tensor).T, self.state_gradient_at_k[index][0:2])
    
    # loss_grad /= len(desired_states)
    loss_grad = loss_grad * learning_rate
    print(loss_grad)
    min_num = 0.1
    self.dynamic_system.set_parameters(
      [
        max(min_num, self.dynamic_system.parameters[0].item() - loss_grad[0][0].item()),
        max(min_num, self.dynamic_system.parameters[1].item() - loss_grad[0][1].item()),
        max(min_num, self.dynamic_system.parameters[2].item() - loss_grad[0][2].item()),
        max(min_num, self.dynamic_system.parameters[3].item() - loss_grad[0][3].item()),
      ]
    )
    self.dynamic_system.reset()
    print(self.dynamic_system.parameters)
    
    


      
