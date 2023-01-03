import torch
import numpy as np
from utils.commons import get_tensor_item
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
      self.dynamic_system.reinit_states_and_params()
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
        for input in self.dynamic_system.inputs:
          input.grad = None
        for param in self.dynamic_system.parameters:
          param.grad = None
        for k_state in k_states:
          k_state.grad = None
        
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
        for k_state in k_states:
          k_state.grad = None
        for parameter in self.dynamic_system.parameters:
          parameter.grad = None
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

      # print(dxkp1duk_all_uk_grads_tensor)
      # print(dukdxk_all_uk_grads_tensor) # 这个好像有点大
      # print(dukdparam_all_params_grads_tensor)
      # print(torch.mm(dxkp1duk_all_uk_grads_tensor, dukdparam_all_params_grads_tensor))
      # print(dxkp1dparam_all_params_grads_tensor)
      # print(self.state_gradient_at_k[index])
      # print(torch.mm(dxkp1duk_all_uk_grads_tensor, dukdxk_all_uk_grads_tensor))
      self.state_gradient_at_k.append(
        torch.mm(dxkp1dxk_all_k_states_grads_tensor + torch.mm(dxkp1duk_all_uk_grads_tensor, dukdxk_all_uk_grads_tensor), self.state_gradient_at_k[index])\
           + torch.mm(dxkp1duk_all_uk_grads_tensor, dukdparam_all_params_grads_tensor)
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

      loss_grad += 2 * torch.mm((state_tensor - desired_state_tensor).T, self.state_gradient_at_k[index][0:2, :])
    
    # loss_grad /= len(desired_states)
    loss_grad = loss_grad * learning_rate
    print(loss_grad)
    min_num = 0.1
    self.dynamic_system.set_parameters(
      [
        torch.tensor(max(min_num, self.dynamic_system.parameters[0].item() - loss_grad[0][0].item())),
        torch.tensor(max(min_num, self.dynamic_system.parameters[1].item() - loss_grad[0][1].item())),
        torch.tensor(max(min_num, self.dynamic_system.parameters[2].item() - loss_grad[0][2].item())),
        torch.tensor(max(min_num, self.dynamic_system.parameters[3].item() - loss_grad[0][3].item())),
      ]
    )
    print(self.dynamic_system.parameters)

    loss = 0
    self.dynamic_system.reset()
    for index, desired_state in enumerate(desired_states):
      self.dynamic_system.state_transition(desired_state)
      loss += torch.norm(
        torch.tensor(
          [
            desired_states[index][0] -self.dynamic_system.states[0],
            desired_states[index][1] -self.dynamic_system.states[1],
            # get_shortest_path_between_angles(desired_states[index][6], self.dynamic_system.states[2]),
            # desired_states[index][3] -self.dynamic_system.states[3],
          ]
        )
      )
    print("Loss: %d" % loss)
    
  # def train(self, desired_states, learning_rate):
  #   self.state_gradient_at_k = []
  #   self.input_gradient_at_k = []
  #   self.state_at_k.append(self.dynamic_system.states)
  #   self.state_gradient_at_k.append(torch.zeros([5, 4]).double())
  #   for index, desired_state in enumerate(desired_states):
  #     self.dynamic_system.reinit_states_and_params()
  #     k_states = self.dynamic_system.states
  #     self.dynamic_system.state_transition(desired_state)
  #     kp1_states = self.dynamic_system.states
  #     dxkp1dxk_all_k_states_grads = []
  #     dxkp1dparam_all_params_grads = []
  #     dxkp1duk_all_uk_grads = []
  #     for kp1_state in kp1_states:
  #       # dxkp1dxk: 5 by 5 matrix
  #       dxkp1dxk_single_k_state_grad = []
  #       dxkp1duk_single_uk_grads = []
  #       dxkp1dparam_single_params_grads = []
  #       kp1_state.backward(retain_graph = True)
  #       for k_state in k_states:
  #         dxkp1dxk_single_k_state_grad.append(
  #           get_tensor_item(k_state.grad)
  #         )
  #       for input in self.dynamic_system.inputs:
  #         dxkp1duk_single_uk_grads.append(
  #           get_tensor_item(input.grad)
  #         )
  #       for param in self.dynamic_system.parameters:
  #         dxkp1dparam_single_params_grads.append(
  #           get_tensor_item(param.grad)
  #         )
  #       dxkp1dparam_all_params_grads.append(dxkp1dparam_single_params_grads)
        
  #     theta_des = desired_state[6]
  #     theta = k_states[2]
  #     v = k_states[3]
  #     dxkp1dxk_all_k_states_grads = torch.tensor(
  #       [
  #         [1., 0., -v * torch.sin(theta), self.dynamic_system.dt * torch.cos(theta), 0.],
  #         [0., 1., -v * torch.cos(theta), self.dynamic_system.dt * torch.sin(theta), 0.],
  #         [0., 0., 1., 0., self.dynamic_system.dt],
  #         [0., 0., 0., 1., 0.],
  #         [0., 0., 0., 0., 1.]
  #       ]
  #     ).double()
  #     dxkp1duk_all_uk_grads = torch.tensor([[0.0000, 0.0000],
  #       [0.0000, 0.0000],
  #       [0.0000, 0.0000],
  #       [0.0100, 0.0000],
  #       [0.0000, 0.0100]]).double()
        

  #     dukdxk_all_uk_grads = []
  #     # dukdxk: 2 by 5 matrix
  #     dukdparam_all_params_grads = []
  #     # dukdparam_all_params_grads : 2 by 4 matrix
  #     for input in self.dynamic_system.inputs:
  #       dukdxk_single_uk_grads = []
  #       dukdparam_single_params_grads = []
  #       input.backward(retain_graph=True)
  #       for k_state in k_states:
  #         dukdxk_single_uk_grads.append(
  #           get_tensor_item(k_state.grad)
  #         )
  #       for parameter in self.dynamic_system.parameters:
  #         dukdparam_single_params_grads.append(
  #           get_tensor_item(parameter.grad)
  #         )
  #       dukdxk_all_uk_grads.append(dukdxk_single_uk_grads)
  #       dukdparam_all_params_grads.append(dukdparam_single_params_grads)
        
  #     dxkp1dxk_all_k_states_grads_tensor = torch.tensor(dxkp1dxk_all_k_states_grads).double()
  #     dxkp1duk_all_uk_grads_tensor = torch.tensor(dxkp1duk_all_uk_grads).double()
  #     dukdxk_all_uk_grads_tensor = torch.tensor(dukdxk_all_uk_grads).double()
  #     dukdparam_all_params_grads_tensor = torch.tensor(dukdparam_all_params_grads).double()
  #     dxkp1dparam_all_params_grads_tensor = torch.tensor(dxkp1dparam_all_params_grads).double()

  #     # print(dxkp1duk_all_uk_grads_tensor)

  #     # print(dxkp1duk_all_uk_grads_tensor)
  #     # print(dukdxk_all_uk_grads_tensor) # 这个好像有点大
  #     # print(dukdparam_all_params_grads_tensor)
  #     # print(torch.mm(dxkp1duk_all_uk_grads_tensor, dukdparam_all_params_grads_tensor))
  #     # print(dxkp1dparam_all_params_grads_tensor)
  #     # print(self.state_gradient_at_k[index])
  #     # print(torch.mm(dxkp1duk_all_uk_grads_tensor, dukdxk_all_uk_grads_tensor))
  #     self.state_gradient_at_k.append(
  #       torch.mm(dxkp1dxk_all_k_states_grads_tensor + torch.mm(dxkp1duk_all_uk_grads_tensor, dukdxk_all_uk_grads_tensor), self.state_gradient_at_k[index])\
  #          + torch.mm(dxkp1duk_all_uk_grads_tensor, dukdparam_all_params_grads_tensor)
  #     )
  #     self.input_gradient_at_k.append(
  #       torch.mm(dukdxk_all_uk_grads_tensor, self.state_gradient_at_k[index]) + dukdparam_all_params_grads_tensor
  #     )

  #     self.input_at_k.append(self.dynamic_system.inputs)
  #     self.state_at_k.append(self.dynamic_system.states)

  #   loss_grad = torch.zeros([1, 4])
  #   for index, desired_state in enumerate(desired_states):
  #     desired_state_tensor = torch.tensor([[desired_state[0]],[desired_state[1]]])
  #     state_tensor = torch.tensor(
  #       [
  #         [self.state_at_k[index][0]],
  #         [self.state_at_k[index][1]]
  #       ]
  #     )
  #     self.state_gradient_at_k[index] = self.state_gradient_at_k[index].double()
  #     # print(desired_state_tensor)
  #     # print(state_tensor)
  #     # print(self.state_gradient_at_k[index][0:2].dtype)
  #     # print((desired_state_tensor - state_tensor))
  #     # print(self.state_gradient_at_k[index][0:2])
  #     #print(2 * torch.mm((desired_state_tensor - state_tensor).T, self.state_gradient_at_k[index][0:2]) / len(desired_states))

  #     loss_grad += 2 * torch.mm((state_tensor - desired_state_tensor).T, self.state_gradient_at_k[index][0:2, :])
    
  #   # loss_grad /= len(desired_states)
  #   loss_grad = loss_grad * learning_rate
  #   print(loss_grad)
  #   min_num = 0.1
  #   self.dynamic_system.set_parameters(
  #     [
  #       torch.tensor(max(min_num, self.dynamic_system.parameters[0].item() - loss_grad[0][0].item())),
  #       torch.tensor(max(min_num, self.dynamic_system.parameters[1].item() - loss_grad[0][1].item())),
  #       torch.tensor(max(min_num, self.dynamic_system.parameters[2].item() - loss_grad[0][2].item())),
  #       torch.tensor(max(min_num, self.dynamic_system.parameters[3].item() - loss_grad[0][3].item())),
  #     ]
  #   )
  #   print(self.dynamic_system.parameters)
    
    
  

      
