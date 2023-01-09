import torch
import math

class DubinCarTunerWithRawFormula:
  def __init__(self, dynamic_system):
    self.dynamic_system = dynamic_system
  
  def train(self, desired_states, initial_states, parameters, dt, learning_rate):
    dxdparam_gradients = []
    states_at_k = []
    dxdparam_gradients.append(torch.zeros([5, 4]).double())
    states = torch.clone(initial_states)
    states_at_k.append(states)
    for index, desired_state in enumerate(desired_states[1:]):
      x_desired, y_desired, vx_desired, vy_desired, accx_desired, accy_desired, angle_desired, angle_dot_desired, angle_ddot_desired = desired_state
      self.dynamic_system.set_desired_state(desired_state)
      inputs = self.dynamic_system.h(states, parameters)
      xkp1_states = self.dynamic_system.f(states, inputs)
      states_at_k.append(torch.t(torch.tensor([xkp1_states])))
      

      px, py, theta, v, w = states
      kp, kv, kori, kw = parameters
      cos = math.cos(theta)
      sin = math.sin(theta)
      states = xkp1_states

      dfdxk = [
        [1, 0, -v * dt * math.sin(theta), math.cos(theta) * dt, 0],
        [0, 1, v * dt * math.cos(theta), math.sin(theta) * dt, 0],
        [0, 0, 1, 0, dt],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
      ]

      dfduk = [
        [0, 0],
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt]
      ]

      dhdxk = [
        [-kp * cos, -kp * sin, 
          kp * ((-sin) * (x_desired - px) + cos * (y_desired - py)) + kv * (-sin * vx_desired - v * 2 * cos * (-sin) + cos * vy_desired - 2 * sin * cos * v) + accx_desired * (-sin) + accy_desired * cos, 
         -kv, 0], 
        [0, 0, -kori, 0, -kw] 
        ]

      dhdparam = [
        [
          cos * (x_desired - px) + sin * (y_desired - py),
          cos * (vx_desired - v * cos) + sin * (vy_desired - v * sin),
          0, 0
        ], 
        [
          0, 0,
          angle_desired - theta,
          angle_dot_desired - w
        ]
      ]

      dfdxk_tensor = torch.tensor(dfdxk).double()
      dfduk_tensor = torch.tensor(dfduk).double()
      dhdxk_tensor = torch.tensor(dhdxk).double()
      dhdparam_tensor = torch.tensor(dhdparam).double()


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
    
    gradient_sum = learning_rate * gradient_sum
    print("Gradient %s" % torch.t(gradient_sum))

    loss = 0
    states = torch.clone(initial_states)
    for index, desired_state in enumerate(desired_states[1:]):
      x_desired, y_desired, vx_desired, vy_desired, accx_desired, accy_desired, angle_desired, angle_dot_desired, angle_ddot_desired = desired_state
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

    max_num = 0.01
    max_tensor = torch.max(torch.ones([4, 1]) * max_num, parameters - gradient_sum)
    return max_tensor