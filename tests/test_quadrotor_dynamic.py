import sys
sys.path.append(".")
from models.quadrotor import Quadrotor
from utils.traj_printer import TrajPrinter
from utils.commons import quaternion_2_rotation_matrix

import torch

g = 9.81

quadrotor = Quadrotor(dt=0.1)

states = (
  torch.tensor([0., 0., 0.]).reshape([3, 1]).double(),
  torch.tensor([1., 0., 0., 0.]).reshape([4, 1]).double(),
  torch.tensor([0., 0., 0.]).reshape([3, 1]).double(),
  torch.tensor([0., 0., 0.]).reshape([3, 1]).double()
)

inputs = (torch.tensor([quadrotor.m * g]), torch.tensor([0., 0., 0.]).reshape([3, 1]))

quadrotor_states = []
for i in range(0, 100):
  states = quadrotor.f(states, inputs)
  quadrotor_states.append((states[0], states[1], states[2]))

# plot 3d trajectory
positions = []
poses = []
velocities = []
traj_printer = TrajPrinter()
for state in quadrotor_states:
  positions.append(state[0])
  poses.append(quaternion_2_rotation_matrix(state[1]))
  velocities.append(state[2])
traj_printer.plot_3d_quadrotor_using_torch_tensors(positions, poses, velocities)

  
