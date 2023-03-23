import torch
from torch.autograd import grad
from collections import deque
import sys
sys.path.append(".")

from utils.trajectory_gen import PolynomialTrajectoryGenerator
from utils.waypoint import WayPoint
from utils.commons import hat, vee, quaternion_2_rotation_matrix, rotation_matrix_2_quaternion
from models.quadrotor import Quadrotor
from utils.traj_printer import TrajPrinter

# assume we have known the accurate model and want to estimate the initial state of the state

# Given the waypoints, we can generate the desired trajectories
quadrotor_waypoints = [
    [
      WayPoint(0, 0, 0, 0),
      WayPoint(1, 2, -1, 2),
      WayPoint(3, 6, -5, 4)
    ]
]
g = 9.81
time_interval = 0.05
learning_rate = 0.5
initial_states = torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True).reshape([13, 1]).double()
initial_pid_parameters = torch.tensor([0.5, 0.2, 0.1, 0.1], requires_grad=True).reshape([4, 1]).double()

traj_generator = PolynomialTrajectoryGenerator()
traj_printer = TrajPrinter()
trajectory_wpss = traj_generator.generate_trajectory(quadrotor_waypoints[0], time_interval, 7)

# get quadrotors' desired states
desired_states_in_each_traj = []
last_desired_pose = quaternion_2_rotation_matrix(initial_states[3:7])
last_desired_angle_dot = torch.zeros([3, 1]).double()
last_desired_angle_ddot = torch.zeros([3, 1]).double()
# desired_states_in_each_traj.append(
#   (torch.zeros([3, 1]).double(), torch.zeros([3, 1]).double(), torch.zeros([3, 1]).double(), last_desired_pose, last_desired_angle_dot, last_desired_angle_ddot)
# )
for wp in trajectory_wpss[1:]:
  position_tensor = torch.tensor([wp.position.x, wp.position.y, wp.position.z]).reshape([3,1]).double()
  velocity_tensor = torch.tensor([wp.vel.x, wp.vel.y, wp.vel.z]).reshape([3,1]).double()
  # minus gravity acc if choose upwards as the positive z-axis 
  raw_acc_tensor = torch.tensor([wp.acc.x, wp.acc.y, wp.acc.z]).reshape([3,1]).double()
  acc_tensor = torch.tensor([wp.acc.x, wp.acc.y, wp.acc.z - g]).reshape([3,1]).double()
  
  # assume the yaw angle stays at 0
  b3_desired = (-acc_tensor / torch.norm(acc_tensor)).double()
  b1_yaw_tensor = torch.tensor([1., 0., 0.]).reshape([3, 1]).double()
  b2_desired = torch.cross(b3_desired, b1_yaw_tensor)
  b2_desired = b2_desired / torch.norm(b2_desired)
  b1_desired = torch.cross(b2_desired, b3_desired)
  pose = (torch.concat([b1_desired, b2_desired, b3_desired], dim=1)).double()
  R_err = torch.mm(pose, torch.t(last_desired_pose))
  q_err = rotation_matrix_2_quaternion(R_err)
  q_err = q_err / torch.norm(q_err)
  axis = torch.tensor([q_err[1], q_err[2], q_err[3]]).reshape([3, 1])
  if torch.norm(axis) != 0:
    axis = axis / torch.norm(axis)
  angle = 2 * torch.acos(q_err[0])
  angle_dot = angle / time_interval * axis
  angle_ddot = ((angle_dot - last_desired_angle_dot) / time_interval).double()

  desired_states_in_each_traj.append((position_tensor, velocity_tensor, raw_acc_tensor, pose, angle_dot, angle_ddot))

  last_desired_pose = pose
  last_desired_angle_dot = angle_dot
  last_desired_angle_ddot = angle_ddot


# Use quadrotor model and SE3 controller to follow the trajectory under noise influentce
it_times = 0
while it_times < 20:
  quadrotor = Quadrotor(dt = time_interval)
  states = deque(maxlen=20)
  real_inputs = deque(maxlen=20)
  states.append(torch.zeros([13, 1]))
  state = initial_states
  controller_parameters = initial_pid_parameters
  gradients = deque(maxlen=20)
  for index, desired_state in enumerate(desired_states_in_each_traj):
      quadrotor.set_desired_state(desired_state)
      measurement_noise=torch.randn(13, 1)*(0.01**0.5)
      inputs_tensor = quadrotor.h(state, controller_parameters, measurement_noise)
      system_plant_noise=torch.randn(4, 1)*(0.01**0.5)
      xkp1_states = quadrotor.f(state, inputs_tensor, system_plant_noise)

      state = xkp1_states
      states.append(state)

      gradients.append(
        torch.stack(
          [
            torch.t(grad(outputs=state[0], inputs=controller_parameters, retain_graph=True)[0])[0],
            torch.t(grad(outputs=state[1], inputs=controller_parameters, retain_graph=True)[0])[0],
            torch.t(grad(outputs=state[2], inputs=controller_parameters, retain_graph=True)[0])[0]
          ]
        )
      )

      # collect states every period and calcuate the gradients
      if index % 20 == 0 and index != 0:
        gradient_sum = torch.zeros([4, 1])
        # calculate the loss gradient
        for i in range(0, 20):
          # get the desired_state
          desired_s = desired_states_in_each_traj[index - 20 + i]
          desired_position = desired_s[0]
          real_positions = states[i][0:3]
          gradient_sum += torch.t(torch.matmul(torch.t(real_positions - desired_position), gradients[i]))
        controller_parameters = torch.tensor(controller_parameters - learning_rate * gradient_sum, requires_grad=True)
        print("Parameters: ", controller_parameters)
  it_times += 1

      

print("Printing final trajectory...")
quadrotor_states_in_each_traj = []
desired_positions = []
desired_vels = []
state = initial_states
for desired_state in desired_states_in_each_traj:
  desired_positions.append(torch.tensor([desired_state[0][0], desired_state[0][1], desired_state[0][2]]))
  desired_vels.append(torch.tensor([desired_state[1][0], desired_state[1][1], desired_state[1][2]]))
  quadrotor.set_desired_state(desired_state)
  inputs_tensor = quadrotor.h(state, controller_parameters, torch.zeros([13,1]))
  xkp1_states = quadrotor.f(state, inputs_tensor, torch.zeros([4,1]))
  state = xkp1_states
  quadrotor_states_in_each_traj.append(state)
    

# plot 3d trajectory
positions = []
poses = []
velocities = []
for state in quadrotor_states_in_each_traj:
  positions.append(torch.tensor([state[0], state[1], state[2]]).reshape([3, 1]))
  poses.append(quaternion_2_rotation_matrix(torch.tensor([state[3], state[4], state[5], state[6]])))
  velocities.append(torch.tensor([state[7], state[8], state[9]]).reshape([3, 1]))
traj_printer.plot_3d_quadrotor_using_torch_tensors(positions, poses, velocities, desired_positions, desired_vels, False)
       

