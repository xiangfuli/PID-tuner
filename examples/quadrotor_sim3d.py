import torch
import sys
sys.path.append(".")

from utils.trajectory_gen import PolynomialTrajectoryGenerator
from utils.waypoint import WayPoint
from utils.traj_printer import TrajPrinter
from utils.commons import hat, vee, quaternion_2_rotation_matrix, rotation_matrix_2_quaternion
from models.quadrotor import Quadrotor
from tuners.quadrotors_pid_tuner_using_infinite_difference import PIDAutoTunerUsingInfiniteDifference
from tuners.quadrotors_pid_auto_tuner_using_sensitivity_propagation import PIDAutoTunerUsingSensituvityPropagation

g = 9.81

# program parameters
time_interval = 0.01
learning_rate = 0.5
initial_states = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([13, 1]).double()
initial_pid_parameters = torch.tensor([1, 10, 0.5, 10]).reshape([4, 1]).double()

quadrotor_waypoints = [
    [
      WayPoint(0, 0, 0, 0),
      WayPoint(1, 2, 1, 2),
      WayPoint(4, 0, 2, 4),
      # WayPoint(2, -2, -3),
      # WayPoint(0, 0, -4),
    ]
]

quadrotor_trajs = []
traj_generator = PolynomialTrajectoryGenerator()

for wps in quadrotor_waypoints:
  # traj_generator.assign_timestamps_in_waypoints(wps, 0.5)
  quadrotor_trajs.append(traj_generator.generate_trajectory(wps, time_interval, 7))

traj_printer = TrajPrinter()

# get quadrotors' desired states
desired_states_in_trajs = []
for wps in quadrotor_trajs:
  desired_states_in_each_traj = []
  last_desired_pose = quaternion_2_rotation_matrix(initial_states[3:7])
  last_desired_angle_dot = torch.zeros([3, 1]).double()
  last_desired_angle_ddot = torch.zeros([3, 1]).double()
  desired_states_in_each_traj.append(
    (torch.zeros([3, 1]).double(), torch.zeros([3, 1]).double(), torch.zeros([3, 1]).double(), last_desired_pose, last_desired_angle_dot, last_desired_angle_ddot)
  )
  for wp in wps[1:]:
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
  desired_states_in_trajs.append(desired_states_in_each_traj)

# for desired_states_in_each_traj in desired_states_in_trajs:
#   positions = []
#   poses = []
#   velocities = []
#   for desired_state in desired_states_in_each_traj:
#     positions.append(desired_state[0])
#     poses.append(desired_state[3])
#     velocities.append(desired_state[1])
#   traj_printer.plot_3d_quadrotor_using_torch_tensors(positions, poses, velocities, False)

quadrotor = Quadrotor(dt = time_interval)
quadrotor_states_in_trajs = []
max_iter_times = 3 / time_interval
for desired_states_in_each_traj in desired_states_in_trajs:
  quadrotor_states_in_each_traj = []
  states = initial_states
  quadrotor_states_in_each_traj.append(states)
  iter_times = 0
  # while iter_times < max_iter_times:
  #   if iter_times >= len(desired_states_in_each_traj):
  #     desired_state = (desired_states_in_each_traj[-1][0], torch.zeros([3, 1]).double(), torch.zeros([3, 1]).double(), quaternion_2_rotation_matrix(initial_states[1]), torch.zeros([3, 1]).double(), torch.zeros([3, 1]).double())
  #   else:
  #     desired_state = desired_states_in_each_traj[iter_times]
  for desired_state in desired_states_in_each_traj:
    quadrotor.set_desired_state(desired_state)
    f, M = quadrotor.h(states, initial_pid_parameters)
    inputs_tensor = torch.concat((f.reshape([1, 1])[0], M.reshape([1, 3])[0]), dim=0)
    xkp1_states = quadrotor.f(states, inputs_tensor)
    states = torch.tensor([xkp1_states[0][0], xkp1_states[0][1], xkp1_states[0][2],
      xkp1_states[1][0], xkp1_states[1][1], xkp1_states[1][2], xkp1_states[1][3],
      xkp1_states[2][0], xkp1_states[2][1], xkp1_states[2][2],
      xkp1_states[3][0], xkp1_states[3][1], xkp1_states[3][2],]).reshape([13, 1])
    quadrotor_states_in_each_traj.append(states)
    iter_times += 1
  quadrotor_states_in_trajs.append(quadrotor_states_in_each_traj)

print("Start plotting...")

# plot 3d trajectory
for index, traj in enumerate(quadrotor_trajs):
  positions = []
  poses = []
  velocities = []
  for state in quadrotor_states_in_trajs[index]:
    positions.append(torch.tensor([state[0], state[1], state[2]]).reshape([3, 1]))
    poses.append(quaternion_2_rotation_matrix(torch.tensor([state[3], state[4], state[5], state[6]])))
    velocities.append(torch.tensor([state[7], state[8], state[9]]).reshape([3, 1]))
  traj_printer.plot_3d_quadrotor_using_torch_tensors(positions, poses, velocities, False)

# tuner = PIDAutoTunerUsingSensituvityPropagation(quadrotor)
tuner = PIDAutoTunerUsingInfiniteDifference(quadrotor)

iteration_times = 0
while iteration_times < 50:
  print("Iteration times: %d.........." % iteration_times)
  initial_pid_parameters = tuner.train(desired_states_in_trajs[0], initial_states, initial_pid_parameters, learning_rate)
  print("Updated parameters: %s" % torch.t(initial_pid_parameters))
  iteration_times += 1

quadrotor_after_optimized = Quadrotor(dt = time_interval)

print("Printing final trajectory...")
quadrotor_after_optimized_trajectory = []
for desired_states_in_each_traj in desired_states_in_trajs:
  quadrotor_states_in_each_traj = []
  states = initial_states
  quadrotor_states_in_each_traj.append(states)
  iter_times = 0
  # while iter_times < max_iter_times:
  #   if iter_times >= len(desired_states_in_each_traj):
  #     desired_state = (desired_states_in_each_traj[-1][0], torch.zeros([3, 1]).double(), torch.zeros([3, 1]).double(), quaternion_2_rotation_matrix(initial_states[1]), torch.zeros([3, 1]).double(), torch.zeros([3, 1]).double())
  #   else:
  #     desired_state = desired_states_in_each_traj[iter_times]
  for desired_state in desired_states_in_each_traj:
    quadrotor.set_desired_state(desired_state)
    f, M = quadrotor.h(states, initial_pid_parameters)
    inputs_tensor = torch.concat((f.reshape([1, 1])[0], M.reshape([1, 3])[0]), dim=0)
    xkp1_states = quadrotor.f(states, inputs_tensor)
    states = torch.tensor([xkp1_states[0][0], xkp1_states[0][1], xkp1_states[0][2],
      xkp1_states[1][0], xkp1_states[1][1], xkp1_states[1][2], xkp1_states[1][3],
      xkp1_states[2][0], xkp1_states[2][1], xkp1_states[2][2],
      xkp1_states[3][0], xkp1_states[3][1], xkp1_states[3][2],]).reshape([13, 1])
    quadrotor_states_in_each_traj.append(states)
    iter_times += 1
  quadrotor_after_optimized_trajectory.append(quadrotor_states_in_each_traj)
# plot 3d trajectory
for index, traj in enumerate(quadrotor_trajs):
  positions = []
  poses = []
  velocities = []
  for state in quadrotor_after_optimized_trajectory[index]:
    positions.append(torch.tensor([state[0], state[1], state[2]]).reshape([3, 1]))
    poses.append(quaternion_2_rotation_matrix(torch.tensor([state[3], state[4], state[5], state[6]])))
    velocities.append(torch.tensor([state[7], state[8], state[9]]).reshape([3, 1]))
  traj_printer.plot_3d_quadrotor_using_torch_tensors(positions, poses, velocities, False)