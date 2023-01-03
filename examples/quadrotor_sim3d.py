import torch
import sys
sys.path.append(".")

from utils.trajectory_gen import PolynomialTrajectoryGenerator
from utils.waypoint import WayPoint
from utils.traj_printer import TrajPrinter
from utils.commons import hat, vee
from models.quadrotor import Quadrotor

dt = 0.01
g = 9.81

quadrotor_waypoints = [
    [
      WayPoint(0, 0, 0),
      WayPoint(2, 2, -1),
      WayPoint(4, 0, -2),
      # WayPoint(2, -2, -3),
      # WayPoint(0, 0, -4),
    ]
]

quadrotor_trajs = []
traj_generator = PolynomialTrajectoryGenerator()

for wps in quadrotor_waypoints:
  traj_generator.assign_timestamps_in_waypoints(wps)
  quadrotor_trajs.append(traj_generator.generate_trajectory(wps, dt, 7))

traj_printer = TrajPrinter()
# plot 3d trajectory
# for traj in quadrotor_trajs:
#   traj_printer.plot_3d_trajectory(traj)

# get quadrotors' desired states
desired_states_in_trajs = []
for wps in quadrotor_trajs:
  desired_states_in_each_traj = []
  last_desired_pose = torch.eye(3)
  last_desired_angle_dot = torch.zeros([3, 1])
  last_desired_angle_ddot = torch.zeros([3, 1])
  desired_states_in_each_traj.append(
    (torch.zeros([3, 1]), torch.zeros([3, 1]), torch.zeros([3, 1]), last_desired_pose, last_desired_angle_dot, last_desired_angle_ddot)
  )
  b1 = torch.tensor([1., 0., 0.]).reshape([3, 1])
  for wp in wps[1:]:
    position_tensor = torch.tensor([wp.position.x, wp.position.y, wp.position.z]).reshape([3,1])
    velocity_tensor = torch.tensor([wp.vel.x, wp.vel.y, wp.vel.z]).reshape([3,1])
    acc_tensor = torch.tensor([wp.acc.x, wp.acc.y, wp.acc.z - g]).reshape([3,1]).float()
    # assume the yaw angle stays at 0
    b3_desired = acc_tensor / torch.norm(acc_tensor)
    b1_yaw_tensor = torch.tensor([1., 0., 0.]).reshape([3, 1])
    b2_desired = torch.cross(b1_yaw_tensor, b3_desired)
    b2_desired = b2_desired / torch.norm(b2_desired)
    b1_desired = torch.cross(b2_desired, b3_desired)
    pose = torch.concat([b1_desired, b2_desired, b3_desired], dim=1).T
    pose_error = pose - last_desired_pose
    angle_dot = vee(torch.mm(pose_error / dt, last_desired_pose))
    angle_ddot = (angle_dot - last_desired_angle_dot) / dt

    desired_states_in_each_traj.append((position_tensor, velocity_tensor, acc_tensor, pose, angle_dot, angle_ddot))

    last_desired_pose = pose
    last_desired_angle_dot = angle_dot
    last_desired_angle_ddot = angle_ddot
  desired_states_in_trajs.append(desired_states_in_each_traj)

for desired_states_in_each_traj in desired_states_in_trajs:
  positions = []
  poses = []
  for desired_state in desired_states_in_each_traj:
    positions.append(desired_state[0])
    poses.append(desired_state[3])
  traj_printer.plot_3d_quadrotor_using_torch_tensors(positions, poses)

# quadrotor = Quadrotor(init_params_kx = 100, init_params_kv = 50, init_params_kr = 8, init_params_ko = 5)
# quadrotor_states_in_trajs = []
# for desired_states_in_each_traj in desired_states_in_trajs:
#   quadrotor_states_in_each_traj = []
#   for desired_state in desired_states_in_each_traj:
#     f, M = quadrotor.get_controller_output(desired_state[0], desired_state[1], desired_state[2], desired_state[3], desired_state[4], desired_state[5])
#     quadrotor.state_update_in_place(f, M, dt)
#     quadrotor_states_in_each_traj.append((quadrotor.position, quadrotor.pose))
#   quadrotor_states_in_trajs.append(quadrotor_states_in_each_traj)

# # # plot 3d trajectory
# for index, traj in enumerate(quadrotor_trajs):
#   positions = []
#   poses = []
#   for state in quadrotor_states_in_trajs[index]:
#     positions.append(state[0])
#     poses.append(state[1])
#   traj_printer.plot_3d_quadrotor_using_torch_tensors(positions, poses)

  