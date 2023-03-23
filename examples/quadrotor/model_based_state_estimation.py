import torch
from torch.autograd import grad
import sys
sys.path.append(".")

from utils.trajectory_gen import PolynomialTrajectoryGenerator
from utils.waypoint import WayPoint
from utils.commons import hat, vee, quaternion_2_rotation_matrix, rotation_matrix_2_quaternion
from models.quadrotor import Quadrotor
from utils.traj_printer import TrajPrinter
from tuners.quadrotors_pid_auto_tuner_using_sensitivity_propagation import PIDAutoTunerUsingSensituvityPropagation

def run_once(dynamic_system, initial_states, desired_states, controller_parameters):
  states = []
  observed_states = []
  states.append(initial_states)
  loss = torch.tensor([0.])
  state = initial_states
  for index, desired_state in enumerate(desired_states):
    dynamic_system.set_desired_state(desired_state)
    measurement_noise=torch.randn(13, 1)*(0.01**0.5)
    observed_states.append(state + measurement_noise)
    inputs_tensor = quadrotor.h(state, controller_parameters, measurement_noise)
    system_plant_noise=torch.randn(4, 1)*(0.01**0.5)
    xkp1_states = quadrotor.f(state, inputs_tensor, system_plant_noise)

    state = xkp1_states
    states.append(state)

    loss += torch.norm(state[0:3] - desired_state[0])

  print("Loss: ", loss)
  return loss

# assume we have known the accurate model and want to estimate the initial state of the state
def get_best_initial_state_using_gradient_descent(initial_state, dynamic_system, controller_parameters, desired_states, observed_states):
  states_window_length = len(observed_states)

  iteration_times = 0

  while iteration_times < 5:
    state = torch.clone(initial_state)
    state.requires_grad_()
    gradients = []
    states = []
    for i in range(0, states_window_length):
      dynamic_system.set_desired_state(desired_states[i])
      input = dynamic_system.h(state, controller_parameters, torch.zeros([13, 1]))
      xkp1_state = dynamic_system.f(state, input, torch.zeros([4, 1]))

      dxp1dxk = torch.stack([
        torch.t(grad(outputs = xkp1_state[0], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[1], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[2], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[3], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[4], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[5], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[6], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[7], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[8], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[9], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[10], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[11], inputs=state, retain_graph=True)[0])[0],
        torch.t(grad(outputs = xkp1_state[12], inputs=state, retain_graph=True)[0])[0],
      ])

      state = torch.clone(xkp1_state)
      state.detach()
      state.requires_grad_()

      gradients.append(dxp1dxk)
      states.append(xkp1_state)

    # print loss
    loss = torch.tensor([0.])
    for i in range(0, states_window_length):
      loss += torch.norm(states[i][0:3] - observed_states[i][0:3])

    print("Loss: ", loss)

    # get the loss gradient
    gradient = torch.eye(13).double()
    gradient_sum = torch.zeros([1, 13])
    for i in range(0, states_window_length):
      desired_position, desired_velocity, desired_acceleration, desired_pose, desired_angular_vel, desired_angular_acc = desired_states[i]
      gradient = torch.mm(gradients[i], gradient)

      desired_state_tensor = torch.concat(
        [desired_position, rotation_matrix_2_quaternion(desired_pose), desired_velocity, desired_angular_vel]
      )
      gradient_sum += 2 * torch.mm(torch.t((states[i] - observed_states[i])), gradient)
    initial_state = initial_state - 0.000002 * torch.t(gradient_sum)
    initial_state[3:7] = initial_state[3:7] / torch.norm(initial_state[3:7])
    iteration_times += 1
    print(initial_state)

    # print the trajectory
    traj_printer.plot_3d_trajectory_only_positions(states, observed_states)
  return initial_state

# Given the waypoints, we can generate the desired trajectories
quadrotor_waypoints = [
    [
      WayPoint(0, 0, 0, 0),
      WayPoint(1, 2, -1, 2),
      WayPoint(3, 6, -5, 4)
    ]
]
g = 9.81

time_interval = 0.01

initial_states = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([13, 1]).double()
controller_parameters = torch.tensor([0.5,  0.1, 0.5, 0.2]).reshape([4, 1]).double()
initial_pid_parameters = controller_parameters

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
# To get the optimized controller parameter, we have to collect enough trajectory and analyze them at the same to train the parameters
quadrotor = Quadrotor(dt = time_interval)

# collect trajectories as samples
number_of_trajectories_to_be_collected = 20
collect_trajectories = []
collect_initial_states_of_optimized_trajectories = []

# while len(collect_trajectories) < number_of_trajectories_to_be_collected:
#   # start to collect trajs
#   states = []
#   observed_states = []
#   states.append(initial_states)
#   state = initial_states
#   for index, desired_state in enumerate(desired_states_in_each_traj):
#     quadrotor.set_desired_state(desired_state)
#     measurement_noise=torch.randn(13, 1)*(0.01**0.5)
#     observed_states.append(state + measurement_noise)
#     inputs_tensor = quadrotor.h(state, controller_parameters, measurement_noise)
#     system_plant_noise=torch.randn(4, 1)*(0.01**0.5)
#     xkp1_states = quadrotor.f(state, inputs_tensor, system_plant_noise)

#     state = xkp1_states
#     states.append(state)
#   collect_trajectories.append(states)
#   optimized_initial_states = get_best_initial_state_using_gradient_descent(initial_states, 
#                                                 quadrotor, 
#                                                 controller_parameters, 
#                                                 desired_states_in_each_traj,
#                                                 observed_states)
#   collect_initial_states_of_optimized_trajectories.append(optimized_initial_states)

# So right now we can infer the whole differentiable trajectory
# After getting the initial state, we can try to tune the parameter based on this trajectory against desired trajectory
iteration_times = 0
learning_rate = 0.001
tuner = PIDAutoTunerUsingSensituvityPropagation(quadrotor)
while iteration_times < 2:
  print("Training interation number: %d " % iteration_times)
  # for each collected trajectory, update the controller parameters once
  # use the average of the controller parameters as the new parameters
  tuned_controller_parameters = []
  for index in range(0, number_of_trajectories_to_be_collected):
    print("Training the trajectory number: %d " % index)
    new_controller_parameters = tuner.train(desired_states_in_each_traj, initial_states, controller_parameters, learning_rate)
    tuned_controller_parameters.append(new_controller_parameters)
  iteration_times += 1

  controller_parameters = sum(tuned_controller_parameters) / len(tuned_controller_parameters)

# follow the trajectories in multiple times with the original and tuned parameters
test_interation_times = 0

loss_under_original_parameters = torch.tensor([0.])
loss_under_tuned_parameters = torch.tensor([0.])

# controller_parameters = torch.tensor([[[0.5003, 0.0996, 0.7173, 0.0100]]]).reshape([4, 1]).double()
while test_interation_times < 40:
  loss_under_original_parameters += run_once(quadrotor, initial_states, desired_states_in_each_traj, initial_pid_parameters)
  loss_under_tuned_parameters += run_once(quadrotor, initial_states, desired_states_in_each_traj, controller_parameters)
  test_interation_times += 1

print("Before Training...")
print("Loss: ", loss_under_original_parameters)
print("After Training...")
print("Loss: ", loss_under_tuned_parameters)


# how to compare the performance of the optimized controller with the original one
# using different noise, check if the optimized controller has less loss (with desired state) against the origin one









