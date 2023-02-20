import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import math

from pylab import figure

class TrajPrinter:
  def __init__(self):
    pass

  @staticmethod
  def plot_2d_desired_waypoints(desired_waypoints):
    traj_index = 1
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(40, 5))
      
    xs = []
    ys = []
    velocities = []
    orientations = []
    orientation_dots = []
    dts = []
    norm = plt.Normalize(0, len(desired_waypoints))
    for index in range(0, len(desired_waypoints)):
      xs.append(desired_waypoints[index][0])
      ys.append(desired_waypoints[index][1])
      orientations.append(desired_waypoints[index][6])
      velocities.append(desired_waypoints[index][3].item())
      orientation_dots.append(desired_waypoints[index][7])
      dts.append(index)
    ax[0].plot(xs, ys)
    ax[0].set_title("Trajectory %s: position" % traj_index)
    ax[1].plot(dts, velocities)
    ax[1].set_title("Trajectory %s: velocities" % traj_index)
    ax[2].plot(dts, orientations)
    ax[2].set_title("Trajectory %s: orientations" % traj_index)
    ax[3].plot(dts, orientation_dots)
    ax[3].set_title("Trajectory %s: angular speed" % traj_index)
  
  @staticmethod
  def print_2d_traj(dynamic_system, desired_states, initial_states, parameters, traj_index):
    car_states = []
    states = torch.clone(initial_states)
    for car_desired_state in desired_states:
      dynamic_system.set_desired_state(car_desired_state)
      inputs = dynamic_system.h(states, parameters)
      xkp1_states = dynamic_system.f(states, inputs)
      car_states.append(xkp1_states)
      states = xkp1_states

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(40, 5))
      
    xs = []
    ys = []
    velocities = []
    orientations = []
    orientation_dots = []
    dts = []
    norm = plt.Normalize(0, len(desired_states))
    for index in range(0, len(desired_states)):
      xs.append(car_states[index][0].item())
      ys.append(car_states[index][1].item())
      orientations.append(car_states[index][2].item())
      velocities.append(car_states[index][3].item())
      orientation_dots.append(car_states[index][4].item())
      dts.append(index)
    ax[0].plot(xs, ys, label="output")
    ax[0].set_title("Trajectory %s: position" % traj_index)
    ax[1].plot(dts, velocities, label="output")
    ax[1].set_title("Trajectory %s: velocities" % traj_index)
    ax[2].plot(dts, orientations, label="output")
    ax[2].set_title("Trajectory %s: orientations" % traj_index)
    ax[3].plot(dts, orientation_dots, label="output")
    ax[3].set_title("Trajectory %s: angular speed" % traj_index)
    xs = []
    ys = []
    velocities = []
    orientations = []
    orientation_dots = []
    for desired_state in desired_states:
      xs.append(desired_state[0])
      ys.append(desired_state[1])
      velocities.append(math.sqrt(desired_state[2] ** 2 + desired_state[3] ** 2))
      orientations.append(desired_state[6])
      orientation_dots.append(desired_state[7])
    ax[0].scatter(xs, ys, c='black', s=4, label="desired")
    ax[2].plot(dts, orientations, c='black', label="desired")
    ax[1].plot(dts, velocities, c='black', label="desired")
    ax[3].plot(dts, orientation_dots, c='black', label="desired")
    plt.legend()

  def plot_3d_trajectory(self, wps):
    xs = []
    ys = []
    zs = []
    acc_x = []
    acc_y = []
    acc_z = []
    dts = []
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.show()
    for wp in wps:
      xs.append(wp.position.x)
      ys.append(wp.position.y)
      zs.append(wp.position.z)
      acc_x.append(wp.acc.x)
      acc_y.append(wp.acc.y)
      acc_z.append(wp.acc.z)
      dts.append(wp.ts)

      ax.plot(xs, ys, zs, c='black')
      plt.pause(0.001)
      
  def plot_3d_quadrotor_using_torch_tensors(self, positions, poses, velocities, pasue_or_not = False):
    xs = []
    ys = []
    zs = []
    vel_xs = []
    vel_ys = []
    vel_zs = []
    fig = plt.figure(figsize=plt.figaspect(2.5))
    ax = fig.add_subplot(3, 1, 1, projection = '3d')
    ax_pos = fig.add_subplot(3, 1, 2)
    ax_vel = fig.add_subplot(3, 1, 3)

    waypoonts_indexes = []

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    
    for index, position in enumerate(positions):
      # position = torch.mm(position_transform_matrix, position)
      waypoonts_indexes.append(index)
      xs.append(position[0].item())
      ys.append(position[1].item())
      zs.append(position[2].item())
      vel_xs.append(velocities[index][0].item())
      vel_ys.append(velocities[index][1].item())
      vel_zs.append(velocities[index][2].item())
      
      if pasue_or_not:
        ax.plot(xs, ys, zs, c='black')
        ax.text(position[0].item(), position[1].item(), position[2].item(),
        "x: %.2f, y: %.2f, z: %.2f" % (position[0].item(), position[1].item(), position[2].item()))
        # plot the quadrotor frame
        # pose = torch.mm(pose_transform_matrix, poses[index])
        pose = poses[index]
        
        axis_length = 1
        x_pos_axis = position + axis_length * pose[:, 0].reshape([3, 1])
        x_neg_axis = position - axis_length * pose[:, 0].reshape([3, 1])
        y_pos_axis = position + axis_length * pose[:, 1].reshape([3, 1])
        y_neg_axis = position - axis_length * pose[:, 1].reshape([3, 1])
        z_pos_axis = position + axis_length * pose[:, 2].reshape([3, 1])

        ax.plot([position[0].item(), x_pos_axis[0].item()], [position[1].item(), x_pos_axis[1].item()], [position[2].item(), x_pos_axis[2].item()], c='red')
        ax.plot([position[0].item(), y_pos_axis[0].item()], [position[1].item(), y_pos_axis[1].item()], [position[2].item(), y_pos_axis[2].item()], c='green')
        ax.plot([position[0].item(), z_pos_axis[0].item()], [position[1].item(), z_pos_axis[1].item()], [position[2].item(), z_pos_axis[2].item()], c='blue')
        ax.invert_xaxis()
        ax.invert_zaxis()
        plt.pause(0.001)
        if index != len(positions)-1:
          ax.cla()
      ax_pos.plot(waypoonts_indexes, xs, c='red')
      ax_pos.plot(waypoonts_indexes, ys, c='green')
      ax_pos.plot(waypoonts_indexes, zs, c='blue')
      ax_vel.plot(waypoonts_indexes, vel_xs, c='red')
      ax_vel.plot(waypoonts_indexes, vel_ys, c='green')
      ax_vel.plot(waypoonts_indexes, vel_zs, c='blue')
    if not pasue_or_not:
      position = positions[-1]
      ax.invert_xaxis()
      ax.invert_zaxis()
      ax.plot(xs, ys, zs, c='black')
      ax.text(position[0].item(), position[1].item(), position[2].item(),
        "x: %.2f, y: %.2f, z: %.2f" % (position[0].item(), position[1].item(), position[2].item()))
      pose = poses[-1]
      axis_length = 1
      x_pos_axis = position + axis_length * pose[:, 0].reshape([3, 1])
      x_neg_axis = position - axis_length * pose[:, 0].reshape([3, 1])
      y_pos_axis = position + axis_length * pose[:, 1].reshape([3, 1])
      y_neg_axis = position - axis_length * pose[:, 1].reshape([3, 1])
      z_pos_axis = position + axis_length * pose[:, 2].reshape([3, 1])
      
      ax.plot([position[0].item(), x_pos_axis[0].item()], [position[1].item(), x_pos_axis[1].item()], [position[2].item(), x_pos_axis[2].item()], c='red')
      ax.plot([position[0].item(), y_pos_axis[0].item()], [position[1].item(), y_pos_axis[1].item()], [position[2].item(), y_pos_axis[2].item()], c='green')
      ax.plot([position[0].item(), z_pos_axis[0].item()], [position[1].item(), z_pos_axis[1].item()], [position[2].item(), z_pos_axis[2].item()], c='blue')
      
      ax_pos.plot(waypoonts_indexes, xs, c='red')
      ax_pos.plot(waypoonts_indexes, ys, c='green')
      ax_pos.plot(waypoonts_indexes, zs, c='blue')
      ax_pos.set_title('Position')
      ax_vel.plot(waypoonts_indexes, vel_xs, c='red')
      ax_vel.plot(waypoonts_indexes, vel_ys, c='green')
      ax_vel.plot(waypoonts_indexes, vel_zs, c='blue')
      ax_vel.set_title('Velocity')

      
    
    plt.show()
  
