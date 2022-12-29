import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

class TrajPrinter:
  def __init__(self):
    pass

  @staticmethod
  def print_2d_traj(dynamic_system, desired_states, traj_index):
    car_states = []
    for car_desired_state in desired_states:
      dynamic_system.state_transition(car_desired_state)
      car_states.append(dynamic_system.states)

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
    ax[0].scatter(xs, ys, c=norm(dts), cmap='viridis', s=4)
    ax[0].set_title("Trajectory %s: position" % traj_index)
    ax[1].plot(dts, velocities)
    ax[1].set_title("Trajectory %s: velocities" % traj_index)
    ax[2].plot(dts, orientations)
    ax[2].set_title("Trajectory %s: orientations" % traj_index)
    ax[3].plot(dts, orientation_dots)
    ax[3].set_title("Trajectory %s: angular speed" % traj_index)

  def plot_3d_trajectory(self, wps):
    xs = []
    ys = []
    zs = []
    acc_x = []
    acc_y = []
    acc_z = []
    dts = []
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
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
      
  def plot_3d_quadrotor_using_torch_tensors(self, positions, poses):
    xs = []
    ys = []
    zs = []
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    for index, position in enumerate(positions):
      xs.append(position[0].item())
      ys.append(position[1].item())
      zs.append(position[2].item())

      ax.plot(xs, ys, zs, c='black')
      # plot the quadrotor frame
      pose = poses[index].T
      axis_length = 1
      x_pos_axis = position + axis_length * pose[:, 0].reshape([3, 1])
      x_neg_axis = position - axis_length * pose[:, 0].reshape([3, 1])
      y_pos_axis = position + axis_length * pose[:, 1].reshape([3, 1])
      y_neg_axis = position - axis_length * pose[:, 1].reshape([3, 1])
      z_pos_axis = position + axis_length * pose[:, 2].reshape([3, 1])

      ax.plot([position[0].item(), x_pos_axis[0].item()], [position[1].item(), x_pos_axis[1].item()], [position[2].item(), x_pos_axis[2].item()], c='red')
      ax.plot([position[0].item(), y_pos_axis[0].item()], [position[1].item(), y_pos_axis[1].item()], [position[2].item(), y_pos_axis[2].item()], c='green')
      ax.plot([position[0].item(), z_pos_axis[0].item()], [position[1].item(), z_pos_axis[1].item()], [position[2].item(), z_pos_axis[2].item()], c='blue')
      plt.pause(0.001)
      if index != len(positions)-1:
        ax.cla()
    plt.show()
  
