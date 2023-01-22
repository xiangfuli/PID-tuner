import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import threading
import time
import sys
sys.path.append(".")

from models.car import Car
from utils.waypoint import WayPoint
from utils.trajectory_gen import PolynomialTrajectoryGenerator
from tuners.pid_auto_tuner_using_sensitivity_propagation import PIDAutoTunerUsingSensituvityPropagation
from utils.traj_printer import TrajPrinter

import torch

# program parameters
use_circle_traj = False
used_traj_index = 4

# tuner parameters
pid_controller_initial_parameters = torch.tensor([5., 5., 5., 5.]).reshape([4, 1])
system_initial_states = torch.tensor([0., 0., 0., 0., 0.]).reshape([5, 1])
time_interval = 0.1
learning_rate = 0.001

# plot attribute
size = 4
marker='o'

traj_printer = TrajPrinter()
traj_gen = PolynomialTrajectoryGenerator()
if use_circle_traj:
  desired_waypoints = traj_gen.generate_circle_waypoints(1, 1, time_interval)
else:
  waypoints_set = [
    [
      WayPoint(0, 0, 0, 0),
      WayPoint(1, 1, 0, 2),
      WayPoint(2, 0, 0, 4),
      WayPoint(3, -1, 0, 6),
      WayPoint(4, 0, 0, 8),
    ],
    [
      WayPoint(0, 0, 0, 0),
      WayPoint(2, 2, 0, 2),
      WayPoint(4, 0, 0, 4),
      WayPoint(2, -2, 0, 6),
      WayPoint(0, 0, 0, 8),
    ],
    [
      WayPoint(0, 0, 0, 0),
      WayPoint(2, 4, 0, 2),
      WayPoint(4, 0, 0, 4),
      WayPoint(2, -4, 0, 6),
      WayPoint(0, 0, 0, 8),
      WayPoint(-2, 4, 0, 10),
      WayPoint(-4, 0, 0, 12),
      WayPoint(-2, -4, 0, 14),
      WayPoint(0, 0, 0, 16)
    ],
    [
      WayPoint(0, 0, 0, 0),
      WayPoint(2, 2, 0, 2),
      WayPoint(4, 0, 0, 4),
      WayPoint(6, 5, 0, 6),
      WayPoint(8, 0, 0, 8),
    ],
    [
      WayPoint(0, 0, 0, 0),
      WayPoint(4, 2, 0, 2),
      WayPoint(8, 0, 0, 4),
      WayPoint(4, -2, 0, 6),
      WayPoint(0, 0, 0, 8),
    ],
    [
      WayPoint(0, 0, 0, 0),
      WayPoint(2, 8, 0, 2),
      WayPoint(4, 6, 0, 4),
      WayPoint(2, 4, 0, 6),
      WayPoint(0, 6, 0, 8),
      WayPoint(2, 8, 0, 10),
      WayPoint(10, 6, 0, 12),
      WayPoint(12, 4, 0, 14),
      WayPoint(10, 2, 0, 16),
      WayPoint(8, 4, 0, 18),
      WayPoint(12, 6, 0, 20),
      WayPoint(16, 0, 0, 22),
  ]]
  middle_wps = waypoints_set[used_traj_index]
  traj_gen.assign_timestamps_in_waypoints(middle_wps)
  desired_states = traj_gen.generate_trajectory(middle_wps, time_interval, 7)
  desired_waypoints = traj_gen.get_desired_states_in_2d(desired_states, time_interval)
  

car = Car(
  1, 1,
  dt = time_interval
)

tuner = PIDAutoTunerUsingSensituvityPropagation(car)

iteration_times = 0
while iteration_times < 0:
  print("Iteration times: %d.........." % iteration_times)
  pid_controller_initial_parameters = tuner.train(desired_waypoints, system_initial_states, pid_controller_initial_parameters, learning_rate)
  print("Updated parameters: %s" % torch.t(pid_controller_initial_parameters))
  iteration_times += 1

car_after_optimized = Car(
  1, 1,
  dt = time_interval
)
TrajPrinter.print_2d_traj(car_after_optimized, desired_waypoints, system_initial_states, pid_controller_initial_parameters, 1)
plt.show()
