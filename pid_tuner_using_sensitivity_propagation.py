import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import threading
import time

from car import Car
from waypoint import WayPoint
from trajectory_gen import PolynomialTrajectoryGenerator
from pid_auto_tuner_using_sensitivity_propagation import PIDAutoTunerUsingSensituvityPropagation
from traj_printer import TrajPrinter
from commons import get_tensor_item

pid_controller_initial_parameters = [10, 10, 10, 10]
time_interval = 0.1

# plot attribute
size = 4
marker='o'

# trajectory generation
traj_gen = PolynomialTrajectoryGenerator()

waypoints_set = [
      [
    WayPoint(0, 0),
    WayPoint(2, 2),
    WayPoint(4, 0),
    WayPoint(2, -2),
    WayPoint(0, 0),
],
  [
    WayPoint(0, 0),
    WayPoint(1, 1),
    WayPoint(2, 0),
    WayPoint(3, -1),
    WayPoint(4, 0),
],

  [
    WayPoint(0, 0),
    WayPoint(2, 4),
    WayPoint(4, 0),
    WayPoint(2, -4),
    WayPoint(0, 0),
    WayPoint(-2, 4),
    WayPoint(-4, 0),
    WayPoint(-2, -4),
    WayPoint(0, 0)
],
[
    WayPoint(0, 0),
    WayPoint(2, 2),
    WayPoint(4, 0),
    WayPoint(6, 5),
    WayPoint(8, 0),
],
[
    WayPoint(0, 0),
    WayPoint(4, 2),
    WayPoint(8, 0),
    WayPoint(4, -2),
    WayPoint(0, 0),
],
[
    WayPoint(0, 0),
    WayPoint(2, 8),
    WayPoint(4, 6),
    WayPoint(2, 4),
    WayPoint(0, 6),
    WayPoint(2, 8),
    WayPoint(10, 6),
    WayPoint(12, 4),
    WayPoint(10, 2),
    WayPoint(8, 4),
    WayPoint(12, 6),
    WayPoint(16, 0),
]]

trajs = []
for wps in waypoints_set:
  traj_gen.assign_timestamps_in_waypoints(wps)
  trajs.append(traj_gen.generate_trajectory(wps, time_interval, 7))

train_split_index = 1
traj_trainset = trajs[0:train_split_index]
traj_testset = trajs[train_split_index:]

car = Car(
  1, 1,
  [0, 0, 0, 0, 0],
  pid_controller_initial_parameters,
  dt = time_interval
)

# compute the desired states
car_desired_states_in_trajs = []
for traj_index, wps in enumerate(traj_trainset):
  car_desired_states_in_trajs.append(traj_gen.get_desired_states_in_2d(wps, time_interval))

train_times = 0
new_car = Car(
  1, 1,
  [0, 0, 0, 0, 0],
  pid_controller_initial_parameters,
  dt = time_interval
)

while train_times < 48:
  print("Train iteration times: %d" % train_times)
  for traj_index, wps in enumerate(traj_trainset):
    # PID auto tuner
    new_car = Car(
      1, 1,
      [0, 0, 0, 0, 0],
      [
        new_car.parameters[0].item(),
        new_car.parameters[1].item(),
        new_car.parameters[2].item(),
        new_car.parameters[3].item()
      ],
      dt = time_interval
    )
    tuner = PIDAutoTunerUsingSensituvityPropagation(new_car)
    tuner.train(car_desired_states_in_trajs[traj_index + train_split_index - 1], 0.001)
  train_times += 1

car_after_optimized = Car(
  1, 1,
  [0, 0, 0, 0, 0],
  [
    tuner.dynamic_system.parameters[0].item(),
    tuner.dynamic_system.parameters[1].item(),
    tuner.dynamic_system.parameters[2].item(),
    tuner.dynamic_system.parameters[3].item()
  ],
  dt = time_interval
)

for traj_index, wps in enumerate(traj_trainset):
  TrajPrinter.print_2d_traj(car_after_optimized, car_desired_states_in_trajs[traj_index], traj_index)

plt.show()
