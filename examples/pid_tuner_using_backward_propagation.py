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
from tuners.pid_auto_tuner_using_backward_propagation import PIDAutoTunerUsingBackwardPropagation
from utils.traj_printer import TrajPrinter

pid_controller_initial_parameters = [20, 10, 10, 10]
time_interval = 0.2

# plot attribute
size = 4
marker='o'

# trajectory generation
traj_gen = PolynomialTrajectoryGenerator()

waypoints_set = [[
    WayPoint(0, 0),
    WayPoint(1, 1),
    WayPoint(2, 0),
    WayPoint(3, -1),
    WayPoint(4, 0),
],
[
    WayPoint(0, 0),
    WayPoint(2, 2),
    WayPoint(4, 0),
    WayPoint(2, -2),
    WayPoint(0, 0), 
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
    WayPoint(4, 4),
    WayPoint(8, 0),
    WayPoint(4, -4),
    WayPoint(2, 0),
    WayPoint(4, 2),
    WayPoint(6, 0),
    WayPoint(4, -2),
    WayPoint(2, 0),
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

car = Car(
  1, 1,
  [0, 0, 0, 0, 0],
  pid_controller_initial_parameters,
  dt = time_interval
)

traj_trainset = trajs[0:1]
# compute the desired states
car_desired_states_in_trajs = []
for traj_index, wps in enumerate(traj_trainset):
  car_desired_states_in_trajs.append(traj_gen.get_desired_states_in_2d(wps, time_interval))

# plot the trajectory by initial parameters
for traj_index, wps in enumerate(traj_trainset):
  car.reset()
  TrajPrinter.print_2d_traj(car, car_desired_states_in_trajs[traj_index], traj_index)

plt.show()

class thread(threading.Thread):
    def __init__(self, thread_name, thread_ID, tuner, desired_states, learning_rate):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.thread_ID = thread_ID
        self.tuner = tuner
        self.desired_states = desired_states
        self.learning_rate = learning_rate
        self.finished = False
 
        # helper function to execute the threads
    def run(self):
        self.tuner.train(self.desired_states, self.learning_rate)
        self.finished = True

index = 1
train_thread = None
tuner = PIDAutoTunerUsingBackwardPropagation(car)
while index < 30:
  print("Train iteration times: %d..." % index)
  for traj_index, wps in enumerate(traj_trainset):
    # PID auto tuner
    new_car = Car(
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
    tuner = PIDAutoTunerUsingBackwardPropagation(new_car)
    car_desired_states = car_desired_states_in_trajs[traj_index]
    train_thread = thread(1, "Tuner", tuner, car_desired_states,  0.1)
    train_thread.start()
    
    while not train_thread.finished:
      time.sleep(1)
    print(tuner.dynamic_system.parameters)
    
  index += 1

print(new_car.parameters)

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
  car_states = []
  car_after_optimized.reset()
  TrajPrinter.print_2d_traj(car_after_optimized, car_desired_states_in_trajs[traj_index], traj_index)

plt.show()