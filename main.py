import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import threading
import time

from car import Car
from waypoint import WayPoint
from trajectory_gen import PolynomialTrajectoryGenerator
from pid_auto_tuner import PIDAutoTuner

pid_controller_initial_parameters = [22, 1.4, 10.8, 3.5]
time_interval = 0.1

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

car = Car(
  1, 1,
  [0, 0, 0, 0, 0],
  pid_controller_initial_parameters,
  dt = time_interval
)
traj_trainset = trajs[0:3]
# compute the desired states
car_desired_states_in_trajs = []
for traj_index, wps in enumerate(traj_trainset):
  car_desired_states = []
  last_desired_state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
  car.reset()
  for wp in wps:
    car_desired_states.append(car.get_desired_state(wp, last_desired_state))
    last_desired_state = car_desired_states[-1:][0]
  car_desired_states_in_trajs.append(car_desired_states)

# plot the trajectory by initial parameters
for traj_index, wps in enumerate(traj_trainset):
  car_states = []
  car.reset()
  for car_desired_state in car_desired_states_in_trajs[traj_index]:
    car.state_transition(car_desired_state)
    car_states.append(car.states)

  fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(40, 5))
    
  xs = []
  ys = []
  velocities = []
  orientations = []
  orientation_dots = []
  dts = []
  norm = plt.Normalize(wps[0].ts, wps[len(wps) - 1].ts)
  for wp_index, wp in enumerate(wps):
    xs.append(car_states[wp_index][0].item())
    ys.append(car_states[wp_index][1].item())
    orientations.append(car_states[wp_index][2].item())

    velocities.append(car_states[wp_index][3].item())
    orientation_dots.append(car_states[wp_index][4].item())
    dts.append(wp_index)
  ax[0].scatter(xs, ys, c=norm(dts), cmap='viridis')
  ax[0].set_title("Trajectory %s: position" % traj_index)
  ax[1].scatter(dts, velocities, c=norm(dts), cmap='viridis')
  ax[1].set_title("Trajectory %s: velocities" % traj_index)
  ax[2].scatter(dts, orientations, c=norm(dts), cmap='viridis')
  ax[2].set_title("Trajectory %s: orientations" % traj_index)
  ax[3].scatter(dts, orientation_dots, c=norm(dts), cmap='viridis')
  ax[3].set_title("Trajectory %s: angular speed" % traj_index)

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
tuner = PIDAutoTuner(car)
while index < 5:
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
    tuner = PIDAutoTuner(new_car)
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
  for car_desired_state in car_desired_states_in_trajs[traj_index]:
    car_after_optimized.state_transition(car_desired_state)
    car_states.append(car_after_optimized.states)

  fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(40, 5))
    
  xs = []
  ys = []
  velocities = []
  orientations = []
  orientation_dots = []
  dts = []
  norm = plt.Normalize(wps[0].ts, wps[len(wps) - 1].ts)
  for wp_index, wp in enumerate(wps):
    xs.append(car_states[wp_index][0].item())
    ys.append(car_states[wp_index][1].item())
    orientations.append(car_states[wp_index][2].item())
    velocities.append(car_states[wp_index][3].item())
    orientation_dots.append(car_states[wp_index][4].item())
    dts.append(wp_index)
  ax[0].scatter(xs, ys, c=norm(dts), cmap='viridis')
  ax[0].set_title("Trajectory %s: position" % traj_index)
  ax[1].scatter(dts, velocities, c=norm(dts), cmap='viridis')
  ax[1].set_title("Trajectory %s: velocities" % traj_index)
  ax[2].scatter(dts, orientations, c=norm(dts), cmap='viridis')
  ax[2].set_title("Trajectory %s: orientations" % traj_index)
  ax[3].scatter(dts, orientation_dots, c=norm(dts), cmap='viridis')
  ax[3].set_title("Trajectory %s: angular speed" % traj_index)
plt.show()