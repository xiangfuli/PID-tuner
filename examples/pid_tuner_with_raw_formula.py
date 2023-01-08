import sys
sys.path.append(".")
import matplotlib.pyplot as plt
from utils.trajectory_gen import PolynomialTrajectoryGenerator
from utils.traj_printer import TrajPrinter
from tuners.dubin_car_tuner_using_raw_formula import DubinCarTunerWithRawFormula
from models.car import Car
from utils.waypoint import WayPoint

# program parameters
use_circle_traj = True
used_traj_index = 4

# tuner parameters
pid_controller_initial_parameters = [5, 5, 5, 5]
time_interval = 0.1
learning_rate = 0.5

traj_printer = TrajPrinter()
traj_gen = PolynomialTrajectoryGenerator()
if use_circle_traj:
  desired_waypoints = traj_gen.generate_circle_waypoints(1, 1, time_interval)
else:
  waypoints_set = [
    [
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
  middle_wps = waypoints_set[used_traj_index]
  traj_gen.assign_timestamps_in_waypoints(middle_wps)
  desired_states = traj_gen.generate_trajectory(middle_wps, time_interval, 7)
  desired_waypoints = traj_gen.get_desired_states_in_2d(desired_states, time_interval)
  
car = Car(
  1, 1,
  [0, 0, 0, 0, 0],
  pid_controller_initial_parameters,
  dt = time_interval
)

tuner = DubinCarTunerWithRawFormula(car)

iteration_times = 0
while iteration_times <= 200:
  car.reset()
  tuner.train(desired_waypoints, time_interval, learning_rate)
  iteration_times += 1

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
TrajPrinter.print_2d_traj(car_after_optimized, desired_waypoints, 1)

plt.show()