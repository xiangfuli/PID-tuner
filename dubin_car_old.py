import math

# angle computation function
def get_shortest_path_between_angles(original_ori, des_ori):
  e_ori = des_ori - original_ori
  if abs(e_ori) > math.pi:
    if des_ori > original_ori:
      e_ori = - (original_ori + 2 * math.pi - des_ori)
    else:
      e_ori = des_ori + 2 * math.pi - original_ori
  return e_ori


def get_desired_angular_speed(original_ori, des_ori, dt):
  return get_shortest_path_between_angles(original_ori, des_ori) / dt

class DubinCar:
  def __init__(self, m, J, kp, kv, kpsi, kw, px = 0, py = 0, v = 0, orientation = 0, w = 0, w_dot=0):
    self.m = m
    self.J = J
    self.kp = kp
    self.kv = kv
    self.kpsi = kpsi
    self.kw = kw
    
    # state
    self.px = px
    self.py = py
    self.v = v
    self.orientation = orientation
    self.w = w
    self.w_dot = 0

    self.max_w = 4 * math.pi
    self.max_v = 3

  def state_update(self, pos_x_des, pos_y_des, vel_x_des, vel_y_des, acc_x_des, acc_y_des, orientation_desired, w_desired, w_dot_desired, dt):
    # calculate the control variables
    vx = self.v * math.cos(self.orientation)
    vy = self.v * math.sin(self.orientation)
    acc_x = self.kp * (pos_x_des - self.px) + self.kv * (vel_x_des - vx) + acc_x_des
    acc_y = self.kp * (pos_y_des - self.py) + self.kv * (vel_y_des - vy) + acc_y_des
    e_ori = get_shortest_path_between_angles(self.orientation, orientation_desired)
    e_w = w_desired - self.w
    w_dot = self.kpsi * e_ori + self.kw * e_w + w_dot_desired
    
    # update the state
    w_end = self.w + w_dot * dt;
    orientation = self.orientation + self.w * dt;
    acc = acc_x * math.cos(self.orientation) + acc_y * math.sin(self.orientation)
    vel_end = self.v + acc * dt

    px = self.px + self.v * dt * math.cos(self.orientation)
    py = self.py + self.v * dt * math.sin(self.orientation)
    orientation = math.atan2(math.sin(orientation), math.cos(orientation))

    return ([
        px,
        py,
        orientation,
        vel_end,
        w_end,
        acc,
        w_dot
    ])

  def state_update_inplace(self, pos_x_des, pos_y_des, vel_x_des, vel_y_des, acc_x_des, acc_y_des, orientation_desired, w_desired, w_dot_desired, dt):
    self.set_state(self.state_update(
        pos_x_des,
        pos_y_des,
        vel_x_des,
        vel_y_des,
        acc_x_des, 
        acc_y_des, 
        orientation_desired, 
        w_desired, 
        w_dot_desired, 
        dt
    ))
  
  def set_state(self, state):
    self.px = state[0]
    self.py = state[1]
    self.orientation = state[2]
    self.v = max(state[3], -self.max_v) if state[3] < 0 else min(state[3], self.max_v)
    self.w = max(state[4], -self.max_w) if state[4] < 0 else min(state[4], self.max_w)
    self.w_dot = state[6]
  
  def set_parameters(self, parameters):
    self.kp = parameters[0]
    self.kv = parameters[1]
    self.kpsi = parameters[2]
    self.kw = parameters[3]