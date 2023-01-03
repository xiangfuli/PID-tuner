class DynamicSystem:
  def __init__(self, states, parameters):
    self.states = states
    self.inputs = []
    self.parameters = parameters
    
    return
  
  def pid_controller_output(self, desired_state):
    pass

  def state_update(self, inputs):
    pass
  
  def state_transition(self, desired_state):
    pass

  def set_parameters(self, parameters):
    pass

  def set_states(self, states):
    pass

  def reset(self):
    pass

  def reinit_states_and_params(self):
    pass

  