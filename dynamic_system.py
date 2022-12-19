class DynamicSystem:
  def __init__(self, states, parameters):
    self.states = states
    self.inputs = []
    self.parameters = parameters
    
    return
  
  def state_transition(self, desired_state):
    pass

  def set_parameters(self, parameters):
    pass

  def reset(self):
    pass