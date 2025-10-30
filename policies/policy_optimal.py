import numpy as np


class Model():

    def __init__(self, instance):
        self.instance = instance

        self.state_value = self.instance.optimal_state_values()

    def choose_action(self, state, rnd_state, is_learning):
        reward_per_action = state.demand + self.instance.discount_rate * np.array(self.state_value[state.time_period+1]) if state.time_period < self.instance.last_period else state.demand 
        action = np.argmax(reward_per_action)

        return action
    
    def transition(self, state, action, rnd_state, is_learning):
        # transition
        state.next_state(action=action, rnd_state=rnd_state)

    def feed_forward(self, last_period):
        pass

    def update_parameter(self):
        pass


