import numpy as np


class Model():

    def __init__(self, instance):
        self.instance = instance   

    def choose_action(self, state, rnd_state, is_learning):

        action = rnd_state.choice(self.instance.nodes)        
        
        return action
    
    def transition(self, state, action, rnd_state, is_learning):
        # transition
        state.next_state(action=action, rnd_state=rnd_state)

    def feed_forward(self, last_period):
        pass

    def update_parameter(self):
        pass


