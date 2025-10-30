import copy
import numpy as np

from collections import deque

class Model():

    def __init__(self, instance, policy_info):

        self.instance = instance
        
        # Memory
        self.memory = deque(maxlen=1)

        # State values
        self.state_value = np.array([[0.0 for n in self.instance.nodes] for t in self.instance.periods]) # [time][node]
        self.state_visit = np.array([[0.0 for n in self.instance.nodes] for t in self.instance.periods]) # [time][node]

        # Learning rate
        self.learning_rate = 0.001 if 'learning_rate' not in policy_info else policy_info['learning_rate']                              # learning rate: Learning rate is used to balance how well you want the algorithm to learn from new values vs old values. This parameter takes a number between 0 and 1.
        self.learning_rate_harmonic = False if 'learning_rate_harmonic' not in policy_info else policy_info['learning_rate_harmonic']   # harmonic stepsize = considering number of visits
        self.learning_rate_harmonic_a = 10 if 'learning_rate_harmonic_a' not in policy_info else policy_info['learning_rate_harmonic_a']     # default: 10; A parameter for a harmonic stepsize

        # Exploration Rate
        self.exploration_rate = 0.0 if 'exploration_rate' not in policy_info else policy_info['exploration_rate']

    def choose_action(self, state, rnd_state, is_learning):
        
        # best action
        reward_per_action = state.demand + self.instance.discount_rate * np.array(self.state_value[state.time_period+1]) if state.time_period < self.instance.last_period else state.demand         
        best_action = np.argmax(reward_per_action)

        # random action
        if is_learning:
            # save to memory (BEST action)
            current_state = copy.copy(state)        
            done = True if (current_state.time_period + 1) > self.instance.last_period else False
            self.memory.append((current_state, best_action, done))

            if self.exploration_rate > 0.0:
                if rnd_state.uniform(0, 1) < self.exploration_rate:
                    rnd_action = rnd_state.choice(self.instance.nodes)
                    return rnd_action

        return best_action
    
    def transition(self, state, action, rnd_state, is_learning):

        # transition
        state.next_state(action=action, rnd_state=rnd_state)

    def feed_forward(self, last_period):

        for state, action, done in self.memory:

            # immediate reward: demand of BEST action (stored in memory)       
            immediate_reward = state.demand[action]

            # future reward: PDS of the BEST action
            future_reward = 0 if done else self.state_value[state.time_period+1][action]

            # stepsize
            stepsize = self.learning_rate_harmonic_a / (self.learning_rate_harmonic_a + self.state_visit[state.time_period][state.vehicle_position]) if self.learning_rate_harmonic else self.learning_rate

            # update post-decision value of CURRENT state (that's why we also include the immediate reward)
            self.state_value[state.time_period][state.vehicle_position] = (1-stepsize) * self.state_value[state.time_period][state.vehicle_position]
            self.state_value[state.time_period][state.vehicle_position] += stepsize * (immediate_reward + self.instance.discount_rate * future_reward)
            self.state_visit[state.time_period][state.vehicle_position] += 1

    def update_parameter(self):

        pass

