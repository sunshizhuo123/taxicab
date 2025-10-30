import copy
import numpy as np

from collections import deque

class Model():

    def __init__(self, instance, policy_info):

        self.instance = instance
        
        # Memory
        self.memory = deque(maxlen=1)

        # Configurations
        self.use_double_q = False if 'use_double_q' not in policy_info else policy_info['use_double_q']  # Use double approach
        self.use_target_q = True if self.use_double_q or ('use_target_q' in policy_info and policy_info['use_target_q']) else False  # Use target network

        # Q-Table
        self.Q_online = {(t, n): [0.0 for a in self.instance.nodes] for t in self.instance.periods for n in self.instance.nodes}   # {state: [a1, a2, ...]}
        self.Q_online_visits = {(t, n): [0 for a in self.instance.nodes] for t in self.instance.periods for n in self.instance.nodes}  # {state: [a1, a2, ...]}

        if self.use_target_q:
            self.Q_target = {(t, n): [0.0 for a in self.instance.nodes] for t in self.instance.periods for n in self.instance.nodes}   # {state: [a1, a2, ...]}
            # .copy()
            self.Q_target_visits = {(t, n): [0 for a in self.instance.nodes] for t in self.instance.periods for n in self.instance.nodes}  # {state: [a1, a2, ...]}

            # Set update frequency
            self.target_update_frequency = 100 if 'target_update_frequency' not in policy_info else policy_info['target_update_frequency']  # Update target network every n steps
            self.target_update_counter = 0  

        # Learning rate
        self.learning_rate = 0.001 if 'learning_rate' not in policy_info else policy_info['learning_rate']                              # learning rate: Learning rate is used to balance how well you want the algorithm to learn from new values vs old values. This parameter takes a number between 0 and 1.
        self.learning_rate_harmonic = False if 'learning_rate_harmonic' not in policy_info else policy_info['learning_rate_harmonic']   # harmonic stepsize = considering number of visits
        self.learning_rate_harmonic_a = 10 if 'learning_rate_harmonic_a' not in policy_info else policy_info['learning_rate_harmonic_a']     # default: 10; A parameter for a harmonic stepsize

        # Stochastic reward factor (to be ignored)
        self.stochastic_reward_factor = 1 if 'stochastic_reward_factor' not in policy_info else policy_info['stochastic_reward_factor']
        self.stochastic_reward_factor_deacy  = 0.0 if 'stochastic_reward_factor_deacy' not in policy_info else policy_info['stochastic_reward_factor_deacy']
        self.stochastic_reward_factor_decay_rounds = 0 if 'stochastic_reward_factor_decay_rounds' not in policy_info else policy_info['stochastic_reward_factor_decay_rounds']
        if self.stochastic_reward_factor_decay_rounds > 0:
            self.stochastic_reward_factor_deacy = 1 / self.stochastic_reward_factor_decay_rounds 

        # Exploration Rate
        self.exploration_rate = 0.0 if 'exploration_rate' not in policy_info else policy_info['exploration_rate']

    def choose_action(self, state, rnd_state, is_learning):
        # random action
        if is_learning:
            if self.exploration_rate > 0.0:
                if rnd_state.uniform(0, 1) < self.exploration_rate:
                    rnd_action = rnd_state.choice(self.instance.nodes)
                    return rnd_action

        # best action
        if self.use_double_q:
            # take the sum of the two networks
            reward_per_action = 2 * state.demand + self.instance.discount_rate * (np.array(self.Q_online[state.hash()]) + np.array(self.Q_target[state.hash()])) if state.time_period < self.instance.last_period else state.demand
        else:
            reward_per_action = state.demand + self.instance.discount_rate * np.array(self.Q_online[state.hash()]) if state.time_period < self.instance.last_period else state.demand
        
        best_action = np.argmax(reward_per_action)

        return best_action
    
    
    def transition(self, state, action, rnd_state, is_learning):
        if is_learning:
            # remember current state for learning
            current_state = copy.copy(state)

            # reward
            reward = 0  

            # transition
            state.next_state(action=action, rnd_state=rnd_state)

            # last period?
            done = True if state.time_period > self.instance.last_period else False
            
            # add to memory
            self.memory.append((current_state, action, reward, copy.copy(state), done))

        else:
            # transition
            state.next_state(action=action, rnd_state=rnd_state)


    def feed_forward(self, last_period):

        for state, action, reward, next_state, done in self.memory:

            # We're updating the downstream reward (also called "maxQ") for a given (state, action)-pair
            # That value is computed by looking at "what is happening in the next state" = looking at our next state and picking the best (state,action)-pair                # 
            # = this is done by considering next state's demand realization + discounted future reward
            # ~ similar to looking at the post-decision state

            # Compute Temporal Difference (TD) target value
            if done:
                # Terminal state: no future reward, so the TD target is just the next state's max immediate reward 
                td_target = max(next_state.demand)
            else:
                if self.use_double_q:
                    # Double Q-Learning
                    # Hu, M. (2023) The Art of Reinforcement Learning. Apress. Algorithm 8 (p. 97)
                    if np.random.random() < 0.5:
                        # Find best action in Q_online
                        best_action = np.argmax(next_state.demand + self.instance.discount_rate * np.array(self.Q_online[next_state.hash()]))

                        # Compute TD target via Q_target
                        td_target = next_state.demand[best_action] + self.instance.discount_rate * self.Q_target[next_state.hash()][best_action]
                    else:
                        # Find best action in Q_target
                        best_action = np.argmax(next_state.demand + self.instance.discount_rate * np.array(self.Q_target[next_state.hash()]))

                        # Compute TD target via Q_online
                        td_target = next_state.demand[best_action] + self.instance.discount_rate * self.Q_online[next_state.hash()][best_action]

                elif self.use_target_q:
                    # Q-Learning with target network
                    # Hu, M. (2023) The Art of Reinforcement Learning. Apress. Algorithm 3 (p. 149) - adapted for Q-Table approach
                    # Find best action in Q_target
                    best_action = np.argmax(next_state.demand + self.instance.discount_rate * np.array(self.Q_target[next_state.hash()]))

                    # Compute TD target via Q_target
                    td_target = next_state.demand[best_action] + self.instance.discount_rate * self.Q_target[next_state.hash()][best_action]
                else:
                    # Standard Q-Learning
                    # Hu, M. (2023) The Art of Reinforcement Learning. Apress. Algorithm 7 (p. 93)
                    # Find best action in Q_online
                    best_action = np.argmax(next_state.demand + self.instance.discount_rate * self.stochastic_reward_factor * np.array(self.Q_online[next_state.hash()]))

                    # Compute TD target via Q_online
                    td_target = next_state.demand[best_action] + self.instance.discount_rate * self.stochastic_reward_factor * self.Q_online[next_state.hash()][best_action]

            # Update the online network with the td_target value
            stepsize = self.learning_rate_harmonic_a / (self.learning_rate_harmonic_a + self.Q_online_visits[state.hash()][action]) if self.learning_rate_harmonic else self.learning_rate            
            self.Q_online[state.hash()][action] = (1 - stepsize) * self.Q_online[state.hash()][action] + stepsize * (td_target) # Update Q-Table
            self.Q_online_visits[state.hash()][action] += 1     # Update number of visits
        
        # Update the target network
        if self.use_target_q:
            self.target_update_counter += 1
            if self.target_update_counter % self.target_update_frequency == 0:
                self.Q_target = self.Q_online.copy()
                self.Q_target_visits = self.Q_online_visits.copy()


    def update_parameter(self):

        pass

        
