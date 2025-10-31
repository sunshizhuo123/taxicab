import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, neurons):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model():

    def __init__(self, instance, policy_info, state_dim):

        self.instance = instance

        seed = 2808
        torch.manual_seed(seed)
        self.rnd_state_dqn = np.random.RandomState(seed)
  
        # Configurations
        self.use_double_q = False if 'use_double_q' not in policy_info else policy_info['use_double_q']  # Use double approach
        self.use_target_q = True if self.use_double_q or ('use_target_q' in policy_info and policy_info['use_target_q']) else False  # Use target network
        self.use_soft_update = False if 'use_soft_update' not in policy_info else policy_info['use_soft_update']  # Use soft update for target network
        self.tau = 0.001 if 'tau' not in policy_info else policy_info['tau']  # Soft update rate (typically 0.001)

        # Learning rate
        self.learning_rate = 0.001 if 'learning_rate' not in policy_info else policy_info['learning_rate']                              # learning rate: Learning rate is used to balance how well you want the algorithm to learn from new values vs old values. This parameter takes a number between 0 and 1.
        self.learning_rate_harmonic = False if 'learning_rate_harmonic' not in policy_info else policy_info['learning_rate_harmonic']   # harmonic stepsize = considering number of visits
        self.learning_rate_harmonic_a = 10 if 'learning_rate_harmonic_a' not in policy_info else policy_info['learning_rate_harmonic_a']     # default: 10; A parameter for a harmonic stepsize
        
        # Exploration Rate
        self.exploration_rate = 0.7 if 'exploration_rate' not in policy_info else policy_info['exploration_rate']                     # default: 0.0 (epsilon)
        self.exploration_rate_decay = 0.0 if 'exploration_rate_decay' not in policy_info else policy_info['exploration_rate_decay']   # default: 0.995   
        self.exploration_rate_decay_rounds = 0 if 'exploration_rate_decay_rounds' not in policy_info else policy_info['exploration_rate_decay_rounds']
        if self.exploration_rate_decay_rounds > 0:
            # Scale the exploration decay to have no exploration after 'exploration_rate_decay_rounds' rounds
            self.exploration_rate_decay = (1-0.01) / self.exploration_rate_decay_rounds 

        # Batch configurations
        self.batch_size = None if 'batch_size' not in policy_info else policy_info['batch_size']       # default:      
        self.buffer_size = None if 'buffer_size' not in policy_info else policy_info['buffer_size']    # default: 
        self.memory = deque() if self.buffer_size is None else deque(maxlen=self.buffer_size)
        # self.memory = deque() if self.buffer_size is None else deque(maxlen=10)

        # # DeepQ (Neural) Network (DQN) parameters
        action_dim = len(self.instance.nodes)
        self.neurons = int(round((state_dim + action_dim)/2, 0)) if 'neurons' not in policy_info else policy_info['neurons'] # default: average of the number of nodes in the input and output layer

        self.DQN_online = DQN(state_dim, action_dim, self.neurons)
        self.DQN_online_optimizer = optim.Adam(self.DQN_online.parameters(), lr=self.learning_rate)

        if self.use_target_q:
            # create a second DQN
            self.DQN_target = DQN(state_dim, action_dim, self.neurons)

            # Initialize target network with same weights and biases as online network
            self.DQN_target.load_state_dict(self.DQN_online.state_dict())

            # Set target network to evaluation mode
            self.DQN_target.eval()

            # Set update frequency
            self.target_update_frequency = 100 if 'target_update_frequency' not in policy_info else policy_info['target_update_frequency']  # Update target network every n steps
            self.target_update_counter = 0  


    def choose_action(self, state, rnd_state, is_learning):
        # random action
        if is_learning:
            if self.exploration_rate > 0.0:
                if rnd_state.uniform(0, 1) < self.exploration_rate:
                    rnd_action = rnd_state.choice(self.instance.nodes)
                    return rnd_action

        # best action        
        reward_per_action = state.demand + self.instance.discount_rate * self.DQN_online(torch.tensor(state.hash_nn(), dtype=torch.float32)).detach().numpy() if state.time_period < self.instance.last_period else state.demand        
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
        if last_period:
            # If batch size is not set, use the entire memory
            if self.batch_size is None:
                minibatch = self.memory
            else:
                # If not enough samples in memory, skip update
                if len(self.memory) < self.batch_size:
                    return

                # Sample a minibatch of indices using NumPy for reproducibility
                indices = self.rnd_state_dqn.choice(len(self.memory), size=self.batch_size, replace=False)
                minibatch = [self.memory[i] for i in indices]
            
            # Unpack transitions into separate lists
            states, actions, rewards, next_states, dones = zip(*minibatch)

            # Set device (GPU if available, else CPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Convert states and next_states to NumPy arrays first for efficient tensor creation
            states_np = np.array([s.hash_nn() for s in states], dtype=np.float32)
            next_states_np = np.array([s.hash_nn() for s in next_states], dtype=np.float32)
            demand_np = np.array([s.demand for s in next_states], dtype=np.float32)

            # Convert all data to PyTorch tensors and move to device
            states = torch.from_numpy(states_np).to(device)
            next_states = torch.from_numpy(next_states_np).to(device)
            demand_tensor = torch.from_numpy(demand_np).to(device)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

            # --- Compute TD Target using Double DQN ---
            with torch.no_grad():

                if self.use_double_q:
                    # Double DQN     
                    # Hu, M. (2023) The Art of Reinforcement Learning. Apress. Algorithm 1 (p. 164)

                    # Select best actions using demand + discounted Q-values from the ONLINE network
                    best_actions = torch.argmax(demand_tensor + self.instance.discount_rate * self.DQN_online(next_states), dim=1, keepdim=True)

                    # Evaluate those actions using the TARGET network
                    q_evaluation = self.DQN_target(next_states)  # shape: [batch_size, action_dim]
                    selected_q_values = q_evaluation.gather(1, best_actions)

                    # Select corresponding demand values
                    selected_demand = demand_tensor.gather(1, best_actions)

                    # Compute TD target
                    target_q_values = selected_demand + (1 - dones) * self.instance.discount_rate * selected_q_values

                elif self.use_target_q:
                    # Target DQN     
                    # Hu, M. (2023) The Art of Reinforcement Learning. Apress. Algorithm 3 (p. 149)

                    # Select best actions using demand + discounted Q-values from the TARGET network
                    best_actions = torch.argmax(demand_tensor + self.instance.discount_rate * self.DQN_target(next_states), dim=1, keepdim=True)

                    # Evaluate those actions using the TARGET network
                    q_evaluation = self.DQN_target(next_states)  # shape: [batch_size, action_dim]
                    selected_q_values = q_evaluation.gather(1, best_actions)

                    # Select corresponding demand values
                    selected_demand = demand_tensor.gather(1, best_actions)

                    # Compute TD target
                    target_q_values = selected_demand + (1 - dones) * self.instance.discount_rate * selected_q_values

                else:
                    # Standard DQN     
                    # Hu, M. (2023) The Art of Reinforcement Learning. Apress. Algorithm 3 (p. 149) simplified

                    # Select best actions using demand + discounted Q-values from the ONLINE network
                    best_actions = torch.argmax(demand_tensor + self.instance.discount_rate * self.DQN_online(next_states), dim=1, keepdim=True)

                    # Evaluate those actions using the ONLINE network
                    q_evaluation = self.DQN_online(next_states)  # shape: [batch_size, action_dim]
                    selected_q_values = q_evaluation.gather(1, best_actions)

                    # Select corresponding demand values
                    selected_demand = demand_tensor.gather(1, best_actions)

                    # Compute TD target
                    target_q_values = selected_demand + (1 - dones) * self.instance.discount_rate * selected_q_values


            # --- Compute Loss and Optimize ---
            # Current Q-values for taken actions
            q_values = self.DQN_online(states).gather(1, actions)

            # Compute MSE loss between predicted and target Q-values
            loss = nn.MSELoss()(q_values, target_q_values)

            # Backpropagation and optimizer step
            self.DQN_online_optimizer.zero_grad()
            loss.backward()
            self.DQN_online_optimizer.step()

            # Update the target network
            if self.use_target_q:
                if self.use_soft_update:
                    # Soft update: θ_target = τ * θ_online + (1 - τ) * θ_target
                    for target_param, online_param in zip(self.DQN_target.parameters(), self.DQN_online.parameters()):
                        target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
                else:
                    # Hard update: copy online network to target network every N steps
                    self.target_update_counter += 1
                    if self.target_update_counter % self.target_update_frequency == 0:
                        self.DQN_target.load_state_dict(self.DQN_online.state_dict())
                        self.DQN_target.eval() 
                        # print(self.target_update_counter)

    def update_parameter(self):

        # update exploration rate
        if self.exploration_rate_decay > 0 and self.exploration_rate > 0:
            self.exploration_rate = max(self.exploration_rate - self.exploration_rate_decay, 0)


