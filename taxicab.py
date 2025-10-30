
import policy_greedy, policy_random, policy_optimal, policy_vfa, policy_Q_table, policy_DQN

from PIL import Image

import cProfile
import io
import numpy as np
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import pstats

import statistics
import time

profiling = False

if profiling:
    profiler = cProfile.Profile()
    profiler.enable()

def generate_excel_letters(n):
    letters = []
    i = 1
    while len(letters) < n:
        s = ''
        j = i
        while j > 0:
            j -= 1
            s = chr(65 + (j % 26)) + s
            j //= 26
        letters.append(s)
        i += 1
    return letters


class Instance:

    def __init__(self, nodes, periods, discount_rate, rnd_seed):
        self.nodes = [n for n in range(nodes)]                # List of nodes as index, starting at 0: 0, 1, 2, ...
        self.nodes_letter = generate_excel_letters(n=nodes)     # List of nodes: A, B, ..., AA, AB, AC, ...
        self.periods = [t for t in range(periods)]            # list of time periods: 1, 2, 3, ... 
        self.last_period = self.periods[-1]
        self.discount_rate = discount_rate                      # discount rate: The discount is used to balance immediate and future reward. This value takes a number between 0 and 1.

        rnd_state_instance = np.random.RandomState(rnd_seed)

        # used in create_demand() in State
        self.demand_and_probabilities = {}
        for i in self.nodes:            
            for t in self.periods:
                p = np.round(rnd_state_instance.random(size=len(self.nodes)), 6)
                d = rnd_state_instance.randint(0, 101, size=len(self.nodes))
                self.demand_and_probabilities[(i, t)] = dict(p=p, d=d)

        # asign x-y coordinates
        self.xy_coordinates = {n: {'x': rnd_state_instance.randint(0, 1000), 'y': rnd_state_instance.randint(0, 1000)} for n in self.nodes}

        # print(self.demand)
        self.name = 'N' + str(nodes) + 'xT' + str(periods)


    def optimal_state_values(self):
        """
        This function computes the value of each state using backward dynamic programming,
        i.e., it applies Bellman's equation.

        :return: n/a
        """

        # create a dict storing the "actual" expected value of all states
        states = np.array([[0.0 for n in self.nodes] for t in self.periods]) # [time][node]

        # start in the last period
        for t in reversed(self.periods):
            # consider all nodes
            for source in self.nodes:
                # D is a list of lists containing the reward and probability
                D = []

                # loop through all possible moves (aka target nodes)
                for target in self.nodes:
                    demand = self.demand_and_probabilities[(source, t)]['d'][target]
                    prob = self.demand_and_probabilities[(source, t)]['p'][target]

                    if t == self.last_period:
                        # only immediate reward in the last period                      
                        D.append([demand, prob])
                    else:
                        # immediate and downstream reward
                        D.append([demand + (self.discount_rate * states[t+1][target]), 
                                  prob])
                        
                        D.append([(self.discount_rate * states[t+1][target]), 
                                  1])

                # compute value of optimal policy (see VFA tutorial paper, published in 4OR, for an explanation)
                D.sort(key=lambda l: l[0], reverse=True)

                rolling_probability = 0
                # states[t][source] = 0.0

                for d in D:
                    states[t][source] += d[0] * (1 - rolling_probability) * d[1]
                    # print(d[0] * (1 - rolling_probability) * d[1], states[t][source])
                    rolling_probability += (1 - rolling_probability) * d[1]

                    if rolling_probability >= 1:
                        break
        
        return states


class Taxicab:

    def __init__(self, instance, policy_info):
        self.instance = instance

        self.policy_name = policy_info['policy']
        self.name = policy_info['name']

        if self.policy_name == 'Greedy':
            self.policy = policy_greedy.Model(instance=self.instance)
        elif self.policy_name == 'Random':
            self.policy = policy_random.Model(instance=self.instance)
        elif self.policy_name == 'Optimal':
            self.policy = policy_optimal.Model(instance=self.instance)
        elif self.policy_name == 'VFA':
            self.policy = policy_vfa.Model(instance=self.instance, policy_info=policy_info)        
        elif self.policy_name == 'Q-Table':
            self.policy = policy_Q_table.Model(instance=self.instance, policy_info=policy_info)
        elif self.policy_name == 'DQN':
            self.policy = policy_DQN.Model(instance=self.instance, 
                                           policy_info=policy_info, 
                                           state_dim=len(State(instance=self.instance, rnd_state=None).hash_nn()))

        # Current state
        self.S = None

    def learn(self, rounds, test_interval=50, test_rounds=250):
        # Set random state
        rnd_state_learn = np.random.RandomState(2506)

        test_history = {'round': [], 'reward': []}
        if test_interval > 0:
            test_history['round'].append(0)
            test_history['reward'].append(statistics.mean(self.test(rounds=test_rounds)))
        
        computation_time_start = time.time()    # Record the start time

        for r in range(1, rounds + 1):
            # reset
            self.S = State(instance=self.instance, rnd_state=rnd_state_learn)
            

            while self.S.time_period <= self.instance.last_period:
                # choose action
                action = self.policy.choose_action(state=self.S, rnd_state=rnd_state_learn, is_learning=True)

                # transition
                self.policy.transition(state=self.S, 
                                       action=action, 
                                       rnd_state=rnd_state_learn,
                                       is_learning=True)             

                # feed forward (learn)
                self.policy.feed_forward(last_period=True if self.S.time_period == self.instance.last_period else False)

            # update parameters, e.g., exploration rate, decay, etc.
            self.policy.update_parameter()

            # test current policy
            if test_interval > 0 and r % test_interval == 0:
                test_history['round'].append(r)
                test_history['reward'].append(statistics.mean(self.test(rounds=test_rounds)))

            if r % 1000 == 0:                
                print('Learn', self.policy_name, 'in round', r, 'of', rounds, ', avg. reward=', round(statistics.mean(self.test(rounds=1000)), 4), time.strftime("%H:%M:%S", time.localtime()), f" ({int((t:=time.time()-computation_time_start)//60)} min {int(t%60)} sec)")
                computation_time_start = time.time()    # Reset the start time

        self.save_policy()
        
        return test_history
                 
    def test(self, rounds, fig_map=False):
        # Set random state
        rnd_state_test_transition = np.random.RandomState(1988)
        rnd_state_test_action = np.random.RandomState(1988)

        rewards = []

        for r in range(1, rounds + 1):

            if fig_map:
                images = []

            self.S = State(instance=self.instance, rnd_state=rnd_state_test_transition)
            reward = 0

            while self.S.time_period <= self.instance.last_period:
                # choose action
                action = self.policy.choose_action(state=self.S, rnd_state=rnd_state_test_action, is_learning=False)

                # compute reward
                reward += pow(self.instance.discount_rate, self.S.time_period) * self.S.demand[action]
             
                if fig_map:
                    fig = self.S.fig_map(action=action)
                    img_bytes = fig.to_image(format="png")
                    img = Image.open(io.BytesIO(img_bytes))
                    images.append(img)

                    if self.S.time_period == 0:
                        fig.show()

                # transition
                self.policy.transition(state=self.S, 
                                       action=action, 
                                       rnd_state=rnd_state_test_transition,
                                       is_learning=False)   

            
            rewards.append(reward)

            if fig_map:
                if not os.path.exists('output'):
                    os.mkdir('output')

                # Create a GIF from the in-memory images
                images[0].save('output/animation_round_' + str(r) + '.gif', 
                               save_all=True, 
                               append_images=images[1:], 
                               duration=1000,    # 500 
                               loop=0)

                print('GIF created for round', r)
        
        return rewards

    def fig_period_location(self):      
        value_matrix = []
        visits_matrix = []
        
        for n in self.instance.nodes:
            tmp_values = []
            tmp_visits = []
            for t in self.instance.periods:
                if self.policy_name == 'Q-Table':
                    maxQ = 0
                    numVisits = 0

                    for a in self.instance.nodes:                            
                        state_action = ((t, n), a)
                        if state_action in self.Q_table:
                            numVisits += self.Q_table[state_action]['visits']

                            if self.Q_table[state_action]['value'] > maxQ:
                                maxQ = self.Q_table[state_action]['value']
                    
                    tmp_values.append(round(maxQ, 1))
                    tmp_visits.append(numVisits)

                elif self.policy_name == 'VFA':
                    state_hash = (t, n)
                    if state_hash not in self.VFA_value:
                        tmp_values.append(0)
                        tmp_visits.append(0)
                    else:
                        tmp_values.append(round(self.VFA_value[t][n], 1))
                        tmp_visits.append(self.VFA_visits[t][n])

                         
            value_matrix.append(tmp_values)
            visits_matrix.append(tmp_visits)

        fig = go.Figure(data=go.Heatmap(
                   z=visits_matrix,
                   x=self.instance.periods,
                   y=self.instance.nodes_letter,
                   colorscale='Reds',
                   hoverongaps = False,                 
                   text=value_matrix,
                   texttemplate="%{text}",
                   textfont={"size":12},
                   ))
        
        fig.update_layout(
            xaxis_title="Periods",
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            yaxis_title="Location",
            width=800, height=800
        )

        fig.show()

    def tab_Q_table(self):
        df_Q = {'t': [], 'node': [], 'action': [], 'value': [], 'visits': []}
        for t in self.instance.periods:
            for n in self.instance.nodes:
                for a in self.instance.nodes:
                    # if self.Q_online[(t,n)][a] == 0 and self.Q_online_visits[(t,n)][a] > 0:
                    df_Q['t'].append(t)
                    df_Q['node'].append(n)
                    df_Q['action'].append(a)
                    df_Q['value'].append(self.Q_online[(t,n)][a])
                    df_Q['visits'].append(self.Q_online_visits[(t,n)][a])
        
        df = pd.DataFrame(df_Q)
        print(df)

    def save_policy(self):
        folder_path = 'models/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        fw = open(folder_path + self.instance.name + ' ' + str(self.name) + '.LEARN', 'wb')

        pickle.dump(self.policy, fw)

        # if self.policy_name == 'Q-Table':
        #     pickle.dump(self.Q_online, fw)
        # elif self.policy_name == 'VFA':
        #     pickle.dump(self.VFA_value, fw)
        # elif self.policy_name == 'DQN':
        #     pickle.dump(self.DQN_online, fw)

        fw.close()

    def load_policy(self):

        fr = open('models/' + self.instance.name + ' ' + str(self.name) + '.LEARN', 'rb')
        self.policy = pickle.load(fr)

        # if self.policy_name == 'Q-Table':
        #     self.Q_online = pickle.load(fr)
        #     # self.tab_Q_table()
        # elif self.policy_name == 'VFA':
        #     self.VFA_value = pickle.load(fr)
        # elif self.policy_name == 'DQN':
        #     self.DQN_online = pickle.load(fr)

        fr.close()


class State:

    def __init__(self, instance, rnd_state):
        self.instance = instance

        # initialize values
        self.vehicle_position = self.instance.nodes[0]
        self.time_period = self.instance.periods[0]
        self.demand = None

        # initial demand
        if rnd_state is not None:
            self.create_demand(rnd_state)

    def hash(self):
        # return str(self.time_period) + '_' + self.vehicle_position
        return (self.time_period, self.vehicle_position)    # (time, node)

    def hash_nn(self):
        # returns the state information as an numpy array, which can be used as input for the neural network
        # hash_list = [self.time_period, self.vehicle_position]
        # for a in self.instance.nodes: 
        #     hash_list.append(1 if self.demand[a] > 0 else 0)

        # just time and node
        hash_list = [1 if t == self.time_period else 0 for t in self.instance.periods]
        for a in self.instance.nodes: 
            hash_list.append(1 if a == self.vehicle_position else 0)
        
        # add demand
        # for a in self.instance.nodes: 
        #     hash_list.append(1 if self.demand[a] > 0 else 0)
        
        return np.array(hash_list)   # [time, node]

    def create_demand(self, rnd_state):
        if self.time_period > self.instance.periods[-1]:
            self.demand = np.array([0 for a in self.instance.nodes])
        else:
            # compute random numbers
            state_probs = rnd_state.rand(len(self.instance.nodes))

            # compare random numbers with demand distribution
            demand_info = self.instance.demand_and_probabilities[(self.vehicle_position, self.time_period)]
            demand_probs = demand_info['p']
            demand_value = demand_info['d']            
            self.demand = np.where(state_probs <= demand_probs, demand_value, 0).astype(int)

            # The following would also work, but less efficient
            # self.demand = [value if np.random.rand() < prob else 0 for value, prob in zip(self.instance.demand_and_probabilities[(self.vehicle_position, self.time_period)]['d'], self.instance.demand_and_probabilities[(self.vehicle_position, self.time_period)]['p'])]

            # The following would also work, but weould be everen less efficient the the previous one.
            # self.demand = np.array([np.random.choice(a=[self.instance.demand[(self.vehicle_position, a, self.time_period)]['d'], 0], 
            #                                         p=[self.instance.demand[(self.vehicle_position, a, self.time_period)]['p'], 1 - self.instance.demand[(self.vehicle_position, a, self.time_period)]['p']]) 
            #                                         for a in self.instance.nodes])
    
    def next_state(self, action, rnd_state):
        self.time_period += 1
        self.vehicle_position = action

        self.create_demand(rnd_state)

    def fig_map(self, action):
        # create DataFrame with state information
        tmp_dict = {
            'x': [], 'y': [], 'demand': [], 'name': [], 'text': []
        }
        for a in self.instance.nodes:
            d = self.demand[a]
            # if d > 0: 
            tmp_dict['x'].extend([self.instance.xy_coordinates[self.vehicle_position]['x'], self.instance.xy_coordinates[a]['x'], np.nan])
            tmp_dict['y'].extend([self.instance.xy_coordinates[self.vehicle_position]['y'], self.instance.xy_coordinates[a]['y'], np.nan])
            tmp_dict['demand'].extend([d, d, np.nan])
            tmp_dict['name'].extend([self.vehicle_position, a, np.nan])
            tmp_dict['text'].extend([self.instance.nodes_letter[self.vehicle_position], self.instance.nodes_letter[a] + ' (' + str(d) + ')', ''])

        df = pd.DataFrame(tmp_dict)

        # create figure        
        fig = px.line(df, x='x', y='y', markers=True, width=1300, height=1300, title='t=' + str(self.time_period), template='simple_white')

        # add labels        
        for i in range(len(df)):
            color = 'red' if df['name'][i] == action else 'black'

            fig.add_annotation(
                x=df['x'][i],
                y=df['y'][i],
                text=df['text'][i],
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
                font=dict(family="Arial", size=32, color=color)
                )

        # fig.update_layout(...) is used to modify the overall layout of the figure (like title, axes, legend, font, size).      
        fig.update_layout(
            title=dict(font=dict(family="Arial", size=28)),
            xaxis=dict(range=[0, 1000], title='X-Axis', visible=False),
            yaxis=dict(range=[0, 1000], title='Y-Axis', visible=False)
            )

        # fig.update_traces(...) is used to modify properties of the data traces (like lines, markers, text).
        fig.update_traces(marker=dict(color='black', size=12, symbol='square'),
                          line=dict(color='black'))

        # show figure
        return fig


def testing(do_function, policies, instance, rounds, fig_map, fig_boxplot, save_figure=True, name=' '):
    if do_function:

        df_dict = {'policy': [], 'reward': [], 'round': []}
            
        for policy_info in policies:
            if policy_info['do']:
                print_str ='Test policy (' + str(rounds) + ' rounds): ' + policy_info['name']  

                M = Taxicab(instance=instance, policy_info=policy_info)

                if policy_info['policy'] in ['Q-Table', 'VFA', 'DQN']:
                    M.load_policy()
                    
                elif policy_info['policy'] == 'Optimal':
                    print_str += ' (Expected value t=n=0: ' + str(round(M.policy.state_value[0][0], 4)) + ')'

                print(print_str)

                # test policy            
                rewards = M.test(rounds=rounds, fig_map=fig_map)

                # store results
                df_dict['policy'].extend([policy_info['name'] for r in range(rounds)])
                df_dict['reward'].extend(rewards)
                df_dict['round'].extend([r+1 for r in range(rounds)])
                
        # Create a DataFrame from the dictionary
        df = pd.DataFrame(df_dict)
        
        # Calculate the average reward, median reward, etc. for each policy
        summary = df.groupby('policy')['reward'].agg(['mean', 'median', 'std', 'min', 'max'])
        summary = summary.sort_values(by='mean', ascending=False)
        print(summary)

        # Create a boxplot using plotly express
        if fig_boxplot:
            fig = px.box(df, x='policy', y='reward', title='Boxplot of Rewards by Policy')
            fig.show()

            if save_figure:
                fig.write_html('output/' + instance.name + ' test ' + name + ' boxplot ' + time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()) + '.html')


def learning(do_function, policies, instance, fig_learning=False, fig_learning_animation=False, fig_PDS=False, tab_Q=False, save_figures=True, name='', test_rounds=200, test_interval=50):
    if do_function:
        
        test_interval = test_interval if fig_learning else  0  # default: 50

        df_dict = {'policy': [], 'reward': [], 'round': []}
            
        for policy_info in policies:
            if policy_info['do']:                
                start_time = time.time()    # Record the start time

                print('Learn policy:', policy_info['name'])
                rounds = 2000 if 'rounds' not in policy_info else policy_info['rounds'] 

                M = Taxicab(instance=instance, policy_info=policy_info)

                test_history = M.learn(rounds=rounds, test_interval=test_interval, test_rounds=test_rounds)

                if policy_info['policy'] in ['VFA']:
                    if fig_PDS:
                        M.fig_period_location()
                elif policy_info['policy'] in ['Q-Table']:
                    if tab_Q:
                        M.tab_Q_table()
                
                df_dict['policy'].extend([policy_info['name'] for p in range(len(test_history['reward']))])
                df_dict['reward'].extend(test_history['reward'])
                df_dict['round'].extend(test_history['round'])

                print(f" -- Total time: {int((t:=time.time()-start_time)//60)} min {int(t%60)} sec")

        
        if fig_learning:
            # Create a DataFrame from the dictionary
            df = pd.DataFrame(df_dict)
            
            # Create a line graph
            fig = px.line(df, x='round', y='reward', color='policy', title='Rewards by Policy over Rounds', template='simple_white', height=1000, width=1800)
            
            # fig.update_layout(...) is used to modify the overall layout of the figure (like title, axes, legend, font, size).      
            fig.update_layout(
                font=dict(family="Arial", size=24),
                title=dict(font=dict(family="Arial", size=28)),
                xaxis=dict(title='Number of iterations', visible=True),
                yaxis=dict( title='Reward following the so-far best policy', visible=True)
                )

            fig.show()

            if save_figures:
                fig.write_html('output/' + instance.name + ' learn ' + name + ' curve ' + time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()) + '.html')

            
            # ANIMATED LEARNING CURVE
            if fig_learning_animation:
                # Loop through unique values in 'round' column
                unique_rounds = df['round'].unique()
                images = []
                for round_value in unique_rounds:
                    # Select rows where 'round' value is less than the current iteration value
                    filtered_df = df[df['round'] < round_value]

                    fig = px.line(filtered_df, x='round', y='reward', color='policy', template='simple_white', height=1000, width=1800)

                    # fig.update_layout(...) is used to modify the overall layout of the figure (like title, axes, legend, font, size).      
                    fig.update_layout(
                        showlegend=False,
                        font=dict(family="Arial", size=24),
                        title=dict(font=dict(family="Arial", size=28)),
                        xaxis=dict(range=[0, 20000], title='Number of iterations', visible=True),
                        yaxis=dict(range=[500, 600], title='Reward following the so-far best policy', visible=True)
                        )

                    # fig.update_traces(...) is used to modify properties of the data traces (like lines, markers, text).
                    fig.update_traces(marker=dict(color='black', size=12, symbol='square'),
                                    line=dict(color='black'))

                    img_bytes = fig.to_image(format="png")
                    img = Image.open(io.BytesIO(img_bytes))
                    images.append(img)

                    print('Learning curve created for round', round_value, '/', unique_rounds[-1])

                if not os.path.exists('output'):
                    os.mkdir('output')

                # Create a GIF from the in-memory images
                images[0].save('output/animation_learning_curve.gif', 
                                save_all=True, 
                                append_images=images[1:], 
                                duration=100,    # 500 
                                loop=0)

                print('GIF created for the learning curve')

                
i = Instance(nodes=10, periods=10, rnd_seed=100, discount_rate=0.9)

very_few_rounds = 2000
few_rounds = 3000
more_rounds = 5000
even_more_rounds = 10000
many_rounds = 15000
default_rounds = few_rounds

learning(do_function=True, fig_learning=True, fig_learning_animation=False, fig_PDS=False, save_figures=True, tab_Q=False, instance=i, test_rounds=500, test_interval=200, policies=[  
    # VFA  
    {'do': True, 'policy': 'VFA', 'name': 'VFA (opt - harmonic)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8}, # opt (harmonic learning rate)
    {'do': False, 'policy': 'VFA', 'name': 'VFA (opt - constant)', 'rounds': 10000, 'learning_rate': 0.01, 'exploration_rate': 0.7}, # opt (constant learning rate)
    {'do': False, 'policy': 'VFA', 'name': 'VFA (test)', 'rounds': 3000, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8}, # test
    
    # Q-Learning
    {'do': True, 'policy': 'Q-Table', 'name': 'Q-Table (standard)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8}, # opt (harmonic learning rate)
    #
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=1)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_target_q': True, 'target_update_frequency': 1}, # 
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=1000)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_target_q': True, 'target_update_frequency': 1000}, # 
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=inf)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_target_q': True, 'target_update_frequency': default_rounds}, #
    #
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=1)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_double_q': True, 'target_update_frequency': 1}, # 
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=1000)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_double_q': True, 'target_update_frequency': 1000}, # 
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=inf)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_double_q': True, 'target_update_frequency': default_rounds}, #
    
    #     
    # DQN
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128}, # opt (stable after about 2k iterations)
    #
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=1)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_target_q': True, 'target_update_frequency': 1}, # 
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target, update frequency=100)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128,'use_target_q': True, 'target_update_frequency': 100}, # 
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=inf)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_target_q': True, 'target_update_frequency': default_rounds}, # 
    #
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=1)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'target_update_frequency': 1}, # 
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double, update frequency=100)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'target_update_frequency': 100}, # 
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=inf)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'target_update_frequency': default_rounds}, # 
    ])

testing(do_function=True, instance=i, rounds=4000, fig_map=False, fig_boxplot=True, save_figure=True, policies=[
    {'do': True, 'policy': 'Optimal', 'name': 'Optimal'},
    #
    {'do': True, 'policy': 'VFA', 'name': 'VFA (opt - harmonic)'},
    {'do': False, 'policy': 'VFA', 'name': 'VFA (opt - constant)'},
    
    # Q-Table    
    {'do': True, 'policy': 'Q-Table', 'name': 'Q-Table (standard)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=1)'},
    {'do': True, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=100)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=inf)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=1)'},
    {'do': True, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=100)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=inf)'},
    
    # DQN
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=1)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target, update frequency=100)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=inf)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=1)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double, update frequency=100)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=inf)'},

    #
    {'do': False, 'policy': 'Greedy', 'name': 'Greedy'},
    {'do': False, 'policy': 'Random', 'name': 'Random'}
    ])

if profiling:
    # Stop profiling
    profiler.disable()

    # Print profiling stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime') # Sort by total time spent in the function
    stats.print_stats(15)   # Print top 15 functions
