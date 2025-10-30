import numpy as np
import os
import pickle
import statistics
import time
from structures.State import State
from PIL import Image


from policies import policy_greedy, policy_random, policy_optimal, policy_vfa, policy_Q_table, policy_DQN

class TaxicabProblem:

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
        folder_path = 'results/learned_models/'

        fw = open(folder_path + self.instance.name + ' ' + str(self.name) + '.LEARN', 'wb')

        pickle.dump(self.policy, fw)

        fw.close()

    def load_policy(self):

        fr = open('results/learned_models/' + self.instance.name + ' ' + str(self.name) + '.LEARN', 'rb')
        self.policy = pickle.load(fr)

        fr.close()
