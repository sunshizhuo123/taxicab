import numpy as np
import plotly.express as px
import pandas as pd


import statistics
import time

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
        return (self.time_period, self.vehicle_position)  # (time, node)

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

        return np.array(hash_list)  # [time, node]

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
            tmp_dict['x'].extend(
                [self.instance.xy_coordinates[self.vehicle_position]['x'], self.instance.xy_coordinates[a]['x'],
                 np.nan])
            tmp_dict['y'].extend(
                [self.instance.xy_coordinates[self.vehicle_position]['y'], self.instance.xy_coordinates[a]['y'],
                 np.nan])
            tmp_dict['demand'].extend([d, d, np.nan])
            tmp_dict['name'].extend([self.vehicle_position, a, np.nan])
            tmp_dict['text'].extend(
                [self.instance.nodes_letter[self.vehicle_position], self.instance.nodes_letter[a] + ' (' + str(d) + ')',
                 ''])

        df = pd.DataFrame(tmp_dict)

        # create figure
        fig = px.line(df, x='x', y='y', markers=True, width=1300, height=1300, title='t=' + str(self.time_period),
                      template='simple_white')

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
