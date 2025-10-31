
import numpy as np

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
        self.nodes = [n for n in range(nodes)]  # List of nodes as index, starting at 0: 0, 1, 2, ...
        self.nodes_letter = generate_excel_letters(n=nodes)  # List of nodes: A, B, ..., AA, AB, AC, ...
        self.periods = [t for t in range(periods)]  # list of time periods: 1, 2, 3, ...
        self.last_period = self.periods[-1]
        self.discount_rate = discount_rate  # discount rate: The discount is used to balance immediate and future reward. This value takes a number between 0 and 1.

        rnd_state_instance = np.random.RandomState(rnd_seed)

        # used in create_demand() in State
        self.demand_and_probabilities = {}
        for i in self.nodes:
            for t in self.periods:
                p = np.round(rnd_state_instance.random(size=len(self.nodes)), 6)
                d = rnd_state_instance.randint(0, 101, size=len(self.nodes))
                self.demand_and_probabilities[(i, t)] = dict(p=p, d=d)

        # asign x-y coordinates
        self.xy_coordinates = {n: {'x': rnd_state_instance.randint(0, 1000), 'y': rnd_state_instance.randint(0, 1000)}
                               for n in self.nodes}

        # print(self.demand)
        self.name = 'N' + str(nodes) + 'xT' + str(periods)

    def optimal_state_values(self):
        """
        This function computes the value of each state using backward dynamic programming,
        i.e., it applies Bellman's equation.

        :return: n/a
        """

        # create a dict storing the "actual" expected value of all states
        states = np.array([[0.0 for n in self.nodes] for t in self.periods])  # [time][node]

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
                        D.append([demand + (self.discount_rate * states[t + 1][target]),
                                  prob])

                        D.append([(self.discount_rate * states[t + 1][target]),
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

