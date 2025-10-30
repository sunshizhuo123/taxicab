from structures.Instance import Instance
from structures.TaxicabProblem import TaxicabProblem

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

import time

profiling = False

if profiling:
    profiler = cProfile.Profile()
    profiler.enable()



def testing(do_function, policies, instance, rounds, fig_map, fig_boxplot, save_figure=True, name=' '):
    if do_function:

        df_dict = {'policy': [], 'reward': [], 'round': []}
            
        for policy_info in policies:
            if policy_info['do']:
                print_str ='Test policy (' + str(rounds) + ' rounds): ' + policy_info['name']  

                M = TaxicabProblem(instance=instance, policy_info=policy_info)

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
                fig.write_html('results/figures/' + instance.name + ' test ' + name + ' boxplot ' + time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()) + '.html')


def learning(do_function, policies, instance, fig_learning=False, fig_learning_animation=False, fig_PDS=False, tab_Q=False, save_figures=True, name='', test_rounds=200, test_interval=50):
    if do_function:
        
        test_interval = test_interval if fig_learning else  0  # default: 50

        df_dict = {'policy': [], 'reward': [], 'round': []}
            
        for policy_info in policies:
            if policy_info['do']:                
                start_time = time.time()    # Record the start time

                print('Learn policy:', policy_info['name'])
                rounds = 2000 if 'rounds' not in policy_info else policy_info['rounds'] 

                M = TaxicabProblem(instance=instance, policy_info=policy_info)

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
                fig.write_html('results/figures/' + instance.name + ' learn ' + name + ' curve ' + time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()) + '.html')

            
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


                # Create a GIF from the in-memory images
                images[0].save('results/animation_learning_curve.gif', 
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
default_rounds = very_few_rounds

learning(do_function=False, fig_learning=True, fig_learning_animation=False, fig_PDS=False, save_figures=True, tab_Q=False, instance=i, test_rounds=500, test_interval=200, policies=[  
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
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=100)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128,'use_target_q': True, 'target_update_frequency': 100}, # 
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=inf)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_target_q': True, 'target_update_frequency': default_rounds}, # 
    #
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=1)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'target_update_frequency': 1}, # 
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=100)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'target_update_frequency': 100}, # 
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
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=100)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=inf)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=1)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=100)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=inf)'},
    
    # DQN
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=1)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=100)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=inf)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=1)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=100)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=inf)'},

    #
    {'do': True, 'policy': 'Greedy', 'name': 'Greedy'},
    {'do': True, 'policy': 'Random', 'name': 'Random'}
    ])

if profiling:
    # Stop profiling
    profiler.disable()

    # Print profiling stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime') # Sort by total time spent in the function
    stats.print_stats(15)   # Print top 15 functions
