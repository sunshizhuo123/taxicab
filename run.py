from structures.Instance import Instance
from structures.Taxicab import Taxicab

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

import cProfile
import io
import numpy as np
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import pstats
from scipy import stats as scipy_stats
import csv
from datetime import datetime

import statistics
import time
import time

profiling = False

if profiling:
    profiler = cProfile.Profile()
    profiler.enable()


def calculate_comprehensive_metrics(rewards, policy_name):
    """Calculate comprehensive metrics for a policy's test results"""
    metrics = {}

    # Effectiveness metrics
    metrics['mean_reward'] = np.mean(rewards)
    metrics['std_reward'] = np.std(rewards)
    metrics['sdr'] = metrics['std_reward'] / metrics['mean_reward'] if metrics['mean_reward'] != 0 else 0

    # Risk metrics
    metrics['percentile_5'] = np.percentile(rewards, 5)
    metrics['percentile_95'] = np.percentile(rewards, 95)
    metrics['median'] = np.median(rewards)
    metrics['min'] = np.min(rewards)
    metrics['max'] = np.max(rewards)

    # Additional statistics(to see if distribution stable or not)
    metrics['skewness'] = scipy_stats.skew(rewards)  #the reward distribution skewed towards high returns or low returns
    metrics['kurtosis'] = scipy_stats.kurtosis(rewards) #Probability of extreme returns (sharp rises/sharp falls) occurring

    return metrics


def save_test_results_to_csv(results_dict, instance_name):
    """Save test results to CSV files"""
    if not os.path.exists('experiment'):
        os.makedirs('experiment')

    # Save raw test results
    test_results_file = f'experiment/test_results_{instance_name}.csv'
    with open(test_results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'round', 'reward'])
        for policy_name, rewards in results_dict.items():
            for i, reward in enumerate(rewards):
                writer.writerow([policy_name, i + 1, reward])

    # Save performance metrics
    metrics_file = f'experiment/performance_metrics_{instance_name}.csv'
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'metric', 'value'])
        for policy_name, rewards in results_dict.items():
            metrics = calculate_comprehensive_metrics(rewards, policy_name)
            for metric_name, metric_value in metrics.items():
                writer.writerow([policy_name, metric_name, metric_value])

    return test_results_file, metrics_file


def testing(do_function, policies, instance, rounds, fig_map, fig_boxplot, save_figure=True, name=' ', save_csv=False):
    if do_function:

        df_dict = {'policy': [], 'reward': [], 'round': []}
        results_dict = {}  # For saving to CSV
            
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
                results_dict[policy_info['name']] = rewards
                
        # Save to CSV if requested
        if save_csv:
            test_file, metrics_file = save_test_results_to_csv(results_dict, instance.name)
            print(f"Test results saved to {test_file}")
            print(f"Performance metrics saved to {metrics_file}")
                
        # Create a DataFrame from the dictionary
        df = pd.DataFrame(df_dict)
        
        # Calculate the average reward, median reward, etc. for each policy
        summary = df.groupby('policy')['reward'].agg(['mean', 'median', 'std', 'min', 'max'])
        summary = summary.sort_values(by='mean', ascending=False)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(summary)

        # Create a boxplot using plotly express
        if fig_boxplot:
            fig = px.box(df, x='policy', y='reward', title='Boxplot of Rewards by Policy')
            fig.show()

            if save_figure:
                fig.write_html('output/' + instance.name + ' test ' + name + ' boxplot ' + time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()) + '.html')


def learning(do_function, policies, instance, fig_learning=False, fig_learning_animation=False, fig_PDS=False, tab_Q=False, save_figures=True, name='', test_rounds=200, test_interval=50, save_progress_csv=False):
    if do_function:
        
        test_interval = test_interval if (fig_learning or save_progress_csv) else  0  # default: 50

        df_dict = {'policy': [], 'reward': [], 'round': []}
        
        # Initialize CSV file for training progress if requested
        if save_progress_csv:
            if not os.path.exists('experiment'):
                os.makedirs('experiment')
            progress_file = 'experiment/training_progress.csv'
            with open(progress_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['algorithm', 'round', 'test_reward', 'timestamp'])
            
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
                
                # Save progress to CSV if requested
                if save_progress_csv:
                    with open(progress_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for i in range(len(test_history['round'])):
                            writer.writerow([
                                policy_info['name'],
                                test_history['round'][i],
                                test_history['reward'][i],
                                datetime.now().isoformat()
                            ])

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


# def robustness_testing(policies, base_instance, test_type='demand_shift', shift_magnitude=0.2, rounds=1000):
#     """
#     Test robustness of policies under different conditions
#
#     test_type: 'demand_shift' for OD-probability shifts, 'new_seeds' for unseen demand seeds
#     shift_magnitude: for demand_shift, the percentage change in probabilities (e.g., 0.2 for ±20%)
#     """
#     if not os.path.exists('experiment'):
#         os.makedirs('experiment')
#
#     results = {}
#
#     for policy_info in policies:
#         if policy_info['do']:
#             print(f"Robustness testing for {policy_info['name']} - {test_type}")
#
#             if test_type == 'new_seeds':
#                 # Test with different random seeds
#                 seed_results = []
#                 for seed in [200, 300, 400, 500, 600]:  # Different seeds from training
#                     test_instance = Instance(nodes=len(base_instance.nodes),
#                                             periods=len(base_instance.periods),
#                                             rnd_seed=seed,
#                                             discount_rate=base_instance.discount_rate)
#
#                     M = Taxicab(instance=test_instance, policy_info=policy_info)
#                     if policy_info['policy'] in ['Q-Table', 'VFA', 'DQN']:
#                         # Load the model trained on base instance
#                         fr = open('models/' + base_instance.name + ' ' + str(policy_info['name']) + '.LEARN', 'rb')
#                         M.policy = pickle.load(fr)
#                         fr.close()
#
#                     rewards = M.test(rounds=rounds, fig_map=False)
#                     seed_results.extend(rewards)
#
#                 results[policy_info['name']] = seed_results
#
#             elif test_type == 'demand_shift':
#                 # Test with shifted demand probabilities
#                 test_instance = Instance(nodes=len(base_instance.nodes),
#                                         periods=len(base_instance.periods),
#                                         rnd_seed=100,  # Use same seed but shift probabilities
#                                         discount_rate=base_instance.discount_rate)
#
#                 # Modify demand probabilities
#                 for key in test_instance.demand_and_probabilities:
#                     probs = test_instance.demand_and_probabilities[key]['p']
#                     # Apply random shift within ±shift_magnitude
#                     shift = np.random.uniform(-shift_magnitude, shift_magnitude, size=len(probs))
#                     new_probs = np.clip(probs * (1 + shift), 0, 1)
#                     new_probs = new_probs / np.sum(new_probs)  # Normalize
#                     test_instance.demand_and_probabilities[key]['p'] = new_probs
#
#                 M = Taxicab(instance=test_instance, policy_info=policy_info)
#                 if policy_info['policy'] in ['Q-Table', 'VFA', 'DQN']:
#                     # Load the model trained on base instance
#                     fr = open('models/' + base_instance.name + ' ' + str(policy_info['name']) + '.LEARN', 'rb')
#                     M.policy = pickle.load(fr)
#                     fr.close()
#
#                 rewards = M.test(rounds=rounds, fig_map=False)
#                 results[policy_info['name']] = rewards
#
#     # Save robustness results
#     filename = f'experiment/robustness_{test_type}_{base_instance.name}.csv'
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['algorithm', 'round', 'reward', 'test_type'])
#         for policy_name, rewards in results.items():
#             for i, reward in enumerate(rewards):
#                 writer.writerow([policy_name, i+1, reward, test_type])
#
#     print(f"Robustness results saved to {filename}")
#
#     # Calculate and print summary statistics
#     for policy_name, rewards in results.items():
#         mean_reward = np.mean(rewards)
#         std_reward = np.std(rewards)
#         print(f"{policy_name}: Mean={mean_reward:.2f}, Std={std_reward:.2f}")
#
#     return results


# def scale_transfer_testing(policies, base_instance, scale_configs, rounds=1000):
#     """
#     Test how policies trained on base instance perform on different scales
#
#     scale_configs: list of tuples (nodes, periods) to test
#     """
#     if not os.path.exists('experiment'):
#         os.makedirs('experiment')
#
#     results = {}
#
#     for nodes, periods in scale_configs:
#         scale_name = f"N{nodes}xT{periods}"
#         print(f"\nScale transfer testing: {scale_name}")
#         results[scale_name] = {}
#
#         # Create new instance with different scale
#         test_instance = Instance(nodes=nodes, periods=periods,
#                                 rnd_seed=100, discount_rate=base_instance.discount_rate)
#
#         for policy_info in policies:
#             if policy_info['do']:
#                 print(f"  Testing {policy_info['name']}")
#
#                 M = Taxicab(instance=test_instance, policy_info=policy_info)
#
#                 if policy_info['policy'] in ['Q-Table', 'VFA', 'DQN']:
#                     if nodes == len(base_instance.nodes) and periods == len(base_instance.periods):
#                         # Same scale, can load trained model
#                         try:
#                             fr = open('models/' + base_instance.name + ' ' + str(policy_info['name']) + '.LEARN', 'rb')
#                             M.policy = pickle.load(fr)
#                             fr.close()
#                         except:
#                             print(f"    Warning: Could not load model for {policy_info['name']}")
#                     else:
#                         # Different scale, need to adapt or retrain
#                         print(f"    Note: Using untrained model for different scale")
#
#                 rewards = M.test(rounds=rounds, fig_map=False)
#                 results[scale_name][policy_info['name']] = rewards
#
#     # Save scale transfer results
#     filename = f'experiment/scale_transfer_{base_instance.name}.csv'
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['scale', 'algorithm', 'round', 'reward'])
#         for scale_name, scale_results in results.items():
#             for policy_name, rewards in scale_results.items():
#                 for i, reward in enumerate(rewards):
#                     writer.writerow([scale_name, policy_name, i+1, reward])
#
#     print(f"\nScale transfer results saved to {filename}")
#
#     # Print summary statistics
#     for scale_name, scale_results in results.items():
#         print(f"\n{scale_name}:")
#         for policy_name, rewards in scale_results.items():
#             mean_reward = np.mean(rewards)
#             std_reward = np.std(rewards)
#             print(f"  {policy_name}: Mean={mean_reward:.2f}, Std={std_reward:.2f}")
#
#     return results

                
# Make scale configurable
def create_instance(nodes=10, periods=10, rnd_seed=100, discount_rate=0.9):
    return Instance(nodes=nodes, periods=periods, rnd_seed=rnd_seed, discount_rate=discount_rate)

# Main execution block - only run if this is the main script
if __name__ == "__main__":
    # Default instance
    i = create_instance(nodes=10, periods=10, rnd_seed=100, discount_rate=0.9)
    
    very_few_rounds = 2000
    few_rounds = 3000
    more_rounds = 5000
    even_more_rounds = 10000
    many_rounds = 15000
    default_rounds = many_rounds
    
    learning(do_function=True, fig_learning=True, fig_learning_animation=False, fig_PDS=False, save_figures=True, tab_Q=False, instance=i, test_rounds=500, test_interval=200, save_progress_csv=True, policies=[
    # VFA
    {'do': True, 'policy': 'VFA', 'name': 'VFA (opt - harmonic)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8}, # opt (harmonic learning rate)
    {'do': False, 'policy': 'VFA', 'name': 'VFA (opt - constant)', 'rounds': 10000, 'learning_rate': 0.01, 'exploration_rate': 0.7}, # opt (constant learning rate)
    {'do': False, 'policy': 'VFA', 'name': 'VFA (test)', 'rounds': 3000, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8}, # test
    {'do': True, 'policy': 'VFA', 'name': 'VFA (opt - harmonic0.5)', 'rounds': default_rounds, 'learning_rate': 0.1, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.5},
    # Q-Learning
    {'do': True, 'policy': 'Q-Table', 'name': 'Q-Table (standard)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8}, # opt (harmonic learning rate)
    #
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=1)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_target_q': True, 'target_update_frequency': 1}, #
    {'do': True, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=100)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_target_q': True, 'target_update_frequency': 100}, #
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=inf)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_target_q': True, 'target_update_frequency': default_rounds}, #
    #
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=1)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_double_q': True, 'target_update_frequency': 1}, #
    {'do': True, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=100)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_double_q': True, 'target_update_frequency': 100}, #
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=inf)', 'rounds': default_rounds, 'learning_rate_harmonic': True, 'learning_rate_harmonic_a': 10, 'exploration_rate': 0.8, 'use_double_q': True, 'target_update_frequency': default_rounds}, #

    #
    # DQN
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard, h=128)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 100000, 'learning_rate': 0.001, 'exploration_rate_decay_rounds': 2000, 'neurons': 128}, # opt (stable after about 2k iterations)
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard, h=256)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 100000, 'learning_rate': 0.0005, 'exploration_rate_decay_rounds': 2000, 'neurons': 256},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard lr=0.001, h=128)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.001, 'exploration_rate_decay_rounds': 2000, 'neurons': 128},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard lr=0.0005, h=128)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.0005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128},

    #
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=1)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_target_q': True, 'target_update_frequency': 1}, #
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target, update frequency=100)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 256,'use_target_q': True, 'target_update_frequency': 100}, #
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=inf)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_target_q': True, 'target_update_frequency': default_rounds}, #
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=1000)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 100000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_target_q': True, 'target_update_frequency': 1000},  #
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=100)', 'rounds': default_rounds, 'learning_rate': 0.005, 'exploration_rate': 0.5, 'exploration_rate_decay': 0.995, 'neurons': 128, 'use_target_q': True, 'target_update_frequency': 100},  #

    #
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=1)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'target_update_frequency': 1}, #
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double, update frequency=100)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 256, 'use_double_q': True, 'target_update_frequency': 100}, #
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=inf)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'target_update_frequency': default_rounds}, #
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double, update frequency=1000)', 'rounds': default_rounds, 'batch_size': 64, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate_decay_rounds': 2000, 'neurons': 256, 'use_double_q': True, 'target_update_frequency': 1000},  #
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=100)', 'rounds': default_rounds, 'learning_rate': 0.005, 'exploration_rate': 0.5, 'exploration_rate_decay': 0.995, 'neurons': 128, 'use_double_q': True, 'target_update_frequency': 100},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, decay0.995)', 'rounds': default_rounds, 'batch_size': 64, 'buffer_size': 20000, 'learning_rate': 0.005, 'exploration_rate': 0.5, 'exploration_rate_decay': 0.995, 'neurons': 128, 'use_double_q': True, 'target_update_frequency': 100},  #

    #
    # Soft update variants
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target + soft update, tau=0.001, h=128)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.001, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_target_q': True, 'use_soft_update': True, 'tau': 0.001},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target + soft update, tau=0.001, h=256)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.001, 'exploration_rate_decay_rounds': 2000, 'neurons': 256, 'use_target_q': True, 'use_soft_update': True, 'tau': 0.001},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target + soft update, tau=0.001, h=128, v2)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.0005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_target_q': True, 'use_soft_update': True, 'tau': 0.001},#test
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target + soft update, tau=0.001, h=256, v2)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.0005, 'exploration_rate_decay_rounds': 2000, 'neurons': 256, 'use_target_q': True, 'use_soft_update': True, 'tau': 0.001},#test

    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, lr=0.0005, h=128)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.0005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'use_soft_update': True, 'tau': 0.001},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, lr=0.001, h=128)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.001, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'use_soft_update': True, 'tau': 0.001},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, lr=0.0005, h=256)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.0005, 'exploration_rate_decay_rounds': 2000, 'neurons': 256, 'use_double_q': True, 'use_soft_update': True, 'tau': 0.001},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, lr=0.001, h=256)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 20000, 'learning_rate': 0.001, 'exploration_rate_decay_rounds': 2000, 'neurons': 256, 'use_double_q': True, 'use_soft_update': True, 'tau': 0.001},

    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, v1)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 120000, 'learning_rate': 0.0005, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'use_soft_update': True, 'tau': 0.005},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, v2)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 120000, 'learning_rate': 0.0005, 'exploration_rate_decay_rounds': 2000, 'neurons': 256, 'use_double_q': True, 'use_soft_update': True, 'tau': 0.005},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target + soft update, tau=0.001, 0.5)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 120000,'learning_rate': 0.001, 'exploration_rate': 0.5, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_target_q': True, 'use_soft_update': True, 'tau': 0.001},#new try
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, 0.5)', 'rounds': default_rounds, 'batch_size': 100, 'buffer_size': 120000, 'learning_rate': 0.001, 'exploration_rate': 0.5, 'exploration_rate_decay_rounds': 2000, 'neurons': 128, 'use_double_q': True, 'use_soft_update': True, 'tau': 0.001},#new try


    ])

    testing(do_function=True, instance=i, rounds=10000, fig_map=False, fig_boxplot=True, save_figure=True, save_csv=True, policies=[
    {'do': True, 'policy': 'Optimal', 'name': 'Optimal'},
    #
    {'do': True, 'policy': 'VFA', 'name': 'VFA (opt - harmonic)'},
    {'do': False, 'policy': 'VFA', 'name': 'VFA (opt - constant)'},
    {'do': True, 'policy': 'VFA', 'name': 'VFA (opt - harmonic0.5)'},

    # Q-Table
    {'do': True, 'policy': 'Q-Table', 'name': 'Q-Table (standard)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=1)'},
    {'do': True, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=100)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (target, update frequency=inf)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=1)'},
    {'do': True, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=100)'},
    {'do': False, 'policy': 'Q-Table', 'name': 'Q-Table (double, update frequency=inf)'},

    # DQN
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard, h=128)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard, h=256)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard lr=0.001, h=128)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (standard lr=0.0005, h=128)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=1)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target, update frequency=100)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (target, update frequency=inf)'},
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=1)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double, update frequency=100)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double, update frequency=1000)'},#test
    #{'do': False, 'policy': 'DQN', 'name': 'DQN (double, decay0.995)'}, #testxxx
    {'do': False, 'policy': 'DQN', 'name': 'DQN (double, update frequency=inf)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target + soft update, tau=0.001, h=128)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target + soft update, tau=0.001, h=256)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target + soft update, tau=0.001, h=128, v2)'},#test
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target + soft update, tau=0.001, h=256, v2)'},#test

    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, lr=0.0005, h=128)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, lr=0.001, h=128)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, lr=0.0005, h=256)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, lr=0.001, h=256)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, v1)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, v2)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (target + soft update, tau=0.001, 0.5)'},
    {'do': True, 'policy': 'DQN', 'name': 'DQN (double + soft update, tau=0.001, 0.5)'},
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


