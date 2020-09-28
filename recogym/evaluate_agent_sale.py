import multiprocessing
import time
from copy import deepcopy
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

import recogym
from recogym import (
    AgentInit,
    AgentStats,
    Configuration,
    EvolutionCase,
    RoiMetrics,
    TrainingApproach
)
from recogym.agents import EpsilonGreedy, epsilon_greedy_args
from .envs.context import DefaultContext
from .envs.observation import Observation
from .envs.session import OrganicSessions
from recogym.envs.reco_env_v1_sale import ff, sig
from recogym.evaluate_agent import (EpsilonDelta, EpsilonSteps, EpsilonPrecision, EvolutionEpsilons,
                                    GraphCTRMin,GraphCTRMax,
                                    to_categorical,build_agent_init, 
                                    gather_agent_stats, build_agents, generate_epsilons, format_epsilon,
                                    )
import pickle as pkl
from datetime import datetime
from IPython.display import display

from joblib import Parallel, delayed
from recogym.agents.sale_agent import train_agents, train_timeagents
from recogym.envs.utils_sale import format_avg_result_extended, avg_result_extended 


GraphSalesMin = 0.009
GraphSalesMax = 0.021

def evaluate_agent_sale(
        env,
        agent,
        num_initial_train_users=100,
        num_step_users=1000,
        num_steps=10,
        training_approach=TrainingApproach.ALL_DATA,
        sliding_window_samples=10000,seed=0,env_reset = True):
    initial_agent = deepcopy(agent)

    unique_user_id = 0
    for u in range(num_initial_train_users):
        if env_reset == True:
            env.reset(unique_user_id + u + seed)
        agent.reset()
        new_observation, reward, done, info = env.step(None)
        while True:
            old_observation = new_observation
            action, new_observation, reward, done, info = env.step_offline(
                new_observation, reward, False, info
            )
            agent.train(old_observation, action, reward, done)
            if done:
                break
    unique_user_id += num_initial_train_users

    rewards = {
        EvolutionCase.SUCCESS: [],
        EvolutionCase.SUCCESS_GREEDY: [],
        EvolutionCase.FAILURE: [],
        EvolutionCase.FAILURE_GREEDY: [],
        EvolutionCase.ACTIONS: dict(),
        
        EvolutionCase.SUCCESS_click: [],
        EvolutionCase.SUCCESS_GREEDY_click: [],
        EvolutionCase.FAILURE_click: [],
        EvolutionCase.FAILURE_GREEDY_click: [],
        EvolutionCase.ACTIONS_click: dict()
    }
    training_agent = deepcopy(agent)
    samples = 0

    for action_id in range(env.config.num_products):
        rewards[EvolutionCase.ACTIONS][action_id] = [0]

    for step in range(num_steps):
        successes = 0
        successes_greedy = 0
        failures = 0
        failures_greedy = 0
        
        successes_click = 0
        successes_greedy_click = 0
        failures_click = 0
        failures_greedy_click = 0

        for u in range(num_step_users):
            env.reset(unique_user_id + u)
            agent.reset()
            new_observation, reward, done, _ = env.step(None)
            while not done:
                old_observation = new_observation
                action = agent.act(old_observation, reward, done)
                new_observation, reward, done, info = env.step(action['a'])
                samples += 1

                should_update_training_data = False
                if training_approach == TrainingApproach.ALL_DATA or training_approach == TrainingApproach.LAST_STEP:
                    should_update_training_data = True
                elif training_approach == TrainingApproach.SLIDING_WINDOW_ALL_DATA:
                    should_update_training_data = samples % sliding_window_samples == 0
                elif training_approach == TrainingApproach.ALL_EXPLORATION_DATA:
                    should_update_training_data = not action['greedy']
                elif training_approach == TrainingApproach.SLIDING_WINDOW_EXPLORATION_DATA:
                    should_update_training_data = (not action[
                        'greedy']) and samples % sliding_window_samples == 0
                else:
                    assert False, f"Unknown Training Approach: {training_approach}"

                if should_update_training_data:
                    training_agent.train(old_observation, action, reward, done)

                if reward:
                    successes += reward 
                    if 'greedy' in action and action['greedy']:
                        successes_greedy += reward 
                    rewards[EvolutionCase.ACTIONS][action['a']][-1] += reward ##H
                else:
                    if 'greedy' in action and action['greedy']:
                        failures_greedy += 1
                    failures += 1
                    
                if (new_observation.click is not None) and (new_observation.click[len(new_observation.click)-1]["click"] == 1): #modif
                    successes_click += 1
                    if ('greedy' in action) and action['greedy']:
                        successes_greedy_click += 1
                    rewards[EvolutionCase.ACTIONS_click][action['a']][-1] += 1
                else:
                    if ('greedy' in action) and action['greedy']:
                        failures_greedy_click += 1
                    failures_click += 1
                    
        unique_user_id += num_step_users

        agent = training_agent
        for action_id in range(env.config.num_products):
            rewards[EvolutionCase.ACTIONS][action_id].append(0)

        if training_approach == TrainingApproach.LAST_STEP:
            training_agent = deepcopy(initial_agent)
        else:
            training_agent = deepcopy(agent)

        rewards[EvolutionCase.SUCCESS].append(successes)
        rewards[EvolutionCase.SUCCESS_GREEDY].append(successes_greedy)
        rewards[EvolutionCase.FAILURE].append(failures)
        rewards[EvolutionCase.FAILURE_GREEDY].append(failures_greedy)
        
        rewards[EvolutionCase.SUCCESS_click].append(successes_click)
        rewards[EvolutionCase.SUCCESS_GREEDY_click].append(successes_greedy_click)
        rewards[EvolutionCase.FAILURE_click].append(failures_click)
        rewards[EvolutionCase.FAILURE_GREEDY_click].append(failures_greedy_click)

    return rewards





def _collect_evolution_stats_sale(args):
    """
    Function that is executed in a separate process.

    :param args: arguments of the process to be executed.

    :return: a dictionary of Success/Failures of applying an Agent.
    """
    start = time.time()
    epsilon = args['epsilon']
    epsilon_key = format_epsilon(epsilon)
    print(f"START: ε = {epsilon_key}")
    num_evolution_steps = args['num_evolution_steps']
    rewards = recogym.evaluate_agent(
        deepcopy(args['env']),
        args['agent'],
        args['num_initial_train_users'],
        args['num_step_users'],
        num_evolution_steps,
        args['training_approach']
    )

    assert (len(rewards[EvolutionCase.SUCCESS]) == len(rewards[EvolutionCase.FAILURE]))
    assert (len(rewards[EvolutionCase.SUCCESS]) == num_evolution_steps)
    print(f"END: ε = {epsilon_key} ({time.time() - start}s)")

    return {
        epsilon_key: {
            EvolutionCase.SUCCESS: rewards[EvolutionCase.SUCCESS],
            EvolutionCase.SUCCESS_GREEDY: rewards[EvolutionCase.SUCCESS_GREEDY],
            EvolutionCase.FAILURE: rewards[EvolutionCase.FAILURE],
            EvolutionCase.FAILURE_GREEDY: rewards[EvolutionCase.FAILURE_GREEDY],
            EvolutionCase.ACTIONS: rewards[EvolutionCase.ACTIONS],
            
            EvolutionCase.SUCCESS_click: rewards[EvolutionCase.SUCCESS_click],
            EvolutionCase.SUCCESS_GREEDY_click: rewards[EvolutionCase.SUCCESS_GREEDY_click],
            EvolutionCase.FAILURE_click: rewards[EvolutionCase.FAILURE_click],
            EvolutionCase.FAILURE_GREEDY_click: rewards[EvolutionCase.FAILURE_GREEDY_click],
            EvolutionCase.ACTIONS_click: rewards[EvolutionCase.ACTIONS_click]
        }
    }


def gather_exploration_stats_sale(
        env,
        env_args,
        extra_env_args,
        agents_init_data,
        training_approach,
        num_initial_train_users=1000,
        num_step_users=1000,
        epsilons=EvolutionEpsilons,
        num_evolution_steps=6
):
    """
    A helper function that collects data regarding Agents evolution
    under different values of epsilon for Epsilon-Greedy Selection Policy.

    :param env: The Environment where evolution should be applied;
         every time when a new step of the evolution is applied, the Environment is deeply copied
         thus the Environment does not interferes with evolution steps.

    :param env_args: Environment arguments (default ones).
    :param extra_env_args: extra Environment conditions those alter default values.
    :param agents_init_data: Agent initialisation data.
        This is a dictionary that has the following structure:
        {
            '<Agent Name>': {
                AgentInit.CTOR: <Constructor>,
                AgentInit.DEF_ARG: <Default Arguments>,
            }
        }


    :param training_approach:  A training approach applied in verification;
     for mode details look at `TrainingApproach' enum.

    :param num_initial_train_users: how many users' data should be used
     to train an initial model BEFORE evolution steps.

    :param num_step_users: how many users' data should be used
     at each evolution step.

     :param epsilons: a list of epsilon values.

    :param num_evolution_steps: how many evolution steps should be applied
     for an Agent with Epsilon-Greedy Selection Policy.

    :return a dictionary of Agent evolution statistics in the form:
        {
            'Agent Name': {
                'Epsilon Values': {
                    EvolutionCase.SUCCESS: [an array of sales (for each ith step of evolution)]
                    EvolutionCase.FAILURE: [an array of failure to draw a sale (for each ith step of evolution)]
                }
            }
        }
    """
    # A dictionary that stores all data of Agent evolution statistics.
    # Key is Agent Name, value is statistics.
    agent_evolution_stats = dict()

    new_env_args = {
        **env_args,
        **extra_env_args,
    }

    new_env = deepcopy(env)
    new_env.init_gym(new_env_args)

    agents = build_agents(agents_init_data, new_env_args)

    for agent_key in agents:
        print(f"Agent: {agent_key}")
        agent_stats = dict()

        with Pool(processes=multiprocessing.cpu_count()) as pool:
            for result in pool.map(
                    _collect_evolution_stats_sale,
                    [
                        {
                            'epsilon': epsilon,
                            'env': new_env,
                            'agent': EpsilonGreedy(
                                Configuration({
                                    **epsilon_greedy_args,
                                    **new_env_args,
                                    'epsilon': epsilon,
                                }),
                                deepcopy(agents[agent_key])
                            ),
                            'num_initial_train_users': num_initial_train_users,
                            'num_step_users': num_step_users,
                            'num_evolution_steps': num_evolution_steps,
                            'training_approach': training_approach,
                        }
                        for epsilon in epsilons
                    ]
            ):
                agent_stats = {
                    **agent_stats,
                    **result,
                }

        agent_evolution_stats[agent_key] = agent_stats

    return agent_evolution_stats


def plot_agent_stats_sale(agent_stats): ##H
    _, (ax,ax_click) = plt.subplots(
        2,
        1,
        figsize=(16, 16)
    )

    user_samples = agent_stats[AgentStats.SAMPLES]
    for agent_key in agent_stats[AgentStats.AGENTS]:
        stats = agent_stats[AgentStats.AGENTS][agent_key]

        # Plot sales
        ax.fill_between(
            user_samples,
            stats[AgentStats.Q0_975],
            stats[AgentStats.Q0_025],
            alpha=.05
        )

        ax.plot(user_samples, stats[AgentStats.Q0_500])

        ax.set_xlabel('Samples #')
        ax.set_ylabel("Sales") 
        ax.legend([
            "$C^{Sales}_{0.5}$: " + f"{agent_key}" for agent_key in agent_stats[AgentStats.AGENTS]
        ])
        
        # Plot clicks
        ax_click.fill_between(
            user_samples,
            stats[AgentStats.Q0_975_click],
            stats[AgentStats.Q0_025_click],
            alpha=.05
        )

        ax_click.plot(user_samples, stats[AgentStats.Q0_500_click])

        ax_click.set_xlabel('Samples #')
        ax_click.set_ylabel("CTR") 
        ax_click.legend([ 
            "$C^{CTR}_{0.5}$: " + f"{agent_key}" for agent_key in agent_stats[AgentStats.AGENTS]
        ])
    plt.show()
    
    

def plot_evolution_stats(
        agent_evolution_stats,
        max_agents_per_row=2,
        epsilons=EvolutionEpsilons,
        plot_min=GraphSalesMin,
        plot_max=GraphSalesMax
):
    figs, axs = plt.subplots(
        2*int(len(agent_evolution_stats) / max_agents_per_row),
        max_agents_per_row,
        figsize=(16, 10),
        squeeze=False
    )
    labels = [("$\epsilon=$" + format_epsilon(epsilon)) for epsilon in epsilons]

    for (ix, agent_key) in enumerate(agent_evolution_stats):
        ax = axs[2*int(ix / max_agents_per_row), int(ix % max_agents_per_row)]
        ax_click = axs[(2*int(ix / max_agents_per_row))+1, int(ix % max_agents_per_row)]
        agent_evolution_stat = agent_evolution_stats[agent_key]

        click_means = []
        sales_means = []
        for epsilon in epsilons:
            epsilon_key = format_epsilon(epsilon)
            evolution_stat = agent_evolution_stat[epsilon_key]

            steps = []
            ms = []
            q0_025 = []
            q0_975 = []
            
            ms_click = []
            q0_025_click = []
            q0_975_click = []

            assert (len(evolution_stat[EvolutionCase.SUCCESS]) == len(
                evolution_stat[EvolutionCase.FAILURE]))
            for step in range(len(evolution_stat[EvolutionCase.SUCCESS])):
                steps.append(step)
                successes = evolution_stat[EvolutionCase.SUCCESS][step]
                failures = evolution_stat[EvolutionCase.FAILURE][step]

                ms.append(beta.ppf(0.5, successes + 1, failures + 1))
                q0_025.append(beta.ppf(0.025, successes + 1, failures + 1))
                q0_975.append(beta.ppf(0.975, successes + 1, failures + 1))
                
                successes_click = evolution_stat[EvolutionCase.SUCCESS_click][step]
                failures_click = evolution_stat[EvolutionCase.FAILURE_click][step]

                ms_click.append(beta.ppf(0.5, successes_click + 1, failures_click + 1))
                q0_025_click.append(beta.ppf(0.025, successes_click + 1, failures_click + 1))
                q0_975_click.append(beta.ppf(0.975, successes_click + 1, failures_click + 1))

            sales_means.append(np.mean(ms))
            click_means.append(np.mean(ms_click))

            ax.fill_between(
                range(len(steps)),
                q0_975,
                q0_025,
                alpha=.05
            )
            ax.plot(steps, ms)
            
            ax_click.fill_between(
                range(len(steps)),
                q0_975_click,
                q0_025_click,
                alpha=.05
            )
            ax_click.plot(steps, ms_click)

        sales_means_mean = np.mean(sales_means)
        sales_means_div = np.sqrt(np.var(sales_means))
        ax.set_title(
            f"Agent: {agent_key}\n"
            + "$\hat{Q}^{sales}_{0.5}="
            + "{0:.5f}".format(round(sales_means_mean, 5))
            + "$, "
            + "$\hat{\sigma}^{sales}_{0.5}="
            + "{0:.5f}".format(round(sales_means_div, 5))
            + "$"
        )
        ax.legend(labels)
        ax.set_ylabel('Sales')
        ax.set_ylim([plot_min, plot_max])
        
        click_means_mean = np.mean(click_means)
        click_means_div = np.sqrt(np.var(click_means))
        ax_click.set_title(
            f"Agent: {agent_key}\n"
            + "$\hat{Q}^{ctr}_{0.5}="
            + "{0:.5f}".format(round(click_means_mean, 5))
            + "$, "
            + "$\hat{\sigma}^{click}_{0.5}="
            + "{0:.5f}".format(round(click_means_div, 5))
            + "$"
        )
        ax_click.legend(labels)
        ax_click.set_ylabel('CTR')
        ax_click.set_ylim([plot_min, plot_max])

    plt.subplots_adjust(hspace=.5)
    plt.show()


def plot_heat_actions_sale(
        agent_evolution_stats,
        epsilons=EvolutionEpsilons
):
    max_epsilons_per_row = len(epsilons)
    the_first_agent = next(iter(agent_evolution_stats.values()))
    epsilon_steps = len(the_first_agent)
    rows = int(len(agent_evolution_stats) * epsilon_steps / max_epsilons_per_row)
    figs, axs = plt.subplots(
        2*int(len(agent_evolution_stats) * epsilon_steps / max_epsilons_per_row),
        max_epsilons_per_row,
        figsize=(16, 4 * rows),
        squeeze=False
    )

    for (ix, agent_key) in enumerate(agent_evolution_stats):
        agent_evolution_stat = agent_evolution_stats[agent_key]
        for (jx, epsilon_key) in enumerate(agent_evolution_stat):
            flat_index = ix * epsilon_steps + jx
            ax = axs[int(flat_index / max_epsilons_per_row), int(flat_index % max_epsilons_per_row)]
            
            evolution_stat = agent_evolution_stat[epsilon_key]

            action_stats = evolution_stat[EvolutionCase.ACTIONS]
            total_actions = len(action_stats)
            heat_data = []
            for kx in range(total_actions):
                heat_data.append(action_stats[kx])

            heat_data = np.array(heat_data)
            im = ax.imshow(heat_data)

            ax.set_yticks(np.arange(total_actions))
            ax.set_yticklabels([f"{action_id}" for action_id in range(total_actions)])

            ax.set_title(f"Agent: {agent_key}\n$\epsilon=${epsilon_key}")

            _ = ax.figure.colorbar(im, ax=ax)

    plt.show()


def plot_roi(
        agent_evolution_stats,
        epsilons=EvolutionEpsilons,
        max_agents_per_row=2,
        sales = True
):
    """
    A helper function that calculates Return of Investment (ROI) for applying Epsilon-Greedy Selection Policy.

    :param agent_evolution_stats: statistic about Agent evolution collected in `build_exploration_data'.

    :param epsilons: a list of epsilon values.

    :param max_agents_per_row: how many graphs should be drawn per a row

    :return: a dictionary of Agent ROI after applying Epsilon-Greedy Selection Strategy in the following form:
        {
            'Agent Name': {
                'Epsilon Value': {
                    Metrics.ROI: [an array of ROIs for each ith step (starting from 1st step)]
                }
            }
        }
    """
    figs, axs = plt.subplots(
        int(len(agent_evolution_stats) / max_agents_per_row),
        max_agents_per_row,
        figsize=(16, 8),
        squeeze=False
    )
    labels = [("$\epsilon=$" + format_epsilon(epsilon)) for epsilon in epsilons if epsilon != 0.0]

    agent_roi_stats = dict()

    for (ix, agent_key) in enumerate(agent_evolution_stats):
        ax = axs[int(ix / max_agents_per_row), int(ix % max_agents_per_row)]
        agent_stat = agent_evolution_stats[agent_key]
        zero_epsilon_key = format_epsilon(0)
        zero_epsilon = agent_stat[zero_epsilon_key]
        if sales == True :
            zero_success_evolutions = zero_epsilon[EvolutionCase.SUCCESS]
            zero_failure_evolutions = zero_epsilon[EvolutionCase.FAILURE]
        else :
            zero_success_evolutions = zero_epsilon[EvolutionCase.SUCCESS_click]
        zero_failure_evolutions = zero_epsilon[EvolutionCase.FAILURE_click]
        assert (len(zero_success_evolutions))

        agent_stats = dict()
        roi_mean_means = []
        for epsilon in generate_epsilons():
            if zero_epsilon_key == format_epsilon(epsilon):
                continue

            epsilon_key = format_epsilon(epsilon)
            agent_stats[epsilon_key] = {
                RoiMetrics.ROI_0_025: [],
                RoiMetrics.ROI_MEAN: [],
                RoiMetrics.ROI_0_975: [],
            }
            epsilon_evolutions = agent_stat[epsilon_key]
            if sales == True : 
                success_greedy_evolutions = epsilon_evolutions[EvolutionCase.SUCCESS_GREEDY]
                failure_greedy_evolutions = epsilon_evolutions[EvolutionCase.FAILURE_GREEDY]
            else :
                success_greedy_evolutions = epsilon_evolutions[EvolutionCase.SUCCESS_GREEDY_click]
                failure_greedy_evolutions = epsilon_evolutions[EvolutionCase.FAILURE_GREEDY_click]
            assert (len(success_greedy_evolutions) == len(failure_greedy_evolutions))
            assert (len(zero_success_evolutions) == len(success_greedy_evolutions))
            steps = []
            roi_means = []
            for step in range(1, len(epsilon_evolutions[EvolutionCase.SUCCESS])):
                previous_zero_successes = zero_success_evolutions[step - 1]
                previous_zero_failures = zero_failure_evolutions[step - 1]
                current_zero_successes = zero_success_evolutions[step]
                current_zero_failures = zero_failure_evolutions[step]
                current_epsilon_greedy_successes = success_greedy_evolutions[step]
                current_epsilon_greedy_failures = failure_greedy_evolutions[step]

                def roi_with_confidence_interval(
                        epsilon,
                        previous_zero_successes,
                        previous_zero_failures,
                        current_zero_successes,
                        current_zero_failures,
                        current_epsilon_greedy_successes,
                        current_epsilon_greedy_failures
                ):
                    def roi_formulae(
                            epsilon,
                            previous_zero,
                            current_zero,
                            current_epsilon_greedy
                    ):
                        current_gain = current_epsilon_greedy / (1 - epsilon) - current_zero
                        roi = current_gain / (epsilon * previous_zero)
                        return roi

                    return {
                        RoiMetrics.ROI_SUCCESS: roi_formulae(
                            epsilon,
                            previous_zero_successes,
                            current_zero_successes,
                            current_epsilon_greedy_successes
                        ),
                        RoiMetrics.ROI_FAILURE: roi_formulae(
                            epsilon,
                            previous_zero_failures,
                            current_zero_failures,
                            current_epsilon_greedy_failures
                        )
                    }

                roi_mean = roi_with_confidence_interval(
                    epsilon,
                    previous_zero_successes,
                    previous_zero_failures,
                    current_zero_successes,
                    current_zero_failures,
                    current_epsilon_greedy_successes,
                    current_epsilon_greedy_failures
                )[RoiMetrics.ROI_SUCCESS]
                agent_stats[epsilon_key][RoiMetrics.ROI_MEAN].append(roi_mean)

                roi_means.append(roi_mean)

                steps.append(step)

            roi_mean_means.append(np.mean(roi_means))
            ax.plot(steps, roi_means)

        roi_means_mean = np.mean(roi_mean_means)
        roi_means_div = np.sqrt(np.var(roi_mean_means))
        ax.set_title(
            "$ROI_{t+1}$ of Agent: " + f"'{agent_key}'\n"
            + "$\hat{\mu}_{ROI}="
            + "{0:.5f}".format(round(roi_means_mean, 5))
            + "$, "
            + "$\hat{\sigma}_{ROI}="
            + "{0:.5f}".format(round(roi_means_div, 5))
            + "$"
        )
        ax.legend(labels, loc=10)
        ax.set_ylabel('ROI')

        agent_roi_stats[agent_key] = agent_stats

    plt.subplots_adjust(hspace=.5)
    plt.show()
    return agent_roi_stats


def verify_agents_sale(env, number_of_users, agents, agent_reset = False, name = '',seed=0,
                       same_env = True, repo = 'data/'): ##H
    stat = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
    
    stat_click = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
    
    stat_tot_sales_att = {
        'Agent': [],
        'TotSalesAtt':[]}

    
    stat_share_user_sale_att = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
    
    stat_tot_sales = {
        'Agent': [],
        'TotSales':[]}
    
    stat_share_user_sale = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
    
    stat_sale_after_click  = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
        

    data_list={}
    embed_list={}
    config_list={}
    all_data = {}
    
    # One success is defined as at least 1 sale
    for agent_id in agents:
        try :
            if same_env:
                envcopy = deepcopy(env)
                envcopy.reset(seed)
            else:
                envcopy = env
            if agent_reset == True :
                agents[agent_id].reset()
            
            data = envcopy.generate_logs(number_of_users, agents[agent_id])
            try :
                report_issue = agents[agent_id].report_issue
            except:
                report_issue = {}
            # print("Min dot product",[np.min(envcopy.list_dot_products[i]) for i in range(len(envcopy.list_dot_products))])
            # print("Max dot product",[np.max(envcopy.list_dot_products[i]) for i in range(len(envcopy.list_dot_products))])
            
            all_data[agent_id]= data
            data_list[agent_id] = data.loc[data["a"] >= 0].loc[data["c"] > 0]
            embed_list[agent_id] = envcopy.user_embedding_list
            config_list[agent_id] = {"beta":env.beta,"Lambda":envcopy.Lambda,"Gamma":envcopy.Gamma,"psale_scale":envcopy.config.psale_scale}
            bandits = data[data['z'] == 'bandit']
            
    
            # clicks
            successes_click = np.sum(bandits['c'] > 0)
            failures_click = bandits[bandits['c'] == 0].shape[0]
            stat_click['Agent'].append(agent_id)
            stat_click['0.025'].append(beta.ppf(0.025, successes_click + 1, failures_click + 1))
            stat_click['0.500'].append(beta.ppf(0.500, successes_click + 1, failures_click + 1))
            stat_click['0.975'].append(beta.ppf(0.975, successes_click + 1, failures_click + 1))
            
            
            # total number of ATTRIBUTED sales
            stat_tot_sales_att['Agent'].append(agent_id)
            stat_tot_sales_att['TotSalesAtt'].append(np.sum(data['r']>0))
            
            # share of users who bought something (ATTRIBUTED)
            stat_share_user_sale_att['Agent'].append(agent_id)
            grouped_data = data[["r","u"]].groupby("u").sum()["r"]
            successes_user_sale = sum(grouped_data>0)
            failures_user_sale = len(data["u"].unique()) - successes_user_sale
            stat_share_user_sale_att['0.025'].append(beta.ppf(0.025, successes_user_sale + 1, failures_user_sale + 1))
            stat_share_user_sale_att['0.500'].append(beta.ppf(0.500, successes_user_sale + 1, failures_user_sale + 1))
            stat_share_user_sale_att['0.975'].append(beta.ppf(0.975, successes_user_sale + 1, failures_user_sale + 1))
            
             # total number of sales
            stat_tot_sales['Agent'].append(agent_id)
            stat_tot_sales['TotSales'].append(np.sum(data['z']=='sale'))
            
            # share of users who bought something
            stat_share_user_sale['Agent'].append(agent_id)
            data["sale"] = (data['z']=="sale")*1
            grouped_data = data[["sale","u"]].groupby("u").sum()["sale"]
            successes_user_sale = sum(grouped_data>0)
            failures_user_sale = len(data["u"].unique()) - successes_user_sale
            stat_share_user_sale['0.025'].append(beta.ppf(0.025, successes_user_sale + 1, failures_user_sale + 1))
            stat_share_user_sale['0.500'].append(beta.ppf(0.500, successes_user_sale + 1, failures_user_sale + 1))
            stat_share_user_sale['0.975'].append(beta.ppf(0.975, successes_user_sale + 1, failures_user_sale + 1))
            
            
            # sales rate
            successes = np.sum(bandits['r'] > 0)
            failures = bandits[bandits['r'] == 0].shape[0]
            stat['Agent'].append(agent_id)
            stat['0.025'].append(beta.ppf(0.025, successes + 1, failures + 1))
            stat['0.500'].append(beta.ppf(0.500, successes + 1, failures + 1))
            stat['0.975'].append(beta.ppf(0.975, successes + 1, failures + 1))
           
            
            # share of sales after a click
            stat_sale_after_click['Agent'].append(agent_id)
            successes_sale_after_click = len(data[(data["c"]==1) & (data["r"]>0)])
            failures_sale_after_click = sum(data["c"]==1)-successes_sale_after_click
            stat_sale_after_click['0.025'].append(beta.ppf(0.025, successes_sale_after_click + 1, failures_sale_after_click + 1))
            stat_sale_after_click['0.500'].append(beta.ppf(0.500, successes_sale_after_click + 1, failures_sale_after_click + 1))
            stat_sale_after_click['0.975'].append(beta.ppf(0.975, successes_sale_after_click + 1, failures_sale_after_click + 1))
            
            # save intermediate result
            agent_dico = {'CTR': pd.DataFrame().from_dict(stat_click), 
                            'Tot sales ATT': pd.DataFrame().from_dict(stat_tot_sales_att),
                            'Share user with sale ATT': pd.DataFrame().from_dict(stat_share_user_sale_att), 
                            'Tot sales': pd.DataFrame().from_dict(stat_tot_sales),
                            'Share user with sale': pd.DataFrame().from_dict(stat_share_user_sale), 
                            'sale rate': pd.DataFrame().from_dict(stat), 
                            'Share sale after click': pd.DataFrame().from_dict(stat_sale_after_click),
                            # only save info of the current agent to save space
                            # 'User embeddings':embed_list[agent_id],
                            # 'reco':data_list[agent_id],
                            'config_list':config_list[agent_id],
                            'all_data':all_data[agent_id],
                            'report_issue':report_issue}
            pkl.dump(agent_dico, open(str(repo)+'res_'+name+agent_id+'.pkl',"wb"))
            
            
        except Exception as e:
            print("Issue with agent : ",agent_id)
            print('exception:',e)
            dico = {'CTR': pd.DataFrame().from_dict(stat_click), 
            'Tot sales ATT': pd.DataFrame().from_dict(stat_tot_sales_att),
            'Share user with sale ATT': pd.DataFrame().from_dict(stat_share_user_sale_att), 
            'Tot sales': pd.DataFrame().from_dict(stat_tot_sales),
            'Share user with sale': pd.DataFrame().from_dict(stat_share_user_sale), 
            'sale rate': pd.DataFrame().from_dict(stat), 
            'Share sale after click': pd.DataFrame().from_dict(stat_sale_after_click),
            # 'User embeddings':embed_list,
            # 'reco':data_list,
            'config_list':config_list,
            'all_data':all_data,
            'report_issue':report_issue}
            pkl.dump(dico, open(str(repo)+'res_before_crash'+name+str(int(datetime.timestamp(datetime.now())))+'.pkl',"wb"))
            
    return {'CTR': pd.DataFrame().from_dict(stat_click), 
            'Tot sales ATT': pd.DataFrame().from_dict(stat_tot_sales_att),
            'Share user with sale ATT': pd.DataFrame().from_dict(stat_share_user_sale_att), 
            'Tot sales': pd.DataFrame().from_dict(stat_tot_sales),
            'Share user with sale': pd.DataFrame().from_dict(stat_share_user_sale), 
            'sale rate': pd.DataFrame().from_dict(stat), 
            'Share sale after click': pd.DataFrame().from_dict(stat_sale_after_click),
            # 'User embeddings':embed_list,
            # 'reco':data_list,
            'config_list':config_list,
            'all_data':all_data,
            'report_issue':report_issue}




def evaluate_IPS_sale(agent, reco_log):
    ee = []
    ee_click = []
    for u in range(max(reco_log.u)):
        t = np.array(reco_log[reco_log['u'] == u].t)
        v = np.array(reco_log[reco_log['u'] == u].v)
        a = np.array(reco_log[reco_log['u'] == u].a)
        c = np.array(reco_log[reco_log['u'] == u].c)
        r = np.array(reco_log[reco_log['u'] == u].r)
        z = list(reco_log[reco_log['u'] == u].z)
        ps = np.array(reco_log[reco_log['u'] == u].ps)

        jj = 0

        session = OrganicSessions()
        agent.reset()
        while True:
            if jj >= len(z):
                break
            if z[jj] == 'organic':
                session.next(DefaultContext(t[jj], u), int(v[jj]))
            else:
                prob_policy = agent.act(Observation(DefaultContext(t[jj], u), session), 0, False)[ #modif
                    'ps-a']
                
                if prob_policy!=():
                    ee.append(r[jj] * prob_policy[int(a[jj])] / ps[jj]) ##H
                    ee_click.append(c[jj] * prob_policy[int(a[jj])] / ps[jj]) ##H
                session = OrganicSessions()
            jj += 1
    return ee, ee_click


def evaluate_SNIPS_sale(agent, reco_log):
    rewards = []
    rewards_click = []
    p_ratio = []
    for u in range(max(reco_log.u)):
        t = np.array(reco_log[reco_log['u'] == u].t)
        v = np.array(reco_log[reco_log['u'] == u].v)
        a = np.array(reco_log[reco_log['u'] == u].a)
        c = np.array(reco_log[reco_log['u'] == u].c) ##H
        r = np.array(reco_log[reco_log['u'] == u].r) ##H
        z = list(reco_log[reco_log['u'] == u].z)
        ps = np.array(reco_log[reco_log['u'] == u].ps)

        jj = 0

        session = OrganicSessions()
        agent.reset()
        while True:
            if jj >= len(z):
                break
            if z[jj] == 'organic':
                session.next(DefaultContext(t[jj], u), int(v[jj]))
            else:
                prob_policy = agent.act(Observation(DefaultContext(t[jj], u), session), 0, False)[ #modif
                    'ps-a']
                rewards.append(r[jj]) 
                rewards_click.append(c[jj])
                p_ratio.append(prob_policy[int(a[jj])] / ps[jj])
                session = OrganicSessions()
            jj += 1
    return rewards, p_ratio


def verify_agents_IPS_sale(reco_log, agents):
    stat = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
    
    stat_click = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }

    for agent_id in agents:
        ee, ee_click = evaluate_IPS_sale(agents[agent_id], reco_log)
        mean_ee, mean_ee_click = np.mean(ee), np.mean(ee_click)
        se_ee, se_ee_click = np.std(ee) / np.sqrt(len(ee)), np.std(ee_click) / np.sqrt(len(ee_click))
        stat['Agent'].append(agent_id)
        stat['0.025'].append(mean_ee - 2 * se_ee)
        stat['0.500'].append(mean_ee)
        stat['0.975'].append(mean_ee + 2 * se_ee)
        
        stat_click['Agent'].append(agent_id)
        stat_click['0.025'].append(mean_ee_click - 2 * se_ee_click)
        stat_click['0.500'].append(mean_ee_click)
        stat_click['0.975'].append(mean_ee_click + 2 * se_ee_click)
        
    return pd.DataFrame().from_dict(stat), pd.DataFrame().from_dict(stat_click)


def verify_agents_SNIPS_click(reco_log, agents):
    stat = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }

    stat_click = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }

    for agent_id in agents:
        ee, ee_click = evaluate_SNIPS_sale(agents[agent_id], reco_log)
        mean_ee, mean_ee_click = np.mean(ee), np.mean(ee_click)
        se_ee, se_ee_click = np.std(ee) / np.sqrt(len(ee)), np.std(ee_click) / np.sqrt(len(ee_click))
        stat['Agent'].append(agent_id)
        stat['0.025'].append(mean_ee - 2 * se_ee)
        stat['0.500'].append(mean_ee)
        stat['0.975'].append(mean_ee + 2 * se_ee)
        
        stat_click['Agent'].append(agent_id)
        stat_click['0.025'].append(mean_ee_click - 2 * se_ee_click)
        stat_click['0.500'].append(mean_ee_click)
        stat_click['0.975'].append(mean_ee_click + 2 * se_ee_click)
        
    return pd.DataFrame().from_dict(stat), pd.DataFrame().from_dict(stat_click)


def evaluate_recall_at_k_sale(agent, reco_log, k=5):
    hits = []
    hits_click = []
    for u in range(max(reco_log.u)):
        t = np.array(reco_log[reco_log['u'] == u].t)
        v = np.array(reco_log[reco_log['u'] == u].v)
        a = np.array(reco_log[reco_log['u'] == u].a)
        c = np.array(reco_log[reco_log['u'] == u].c) ##H
        r = np.array(reco_log[reco_log['u'] == u].r) ##H
        z = list(reco_log[reco_log['u'] == u].z)
        ps = np.array(reco_log[reco_log['u'] == u].ps)

        jj = 0

        session = OrganicSessions()
        agent.reset()
        while True:
            if jj >= len(z):
                break
            if z[jj] == 'organic':
                session.next(DefaultContext(t[jj], u), int(v[jj]))
            else:
                prob_policy = agent.act(Observation(DefaultContext(t[jj], u), session), 0, False)[ #modif
                    'ps-a']
                # Does the next session exist?
                if (jj + 1) < len(z):
                    # Is the next session organic?
                    if z[jj + 1] == 'organic':
                        # Whas there no sale for this bandit event?
                        if not r[jj]: ##H
                            # Generate a top-K from the probability distribution over all actions
                            top_k = set(np.argpartition(prob_policy, -k)[-k:])
                            # Is the next seen item in the top-K?
                            if v[jj + 1] in top_k:
                                hits.append(1)
                            else:
                                hits.append(0)
                        if not c[jj]: ##H
                            # Generate a top-K from the probability distribution over all actions
                            top_k = set(np.argpartition(prob_policy, -k)[-k:])
                            # Is the next seen item in the top-K?
                            if v[jj + 1] in top_k:
                                hits_click.append(1)
                            else:
                                hits_click.append(0)
                session = OrganicSessions()
            jj += 1
    return hits, hits_click


def verify_agents_recall_at_k_sale(reco_log, agents, k=5):
    stat = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
    
    stat_click = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }

    for agent_id in agents:
        hits, hits_click = evaluate_recall_at_k_sale(agents[agent_id], reco_log, k=k)
        mean_hits = np.mean(hits)
        se_hits = np.std(hits) / np.sqrt(len(hits))
        stat['Agent'].append(agent_id)
        stat['0.025'].append(mean_hits - 2 * se_hits)
        stat['0.500'].append(mean_hits)
        stat['0.975'].append(mean_hits + 2 * se_hits)
        mean_hits_click = np.mean(hits_click)
        se_hits_click = np.std(hits_click) / np.sqrt(len(hits_click))
        stat_click['Agent'].append(agent_id)
        stat_click['0.025'].append(mean_hits_click - 2 * se_hits_click)
        stat_click['0.500'].append(mean_hits_click)
        stat_click['0.975'].append(mean_hits_click + 2 * se_hits_click)
        
    return pd.DataFrame().from_dict(stat), pd.DataFrame().from_dict(stat_click)


def plot_verify_agents_sale(result_clicks, result_tot_sales_att, result_share_user_sale_att,
                            result_tot_sales, result_share_user_sale, agent_names = None):

    
    names = agent_names if agent_names is not None else result_clicks['Agent']
    # CTR
    fig1, ax = plt.subplots()
    ax.set_title('CTR estimates')
    plt.errorbar(result_clicks['Agent'],
                 result_clicks['0.500'],
                 yerr=(result_clicks['0.500'] - result_clicks['0.025'],
                       result_clicks['0.975'] - result_clicks['0.500']),
                 fmt='o',
                 capsize=4)
    # plt.xticks(result_clicks['Agent'], result_clicks['Agent'], rotation='vertical')
    plt.xticks(result_clicks['Agent'], names, rotation='vertical')
    
    
     # Total sales ATT
    fig5, ax = plt.subplots()
    ax.set_title('Total Attributed sales')
    plt.scatter(result_tot_sales_att['Agent'],
                 result_tot_sales_att['TotSalesAtt'])
#     plt.xticks(result_tot_sales_att['Agent'], result_tot_sales_att['Agent'], rotation='vertical')
    plt.xticks(result_tot_sales_att['Agent'], names, rotation='vertical')
    
    # Share of users with sale ATT
    fig6, ax = plt.subplots()
    ax.set_title('Share of users with Attributed sales (ACR)')
    plt.errorbar(result_share_user_sale_att['Agent'],
                 result_share_user_sale_att['0.500'],
                 yerr=(result_share_user_sale_att['0.500'] - result_share_user_sale_att['0.025'],
                       result_share_user_sale_att['0.975'] - result_share_user_sale_att['0.500']),
                 fmt='o',
                 capsize=4)
#     plt.xticks(result_share_user_sale_att['Agent'], result_share_user_sale_att['Agent'], rotation='vertical')
    plt.xticks(result_share_user_sale_att['Agent'], names, rotation='vertical')
    
    
    # Share of clicks with sales ATT
    fig7, ax = plt.subplots()
    ax.set_title('Attributed CR (ACR) as a function of CTR')
    for i in range(len(result_clicks['0.500'])):
        plt.errorbar(result_clicks.iloc[i]['0.500'],
                     result_share_user_sale_att.iloc[i]['0.500'],
                     xerr=result_clicks.iloc[i]['0.500'] - result_clicks.iloc[i]['0.025'],
                     yerr=result_share_user_sale_att.iloc[i]['0.500'] - result_share_user_sale_att.iloc[i]['0.025'],
                     fmt='o',
                     capsize=4, 
#                      label = result_clicks.iloc[i]["Agent"])
                     label = names[i])
    ax.legend()
    
    
    # Total sales
    fig2, ax = plt.subplots()
    ax.set_title('Total sales')
    plt.scatter(result_tot_sales['Agent'],
                 result_tot_sales['TotSales'])
#     plt.xticks(result_tot_sales['Agent'], result_tot_sales['Agent'], rotation='vertical')
    plt.xticks(result_tot_sales['Agent'], names, rotation='vertical')
    
    
    # Share of users with sale
    fig3, ax = plt.subplots()
    ax.set_title('Share of users with sales')
    plt.errorbar(result_share_user_sale['Agent'],
                 result_share_user_sale['0.500'],
                 yerr=(result_share_user_sale['0.500'] - result_share_user_sale['0.025'],
                       result_share_user_sale['0.975'] - result_share_user_sale['0.500']),
                 fmt='o',
                 capsize=4)
#     plt.xticks(result_share_user_sale['Agent'], result_share_user_sale['Agent'], rotation='vertical')
    plt.xticks(result_share_user_sale['Agent'], names, rotation='vertical')
    
    
    # Share of clicks with sales
    fig4, ax = plt.subplots()
    ax.set_title('CR as a function of CTR')
    for i in range(len(result_clicks['0.500'])):
        plt.errorbar(result_clicks.iloc[i]['0.500'],
                     result_share_user_sale.iloc[i]['0.500'],
                     xerr=result_clicks.iloc[i]['0.500'] - result_clicks.iloc[i]['0.025'],
                     yerr=result_share_user_sale.iloc[i]['0.500'] - result_share_user_sale.iloc[i]['0.025'],
                     fmt='o',
                     capsize=4, 
#                      label = result_clicks.iloc[i]["Agent"])
                     label = names[i])
    ax.legend()
    
    return fig1, fig5, fig6, fig7, fig2, fig3, fig4



def plot_CR_CTR(agent,result):
    ''' Plot conversion rate as a function of the CTR based on verify_agent results, and agent name'''
    
    # Load env infos
    beta = result['config_list'][agent]['beta']
    psale_scale = result['config_list'][agent]['psale_scale']
    Lambda = result['config_list'][agent]['Lambda']
    
    # Get user embeddings and list of taken actions
    embeddings = [result["User embeddings"][agent][i]["init"] for i in result["reco"][agent]["u"]]
    actions = list(result["reco"][agent]["a"])

    # plot the embeddings dot product
    plt.scatter([embeddings[i][:,0]@beta[actions[i],:] for i in range(len(embeddings))],
                [embeddings[i][:,0]@Lambda[actions[i],:] for i in range(len(embeddings))], 
                alpha=0.3, label='init')

    embeddings_end = [result["User embeddings"][agent][i]["end"] for i in result["reco"][agent]["u"]]
    actions = list(result["reco"][agent]["a"])
    
    plt.scatter([embeddings_end[i][:,0]@beta[actions[i],:] for i in range(len(embeddings_end))],
                [embeddings_end[i][:,0]@Lambda[actions[i],:] for i in range(len(embeddings_end))], 
                alpha=0.3, label='end')
    plt.xlabel("CTR embedding")
    plt.ylabel("CR embedding")
    plt.title("Features of CR as a function of features of CTR with "+agent+" agent")
    plt.legend()
    plt.show()

    # Plot the probas after embedding dot product transformation
    plt.scatter([ff(embeddings[i][:,0]@beta[actions[i],:]) for i in range(len(embeddings))],
                [psale_scale*sig(embeddings[i][:,0]@Lambda[actions[i],:]) for i in range(len(embeddings))], 
                alpha=0.3, label='init')
    
    plt.scatter([ff(embeddings_end[i][:,0]@beta[actions[i],:]) for i in range(len(embeddings_end))],
                [psale_scale*sig(embeddings_end[i][:,0]@Lambda[actions[i],:]) for i in range(len(embeddings_end))], 
                alpha=0.3, label='end')
    plt.xlabel("CTR")
    plt.ylabel("CR")
    plt.title("CR as a function of CTR with "+agent+" agent")
    plt.legend()
    plt.show()


list_metrics = ['CTR', 'Tot sales ATT', 'Share user with sale ATT', 'Tot sales', 'Share user with sale']
def display_metrics(res,metrics = list_metrics):
    for m in list_metrics:
        res[m].index = range(len(res[m]))
        print('-----'+m+'-----')
        display(res[m].style.background_gradient(cmap='viridis'))
        



def verify_agents_sale_extended(env, number_of_users, agents, agent_reset = False, name = '',seed=0,
                       same_env = True, repo = 'data/'): ##H
    stat = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
    
    stat_click = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
    
    stat_tot_sales_att = {
        'Agent': [],
        'TotSalesAtt':[]}

    
    stat_share_user_sale_att = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
    
    stat_tot_sales = {
        'Agent': [],
        'TotSales':[]}
    
    stat_share_user_sale = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
    
    stat_sale_after_click  = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }
        
    true_ctr = {
        'Agent': [],
        'Mean':[],
        'Median':[],
        'Q1':[],
        'Q3':[],
        'Min':[],
        'Max':[]
        }
    
    true_pcs = {
        'Agent': [],
        'Mean':[],
        'Median':[],
        'Q1':[],
        'Q3':[],
        'Min':[],
        'Max':[]
        }
    
    true_os = {
        'Agent': [],
        'Mean':[],
        'Median':[],
        'Q1':[],
        'Q3':[],
        'Min':[],
        'Max':[]
        }
    
    true_ncs = {
        'Agent': [],
        'Mean':[],
        'Median':[],
        'Q1':[],
        'Q3':[],
        'Min':[],
        'Max':[]
        }
    
    data_list={}
    embed_list={}
    config_list={}
    all_data = {}
    
    # One success is defined as at least 1 sale
    for agent_id in agents:
        try :
            if same_env:
                envcopy = deepcopy(env)
                envcopy.reset(seed)
            else:
                envcopy = env
            if agent_reset == True :
                agents[agent_id].reset()
            
            data, data_true  = envcopy.generate_logs_trueprobas(number_of_users, agents[agent_id])
            try :
                report_issue = agents[agent_id].report_issue
            except:
                report_issue = {}
            # print("Min dot product",[np.min(envcopy.list_dot_products[i]) for i in range(len(envcopy.list_dot_products))])
            # print("Max dot product",[np.max(envcopy.list_dot_products[i]) for i in range(len(envcopy.list_dot_products))])
            
            all_data[agent_id]= data
            data_list[agent_id] = data.loc[data["a"] >= 0].loc[data["c"] > 0]
            embed_list[agent_id] = envcopy.user_embedding_list
            config_list[agent_id] = {"beta":env.beta,"Lambda":envcopy.Lambda,"Gamma":envcopy.Gamma,"psale_scale":envcopy.config.psale_scale}
            bandits = data[data['z'] == 'bandit']
            
    
            # clicks
            successes_click = np.sum(bandits['c'] > 0)
            failures_click = bandits[bandits['c'] == 0].shape[0]
            stat_click['Agent'].append(agent_id)
            stat_click['0.025'].append(beta.ppf(0.025, successes_click + 1, failures_click + 1))
            stat_click['0.500'].append(beta.ppf(0.500, successes_click + 1, failures_click + 1))
            stat_click['0.975'].append(beta.ppf(0.975, successes_click + 1, failures_click + 1))
            
            
            # total number of ATTRIBUTED sales
            stat_tot_sales_att['Agent'].append(agent_id)
            stat_tot_sales_att['TotSalesAtt'].append(np.sum(data['r']>0))
            
            # share of users who bought something (ATTRIBUTED)
            stat_share_user_sale_att['Agent'].append(agent_id)
            grouped_data = data[["r","u"]].groupby("u").sum()["r"]
            successes_user_sale = sum(grouped_data>0)
            failures_user_sale = len(data["u"].unique()) - successes_user_sale
            stat_share_user_sale_att['0.025'].append(beta.ppf(0.025, successes_user_sale + 1, failures_user_sale + 1))
            stat_share_user_sale_att['0.500'].append(beta.ppf(0.500, successes_user_sale + 1, failures_user_sale + 1))
            stat_share_user_sale_att['0.975'].append(beta.ppf(0.975, successes_user_sale + 1, failures_user_sale + 1))
            
             # total number of sales
            stat_tot_sales['Agent'].append(agent_id)
            stat_tot_sales['TotSales'].append(np.sum(data['z']=='sale'))
            
            # share of users who bought something
            stat_share_user_sale['Agent'].append(agent_id)
            data["sale"] = (data['z']=="sale")*1
            grouped_data = data[["sale","u"]].groupby("u").sum()["sale"]
            successes_user_sale = sum(grouped_data>0)
            failures_user_sale = len(data["u"].unique()) - successes_user_sale
            stat_share_user_sale['0.025'].append(beta.ppf(0.025, successes_user_sale + 1, failures_user_sale + 1))
            stat_share_user_sale['0.500'].append(beta.ppf(0.500, successes_user_sale + 1, failures_user_sale + 1))
            stat_share_user_sale['0.975'].append(beta.ppf(0.975, successes_user_sale + 1, failures_user_sale + 1))
            
            
            # sales rate
            successes = np.sum(bandits['r'] > 0)
            failures = bandits[bandits['r'] == 0].shape[0]
            stat['Agent'].append(agent_id)
            stat['0.025'].append(beta.ppf(0.025, successes + 1, failures + 1))
            stat['0.500'].append(beta.ppf(0.500, successes + 1, failures + 1))
            stat['0.975'].append(beta.ppf(0.975, successes + 1, failures + 1))
           
            
            # share of sales after a click
            stat_sale_after_click['Agent'].append(agent_id)
            successes_sale_after_click = len(data[(data["c"]==1) & (data["r"]>0)])
            failures_sale_after_click = sum(data["c"]==1)-successes_sale_after_click
            stat_sale_after_click['0.025'].append(beta.ppf(0.025, successes_sale_after_click + 1, failures_sale_after_click + 1))
            stat_sale_after_click['0.500'].append(beta.ppf(0.500, successes_sale_after_click + 1, failures_sale_after_click + 1))
            stat_sale_after_click['0.975'].append(beta.ppf(0.975, successes_sale_after_click + 1, failures_sale_after_click + 1))
            
            
            # true probas
            true_ctr['Agent'].append(agent_id)
            true_ctr['Mean'].append(np.mean(data_true['ctr']))
            true_ctr['Median'].append(np.median(data_true['ctr']))
            true_ctr['Q1'].append(np.quantile(data_true['ctr'], .25))
            true_ctr['Q3'].append(np.quantile(data_true['ctr'], .75))
            true_ctr['Min'].append(np.min(data_true['ctr']))
            true_ctr['Max'].append(np.max(data_true['ctr']))
    
            true_pcs['Agent'].append(agent_id)
            true_pcs['Mean'].append(np.mean(data_true['pcs']))
            true_pcs['Median'].append(np.median(data_true['pcs']))
            true_pcs['Q1'].append(np.quantile(data_true['pcs'], .25))
            true_pcs['Q3'].append(np.quantile(data_true['pcs'], .75))
            true_pcs['Min'].append(np.min(data_true['pcs']))
            true_pcs['Max'].append(np.max(data_true['pcs']))
            
            true_os['Agent'].append(agent_id)
            true_os['Mean'].append(np.mean(data_true['os']))
            true_os['Median'].append(np.median(data_true['os']))
            true_os['Q1'].append(np.quantile(data_true['os'], .25))
            true_os['Q3'].append(np.quantile(data_true['os'], .75))
            true_os['Min'].append(np.min(data_true['os']))
            true_os['Max'].append(np.max(data_true['os']))
            
            true_ncs['Agent'].append(agent_id)
            true_ncs['Mean'].append(np.mean(data_true['ncs']))
            true_ncs['Median'].append(np.median(data_true['ncs']))
            true_ncs['Q1'].append(np.quantile(data_true['ncs'], .25))
            true_ncs['Q3'].append(np.quantile(data_true['ncs'], .75))
            true_ncs['Min'].append(np.min(data_true['ncs']))
            true_ncs['Max'].append(np.max(data_true['ncs']))
            
            
            
            # save intermediate result
            agent_dico = {'CTR': pd.DataFrame().from_dict(stat_click), 
                            'Tot sales ATT': pd.DataFrame().from_dict(stat_tot_sales_att),
                            'Share user with sale ATT': pd.DataFrame().from_dict(stat_share_user_sale_att), 
                            'Tot sales': pd.DataFrame().from_dict(stat_tot_sales),
                            'Share user with sale': pd.DataFrame().from_dict(stat_share_user_sale), 
                            
                            'True CTR': pd.DataFrame().from_dict(true_ctr), 
                            'True PCS': pd.DataFrame().from_dict(true_pcs), 
                            'True OS': pd.DataFrame().from_dict(true_os), 
                            'True NCS': pd.DataFrame().from_dict(true_ncs), 
                            
                            'sale rate': pd.DataFrame().from_dict(stat), 
                            'Share sale after click': pd.DataFrame().from_dict(stat_sale_after_click),
                            # only save info of the current agent to save space
                            # 'User embeddings':embed_list[agent_id],
                            # 'reco':data_list[agent_id],
                            'config_list':config_list[agent_id],
                            'all_data':all_data[agent_id],
                            'report_issue':report_issue}
            pkl.dump(agent_dico, open(str(repo)+'res_'+name+agent_id+'_full.pkl',"wb"))
            
            
        except Exception as e:
            print("Issue with agent : ",agent_id)
            print('exception:',e)
            dico = {'CTR': pd.DataFrame().from_dict(stat_click), 
            'Tot sales ATT': pd.DataFrame().from_dict(stat_tot_sales_att),
            'Share user with sale ATT': pd.DataFrame().from_dict(stat_share_user_sale_att), 
            'Tot sales': pd.DataFrame().from_dict(stat_tot_sales),
            'Share user with sale': pd.DataFrame().from_dict(stat_share_user_sale), 
            
            'True CTR': pd.DataFrame().from_dict(true_ctr), 
            'True PCS': pd.DataFrame().from_dict(true_pcs), 
            'True OS': pd.DataFrame().from_dict(true_os), 
            'True NCS': pd.DataFrame().from_dict(true_ncs), 
            
            'sale rate': pd.DataFrame().from_dict(stat), 
            'Share sale after click': pd.DataFrame().from_dict(stat_sale_after_click),
            # 'User embeddings':embed_list,
            # 'reco':data_list,
            'config_list':config_list,
            'all_data':all_data,
            'report_issue':report_issue}
            pkl.dump(dico, open(str(repo)+'res_before_crash'+name+str(int(datetime.timestamp(datetime.now())))+'_full.pkl',"wb"))
            
    return {'CTR': pd.DataFrame().from_dict(stat_click), 
            'Tot sales ATT': pd.DataFrame().from_dict(stat_tot_sales_att),
            'Share user with sale ATT': pd.DataFrame().from_dict(stat_share_user_sale_att), 
            'Tot sales': pd.DataFrame().from_dict(stat_tot_sales),
            'Share user with sale': pd.DataFrame().from_dict(stat_share_user_sale), 
            'True CTR': pd.DataFrame().from_dict(true_ctr), 
            'True PCS': pd.DataFrame().from_dict(true_pcs), 
            'True OS': pd.DataFrame().from_dict(true_os), 
            'True NCS': pd.DataFrame().from_dict(true_ncs), 
            'sale rate': pd.DataFrame().from_dict(stat), 
            'Share sale after click': pd.DataFrame().from_dict(stat_sale_after_click),
            # 'User embeddings':embed_list,
            # 'reco':data_list,
            'config_list':config_list,
            'all_data':all_data,
            'report_issue':report_issue}

