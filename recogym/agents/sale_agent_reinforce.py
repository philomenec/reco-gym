
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim

import numpy as np

import gym
from gym.wrappers import Monitor
from pprint import pprint
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output

from pathlib import Path
import base64
import pandas as pd
import itertools
import seaborn as sns

class Model(nn.Module):
    def __init__(self, dim_observation, n_actions):
        super(Model, self).__init__()
        
        self.n_actions = n_actions
        self.dim_observation = dim_observation
        
        self.net = nn.Sequential(
            nn.Linear(in_features=self.dim_observation, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.n_actions),
            nn.Softmax(dim=0)
        )
        
    def forward(self, state):
        return self.net(state)
    
    def select_action(self, state):
        action = torch.multinomial(self.forward(state), 1)
        return action

def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    
class REINFORCE():
    def __init__(self, config, env, user_features):
        self.config = config
        self.env = env
        make_seed(config['seed'])
        self.env.seed(config['seed'])
        self.user_features = user_features
        self.state_dim = self.user_features.feature_size
        self.model = Model(self.state_dim,
                           self.env.action_space.n)
        self.gamma = config['gamma']
        
        # the optimizer used by PyTorch (Stochastic Gradient, Adagrad, Adam, etc.)
        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=config['learning_rate'])
    
        self.current_ep = 0
        self.rewards = []
    
    def observation_to_log(self,observation, reward):
        data = {'t': [],'u': [], 'z': [],'v': [], 'a': [],
                                   'c': [],'r': [],'ps': [], 'ps-a': []}
        
        def _store_organic(observation):
            assert (observation is not None)
            assert (observation.sessions() is not None)
            for session in observation.sessions():
                data['t'].append(session['t'])
                data['u'].append(session['u'])
                data['z'].append('organic' if session['z']=='pageview' else 'sale') 
                data['v'].append(session['v'])
                data['a'].append(None)
                data['c'].append(None)
                data['r'].append(None) ##H
                data['ps'].append(None)
                data['ps-a'].append(None)

        def _store_clicks(observation,reward):
            assert (observation is not None)
            assert (observation.click is not None)
            # only keep the last bandit event:
            if len(observation.click)>0:
                session = observation.click[len(observation.click)-1]
                data['t'].append(session['t'])
                data['u'].append(session['u'])
                data['z'].append('bandit') 
                data['v'].append(None)
                data['a'].append(session['a'])
                data['c'].append(session['c'])
                data['r'].append(reward) 
                data['ps'].append(None)
                data['ps-a'].append(None)

        _store_organic(observation)
        _store_clicks(observation,reward)
        
        # return as dataframe
        df = pd.DataFrame(data)
        df = df.sort_values('t')
        df.index = range(len(df))
        self.logged_observation = df
        return df
    
    def _compute_returns(self, rewards):
        num_rew = len(rewards)
        exponents = np.arange(num_rew)
        gammas = np.power(self.gamma, exponents)
        
        return rewards.dot(gammas)
    
        
    def optimize_model(self, n_trajectories):

        reward_trajectories = np.empty(n_trajectories)
        loss = 0.
        
        for i in range(n_trajectories):
            print('trajectory nb'+str(i))
            print(datetime.now())
            traj_rewards = []  # rewards of the trajectory
            traj_proba = 0.  # sum of log-probabilities of trajectory
            
            # Build trajectory
            done = False
            self.env.reset()
            # reset user features
            self.reset()
            obs_raw, _, done, _ = self.env.step(None)
            log = self.observation_to_log(obs_raw)
            self.user_features.observe(log)
            state = self.user_features.features()
            state = torch.from_numpy(state).float()
            while not done:
                action = self.model.select_action(state)  # can be cast to int for action idx
                # Get proba
                prob = self.model(state)[int(action)]
                traj_proba += torch.log(prob)
                
                obs_raw, reward, done, info = self.env.step(int(action))
                log = self.observation_to_log(obs_raw)
                self.user_features.observe(log)
                state = self.user_features.features()
                state = torch.from_numpy(state).float()
            
                # Store the new reward
                traj_rewards.append(reward)
                
                
            traj_rewards = np.array(traj_rewards)  # NumPy array
            
            # Get total reward
            total_reward = self._compute_returns(traj_rewards)  # NumPy array
            reward_trajectories[i] = total_reward
            
            loss = loss + total_reward * traj_proba / n_trajectories  # accumulate the negative criterion
            # reset user features construction and delete logs in memory
            self.reset()
            
        self.env.close()  # important
        
        loss = -loss
        
        #  gradient descent step for the variable loss
        print("Loss:", loss.data.numpy())
        
        # Discard previous gradients
        self.optimizer.zero_grad()
        # Compute the gradient 
        loss.backward()
        # Do the gradient descent step
        self.optimizer.step()
        return reward_trajectories
    
    def train(self, n_trajectories, n_update):
        """Training method

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expected gradient
        n_update : int
            The number of gradient updates
            
        """
        
        final_update = self.current_ep + n_update
        rewards = self.rewards  # restart the reward record
        for episode in range(self.current_ep, final_update):
            rewards.append(self.optimize_model(n_trajectories))
            print(f'Episode {episode + 1}/{final_update}: rewards ' 
                  +f'{round(rewards[-1].mean(), 2)} +/- {round(rewards[-1].std(), 2)}')
            self.current_ep += 1
        
        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards[i]) for i in range(len(rewards))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');
        
    def evaluate(self):
        """Evaluate the agent on a single trajectory            
        """
        self.reset()
        self.env.reset()
        ## Wrap in torch.no_grad 
        with torch.no_grad():
            obs_raw, _, done, _ = self.env.step(None)
            log = self.observation_to_log(obs_raw)
            self.user_features.observe(log)
            state = self.user_features.features()
            state = torch.from_numpy(state).float()
            reward_episode = 0
            while not done:
                action = self.model.select_action(state)
                obs_raw, reward, done, info = self.env.step(int(action))
                log = self.observation_to_log(obs_raw)
                self.user_features.observe(log)
                state = self.user_features.features()
                state = torch.from_numpy(state).float()
                reward_episode = self.gamma * reward_episode + reward
        print(f'Reward: {reward_episode}')
        
    def reset(self):
        self.user_features.reset() 