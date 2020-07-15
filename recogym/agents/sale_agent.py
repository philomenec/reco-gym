import numpy as np
# from numba import njit
from numpy.random.mtrand import RandomState
from sklearn.linear_model import LogisticRegression
# import pandas as pd
# from copy import deepcopy
# from scipy.stats.distributions import beta
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# from recogym.util import FullBatchLBFGS
import pandas as pd

# from ..envs.configuration import Configuration
# from ..envs.reco_env_v1_sale import env_1_args, ff, sig
from .abstract import Agent
# from recogym.envs.reco_env_v1_sale import RecoEnv1Sale

from recogym.agents import FeatureProvider


################################################################
################################ Reward shaping ################
################################################################


## Attribute click boolean as reward

class ClickRewardProvider(FeatureProvider):
    """Reward shaping class"""

    def __init__(self, clicks_only=False):
        self.data_rewards = None
        self.clicks_only = clicks_only
        
    def observe(self, data):
        # Consider unclicked recos as negatives
        data_clicked = data.loc[data["z"]=='bandit']
            
        # Set target as MDP reward
        data_clicked["y"] = data_clicked["c"]

        self.data_rewards = data_clicked

    def features(self):
        return self.data_rewards
    
    def reset(self):
        self.data_rewards = None
        
    @property
    def name(self):
        name = "ClickRewardProvider"
        if self.clicks_only == False:
            name += "_all"
        return name



## V0 attribute the reward of the MDP
class MDPRewardProvider(FeatureProvider):
    """Reward shaping class"""

    def __init__(self, clicks_only=True):
        self.data_rewards = None
        self.clicks_only = clicks_only
        
    def observe(self, data):
        # Only keep clicked rows
        if self.clicks_only == True :
            data_clicked = data.loc[data["c"]==1]
        else :
            # Consider unclicked recos as negatives
            data_clicked = data.loc[data["z"]=='bandit']
            
        # Set target as MDP reward
        data_clicked["y"] = data_clicked["r"]

        self.data_rewards = data_clicked

    def features(self):
        return self.data_rewards
    
    def reset(self):
        self.data_rewards = None
   
    @property
    def name(self):
        name = "MDPRewardProvider"
        if self.clicks_only == False:
            name += "_all"
        return name
        
## V1 : Only attribute reward if the recommended product is sold before the next clicked recommendation
class ShortTermRewardProvider(FeatureProvider):
    """Reward shaping class"""

    def __init__(self, clicks_only=True):
        self.data_rewards = None
        self.clicks_only = clicks_only
        
    def observe(self, data):
    # List indices that correspond to clicks
        clicked_indices = list(data.loc[data["c"]==1].index)

        # Only keep clicked rows
        if self.clicks_only == True :
            data_clicked = data.loc[data["c"]==1]
        else :
            # Consider unclicked recos as negatives
            data_clicked = data.loc[data["z"]=='bandit']
        data_clicked["y"] = 0

        last_user_clicked = list(data.loc[data["u"]==data.iloc[clicked_indices[len(clicked_indices)-1]].loc["u"]].index)
        last_user_clicked = last_user_clicked[len(last_user_clicked)-1] +1

        # add one value to clicked indices : the last row for the last user who clicked
        clicked_indices += [last_user_clicked]

        for i in range(len(clicked_indices)-1):
            # Only keep the indices before the next time the user clicks
            data_slice = data.iloc[clicked_indices[i]:clicked_indices[i+1]]
            # Only keep the observations for the same user
            data_slice.loc[data_slice["u"]==data_slice["u"].iloc[0]]

            # Count the number of times a sale for the recommended product is observed in the time frame of interest
            data_clicked["y"].iloc[i] = len(data_slice.loc[(data_slice["z"]=="sale") & (data_slice["v"]==data_slice["a"].iloc[0])])
        self.data_rewards = data_clicked

    def features(self):
        return self.data_rewards
    
    def reset(self):
        self.data_rewards = None

    @property
    def name(self):
        name = "ShortTermRewardProvider"
        if self.clicks_only == False:
            name += "_all"
        return name

## V2 : Attribute a reward if the product was sold later at one point during the user session

class CumulativeRewardProvider(FeatureProvider):
    """Reward shaping class"""
    
    def __init__(self, clicks_only=True):
        self.data_rewards = None
        self.clicks_only = clicks_only

    def observe(self, data):
        # from tqdm import tqdm
        
        # Only keep clicked rows
        if self.clicks_only == True :
            data_clicked2 = data.loc[data["c"]==1]
        else :
            # Consider unclicked recos as negatives
            data_clicked2 = data.loc[data["z"]=='bandit']
        data_clicked2["y"] = 0

        # Loop over users
        # for u in tqdm(data["u"].unique()):
        for u in data["u"].unique():
            # Create dataframe with current user
            data_u = data.loc[data["u"]==u]
            # Keep track of indices of sales, and corresponding products
            sales = {"index":data_u.loc[data["z"]=="sale"].index,
                     "product":list(data_u.loc[data["z"]=="sale"]["v"])}
            # Loop over sales
            for i in range(len(sales["index"])):
                # Create dataframe that includes everything up until the sale
                data_slice = data_u.iloc[:sales["index"][i]]
                # Only keep clicked recos that correspond to the sold product
                data_recos = data_slice.loc[(data_slice["c"]==1) & (data_slice["a"]==sales["product"][i])]
                if len(data_recos) > 0 :
                    index = data_recos.index[len(data_recos.index)-1]
                    data_clicked2["y"].loc[data_clicked2.index == index] = data_clicked2["y"].loc[data_clicked2.index == index] + 1
        self.data_rewards = data_clicked2

    def features(self):
        return self.data_rewards
        
    def reset(self):
        self.data_rewards = None

    @property
    def name(self):
        name = "CumulativeRewardProvider"
        if self.clicks_only == False:
            name += "_all"
        return name





################################################################
################################ Build features ################
################################################################


## Only integrate views

class CountViewsFeatureProvider(FeatureProvider):
    """Feature provider as an abstract class that defines interface of setting/getting features
    The class counts both clicks and views """

    def __init__(self, config):
        super(CountViewsFeatureProvider, self).__init__(config)
        self.num_products = config.num_products
        self.user_features = np.zeros(self.num_products)
        
    def observe(self, data):
        if type(data) == pd.core.series.Series:
            if data["z"]=="organic":
                views = [(data["v"]==p)*1 for p in range(self.num_products)]
            else :
                views = np.zeros(self.num_products)
        else :
            data_organic = data.loc[data["z"]=="organic"].loc[data["v"]>=0]
            views = [np.sum(data_organic["v"]==p) for p in range(self.num_products)]
        self.user_features = np.array(views)
        
    def features(self):
        """Provide feature values adjusted to a particular feature set"""
        return self.user_features

    def reset(self):
        self.user_features = np.zeros(self.num_products)

    @property
    def name(self):
        return "CountViewsFeatureProvider"

## Integrate both clicks and views as features

class CountViewsClicksFeatureProvider(FeatureProvider):
    """Feature provider as an abstract class that defines interface of setting/getting features
    The class counts both clicks and views """

    def __init__(self, config):
        super(CountViewsClicksFeatureProvider, self).__init__(config)
        self.num_products = config.num_products
        self.view_feature = np.zeros(self.num_products)
        self.click_feature = np.zeros(self.num_products)
        self.user_features = np.zeros(2*self.num_products)
        
    def observe(self, data):
        if type(data) == pd.core.series.Series:
            row = data
            if row["z"]=='organic':
                self.view_feature[row["v"]] += 1 
            if (row["z"]=='bandit') & (row["c"]==1):
                self.click_feature[row["a"]] += 1 
        else :
            for _, row in data.iterrows():
                if row["z"]=='organic':
                    self.view_feature[int(row["v"])] += 1 
                if (row["z"]=='bandit') & (row["c"]==1):
                    self.click_feature[int(row["a"])] += 1 
                
        self.user_features = np.concatenate([self.view_feature,self.click_feature])
        
    def features(self):
        """Provide feature values adjusted to a particular feature set"""
        return self.user_features

    def reset(self):
        self.view_feature = np.zeros(self.num_products)
        self.click_feature = np.zeros(self.num_products)
        self.user_features = np.zeros(2*self.num_products)

    @property
    def name(self):
        return "CountViewsClicksFeatureProvider"




################################################################
################################ Build train data ##############
################################################################

def info_reward_provider(data,featured_data, name, share_sales=True):
    dict_info = {'Name':name,
                'Total length data':len(data),
                 'Total length trainset':len(featured_data)}
    if share_sales==True :
        dict_info['Share of sales in trainset'] = np.sum(featured_data["y"])/np.sum(data["z"]=="sale")
    return dict_info


def build_train_data(logs, feature_provider, reward_provider):
    user_states = []

    # Define clicked logs
    reward_provider.observe(logs)
    clicked_log = reward_provider.features()
    
    if reward_provider.name != 'Click_rewards':
        info = info_reward_provider(logs,clicked_log, reward_provider.name, share_sales=True)
    else :
        info = info_reward_provider(logs,clicked_log, reward_provider.name, share_sales=False)
        
    # Restrict logs to users that clicked at least once
    logs = logs[logs["u"].isin(list(clicked_log["u"].unique()))]
    
    current_user = None #for checkup
    for index, row in logs.iterrows():
        if current_user != row['u']:
            # User has changed: reset user state.
            current_user = row['u'] #for checkup
            feature_provider.reset()
        
        feature_provider.observe(row)
        
        if index in clicked_log.index :
            user_states.append(feature_provider.features().copy())
            assert clicked_log["u"][index] == current_user

    return (np.array(user_states), 
            np.array(clicked_log["a"]).astype(int), 
            np.array(clicked_log["y"].astype(int)), 
            np.array(clicked_log["ps"]),
            info)




################################################################
################################ Agents ########################
################################################################






class SaleLikelihoodAgent(Agent):
    def __init__(self, feature_provider, reward_provider, epsilon_greedy = False, epsilon = 0.3, seed=43):
        self.feature_provider = feature_provider
        self.reward_provider = reward_provider
        self.random_state = RandomState(seed)
        self.model = None
        self.epsilon_greedy = epsilon_greedy
        self.epsilon = epsilon
        self.ctr = None
        
    @property
    def num_products(self):
        return self.feature_provider.config.num_products
    
    def _create_features(self, user_state, action):
        """Create the features that are used to estimate the expected reward from the user state"""
        features = np.zeros(len(user_state) * self.num_products)
        # perform kronecker product directly on the flattened version of the features matrix
        features[action * len(user_state): (action + 1) * len(user_state)] = user_state
        return features
    
    def train(self, logs):
        user_states, actions, rewards, proba_actions, self.info = build_train_data(logs, 
                                                                        self.feature_provider, 
                                                                        self.reward_provider)
        
        rewards = (rewards > 0)*1
        # estimate sales rate (boolean)
        count_actions = np.unique(actions,return_counts = True)[1]
        # assert len(count_actions) == self.num_products
        count_sales_bool = np.array([len(np.where((actions==_) & (rewards>0))[0]) for _ in range(self.num_products)])
        self.salesrate = count_sales_bool / count_actions
        
        features = np.vstack([
            self._create_features(user_state, action) 
            for user_state, action in zip(user_states, actions)
        ])
        self.model = LogisticRegression(solver='lbfgs', max_iter=5000)
        self.model.fit(features, rewards)
        
    
    def _score_products(self, user_state):
        all_action_features = np.array([
            # How do you create the features to feed the logistic model ?
            self._create_features(user_state, action) for action in range(self.num_products)
        ])
        return self.model.predict_proba(all_action_features)[:, 1]
    
    def observation_to_log(self,observation):
        data = {
                    't': [],
                    'u': [],
                    'z': [],
                    'v': [],
                    'a': [],
                    'c': [],
                    'r': [],
                    'ps': [],
                    'ps-a': [],
                }
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

        def _store_clicks(observation):
            assert (observation is not None)
            assert (observation.click is not None)
            for session in observation.click:
                data['t'].append(session['t'])
                data['u'].append(session['u'])
                data['z'].append('bandit') 
                data['v'].append(None)
                data['a'].append(session['a'])
                data['c'].append(session['c'])
                data['r'].append(None) 
                data['ps'].append(None)
                data['ps-a'].append(None)

        _store_organic(observation)
        _store_clicks(observation)
        df = pd.DataFrame(data)
        df.sort_values('t')
        return df
        
    
    def act(self, observation, reward, done):
        """Act method returns an action based on current observation and past history"""
        logged_observation = self.observation_to_log(observation)
        self.feature_provider.observe(logged_observation)      
        user_state = self.feature_provider.features()
        if (self.epsilon_greedy == True) & (np.random.rand() < self.epsilon) : 
            print("Explore")
            action = np.random.randint(self.num_products())
        else :
            action = np.argmax(self._score_products(user_state))
        
        ps = 1.0
        all_ps = np.zeros(self.num_products)
        all_ps[action] = 1.0        
        
        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': ps,
                'ps-a': all_ps,
            }
        }

    def reset(self):
        self.feature_provider.reset()  



