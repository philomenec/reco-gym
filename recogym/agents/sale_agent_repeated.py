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
from ..envs.reco_env_v1_sale import sig
# from ..envs.reco_env_v1_sale import env_1_args, ff, sig
from recogym.agents.abstract import Agent
# from recogym.envs.reco_env_v1_sale import RecoEnv1Sale

from recogym.agents import FeatureProvider
from scipy.special import logit
import pickle as pkl
from IPython.display import display
from sklearn.linear_model import LinearRegression





################################################################
################################ Reward shaping ################
################################################################


## Pseudo-rewards

class PseudoRewards(FeatureProvider):
    """Reward shaping class"""

    def __init__(self, clicks_only=True):
        self.data_rewards = None
        self.clicks_only = clicks_only
        self.data = None
        
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
            
    def pseudo_observe(self, data):
            
        data['immediate_sale'] = ((data['c']==1)&(data.shift(-2)["z"]=='sale')&(data.shift(-2)["u"]==data["u"]))*1
        self.data = data
        
        # Only keep clicked rows
        if self.clicks_only == True :
            data_clicked = data.loc[data["c"]==1]
        else :
            # Consider unclicked recos as negatives
            data_clicked = data.loc[data["z"]=='bandit']
        data_clicked['pseudo_y'] = data_clicked['r'] - data_clicked['immediate_sale']
        self.data_rewards = data_clicked

    def features(self):
        return self.data_rewards
    
    def reset(self):
        self.data_rewards = None
        self.data = None
   
    @property
    def name(self):
        name = "PseudoRewardProvider"
        if self.clicks_only == False:
            name += "_all"
        return name
       
  
################################################################
################################ Build user state ##############
################################################################

    
class ShareViewsCountClicksFeatureProvider(FeatureProvider):
    """Feature provider as an abstract class that defines interface of setting/getting features
    The class counts both clicks and views """

    def __init__(self, config):
        super(FeatureProvider, self).__init__()
        self.num_products = config.num_products
        self.view_feature = np.zeros(self.num_products)
        self.click_feature = np.zeros(self.num_products)
        self.view_count = np.zeros(self.num_products)
        self.click_count = np.zeros(self.num_products)
        self.user_features = np.zeros(2*self.num_products)
        
        
    def observe(self, data, memory=True):
        if type(data) == pd.core.series.Series:
            row = data
            if memory:
                if row["z"]=='organic':
                    self.view_count[int(row["v"])] += 1 
                if (row["z"]=='bandit') & (row["c"]==1):
                    self.click_count[int(row["a"])] += 1 
            else:
                if row["z"]=='organic':
                    self.view_count[int(row["v"])] = 1 
                if (row["z"]=='bandit') & (row["c"]==1):
                    self.click_count[int(row["a"])] = 1 
        else :
            for _, row in data.iterrows():
                if memory:
                    if row["z"]=='organic':
                        self.view_count[int(row["v"])] += 1 
                    if (row["z"]=='bandit') & (row["c"]==1):
                        self.click_count[int(row["a"])] += 1 
                else:
                    if row["z"]=='organic':
                        self.view_count[int(row["v"])] = 1 
                    if (row["z"]=='bandit') & (row["c"]==1):
                        self.click_count[int(row["a"])] = 1 
        nb_views = np.sum(self.view_count)
        if nb_views > 0:
            self.view_feature = self.view_count/nb_views
        self.click_feature = self.click_count
            
        self.user_features = np.concatenate([self.view_feature,self.click_feature])
        
    def features(self):
        """Provide feature values adjusted to a particular feature set"""
        return self.user_features

    def reset(self):
        self.view_feature = np.zeros(self.num_products)
        self.click_feature = np.zeros(self.num_products)
        self.user_features = np.zeros(2*self.num_products)
        self.view_count = np.zeros(self.num_products)
        self.click_count = np.zeros(self.num_products)
       
    @property
    def weight(self):
        # Note : this could be defined differently
        return np.sum(self.view_count)+np.sum(self.click_count)
 
       
    @property
    def name(self):
        return "ShareViewsCountClicksFeatureProvider"

    @property
    def feature_size(self):
        return 2*self.num_products
    


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


def build_train_data(logs, feature_provider, reward_provider, weights=False):
    user_states = []
    
    # Define clicked logs
    reward_provider.observe(logs)
    train_log = reward_provider.features()
    
    if sum([subtext in reward_provider.name.lower() for subtext in ('click','pure','postdisplay')])==0:
        info = info_reward_provider(logs,train_log, reward_provider.name, share_sales=True)
    else :
        info = info_reward_provider(logs,train_log, reward_provider.name, share_sales=False)
    
    # Restrict logs to users that are present in the trainset
    logs = logs[logs["u"].isin(list(train_log["u"].unique()))]
    
    if weights == True:
        # initialise the weights
        info["weights"] = []
        weights_list = []
    else:
        info["weights"] = None
    
    # initialise the user
    current_user = None #for checkup
    for index, row in logs.iterrows():
        if current_user != row['u']:
            # User has changed: reset user state.
            current_user = row['u'] #for checkup
            feature_provider.reset()
            if weights == True:
                weights_list.append([])
                # info["weights"].append([])
            
        feature_provider.observe(row)
        feature_sum = feature_provider.features()
        
        if index in train_log.index :
        # if index in np.array(train_log.index)-1 :
            user_states.append(feature_sum)
            assert train_log["u"][index] == current_user  
            
            if weights == True:
                weights_list[len(weights_list)-1].append(feature_provider.weight)
                # info["weights"][len(weights_list)-1].append(np.sqrt(np.sum(weights_list[len(weights_list)-1])))
                # info["weights"].append(np.sqrt(np.sum(weights_list[len(weights_list)-1])))
                # avg_product_variance = np.mean([p*(1-p) for p in feature_sum])
                # assert ((avg_product_variance >= 0) and (avg_product_variance <= 1))
                # info["weights"].append(np.sum(weights_list[len(weights_list)-1])/(avg_product_variance+1e-2))
                info["weights"].append(np.sum(weights_list[len(weights_list)-1]))
        
        # if weights == True:
        #     # Flatten the list and turn it into a numpy array
        #     info["weights"] = np.array([i for user_sublist in info["weights"] for i in user_sublist])
    
    return (np.array(user_states), 
            np.array(train_log["a"]).astype(int), 
            np.array(train_log["y"].astype(int)), 
            np.array(train_log["ps"]),
            info)


def build_lookahead_train_data(logs, feature_provider, reward_provider,weights=False):
    user_states = []

    # Define clicked logs
    reward_provider.observe(logs)
    train_log = reward_provider.features()
    
    if sum([subtext in reward_provider.name.lower() for subtext in ('click','pure','postdisplay')])==0:
        info = info_reward_provider(logs,train_log, reward_provider.name, share_sales=True)
    else :
        info = info_reward_provider(logs,train_log, reward_provider.name, share_sales=False)
        
    # Restrict logs to users that are present in the trainset
    logs = logs[logs["u"].isin(list(train_log["u"].unique()))]
    
    # look at all user information at once (look ahead)
    for u in logs["u"].unique():
        logs_u = logs.loc[logs["u"]==u]
        feature_provider.reset()
        feature_provider.observe(logs_u)
        user_state_u = feature_provider.features()
        user_states = user_states + list([user_state_u]*len(train_log.loc[train_log["u"]==u]))
    
    assert len(user_states) == len(train_log)
    
    return (np.array(user_states), 
            np.array(train_log["a"]).astype(int), 
            np.array(train_log["y"].astype(int)), 
            np.array(train_log["ps"]),
            info)




################################################################
################################ Agents ########################
################################################################






class SaleLikelihoodAgent(Agent):
    def __init__(self, feature_provider, reward_provider, kronecker_features = False, 
                 look_ahead = False, sample_weight = False, softmax = False,
                 # multiply_features = False,
                 epsilon_greedy = False, epsilon = 0.3, seed=43):
        self.feature_provider = feature_provider
        self.reward_provider = reward_provider
        self.random_state = RandomState(seed)
        self.model = None
        # self.multiply_features = multiply_features
        # self.softmax = softmax
        self.epsilon_greedy = epsilon_greedy
        self.epsilon = epsilon
        self.logged_observation = {'t': [],'u': [], 'z': [],'v': [], 'a': [],
                                   'c': [],'r': [],'ps': [], 'ps-a': []}
        self.kronecker_features = kronecker_features
        self.look_ahead = look_ahead
        self.sample_weight = sample_weight
        self.report_issue = {}
        
    @property
    def num_products(self):
        return self.feature_provider.config.num_products
    
    def _create_features(self, user_state, action):
        """Create the features that are used to estimate the expected reward from the user state"""
        features = np.zeros(len(user_state) + self.num_products)
        # just put a dummy variable
        features[:len(user_state)] = user_state
        features[int(len(user_state) + action)] = 1
        return features
    
    def _create_features_kronecker(self, user_state, action):
        """Create the features that are used to estimate the expected reward from the user state"""
        features = np.zeros(len(user_state) * self.num_products)
        # perform kronecker product directly on the flattened version of the features matrix
        features[action*len(user_state):(action+1)*len(user_state)] = user_state
        return features
    
    def train(self, logs):
        if self.look_ahead == False:
            user_states, actions, rewards, proba_actions, self.info = build_train_data(logs, 
                                                                            self.feature_provider, 
                                                                            self.reward_provider,
                                                                            weights = self.sample_weight)
            weights = np.array(self.info["weights"]) if self.sample_weight == True else None

        else:
            user_states, actions, rewards, proba_actions, self.info = build_lookahead_train_data(logs, 
                                                                            self.feature_provider, 
                                                                            self.reward_provider)
            weights = None
            
        rewards = (rewards > 0)*1
        # self.cr = sum(rewards)/len(rewards)
        
        if self.kronecker_features == False:
            actions_onehot = np.zeros((len(actions), self.num_products))
            actions_onehot[np.arange(len(actions)),actions] = 1
            features = np.hstack([
                user_states,
                actions_onehot
            ])
        else :
            features = np.array([self._create_features_kronecker(user_states[i],
                                                                    actions[i]) 
                                         for i in range(len(actions))])
        
        self.model = LogisticRegression(solver='lbfgs', max_iter=5000)
        self.model.fit(features, rewards, sample_weight=weights)
        
    
    def _score_products(self, user_state):
        if self.kronecker_features == False:
            all_action_features = np.array([self._create_features(user_state,action)
                                            for action in range(self.num_products)])
        else :
            all_action_features = np.array([self._create_features_kronecker(user_state,action)
                                            for action in range(self.num_products)])
        try:
            score = self.model.predict_proba(all_action_features)[:, 1]
        except:
            score = self.model.predict_proba(all_action_features)
        # if self.multiply_features:
        #     try : 
        #         score = score * np.array(user_state)
        #     except Exception as e: 
        #         print(e)
        return score
    
    def observation_to_log(self,observation):
        data = self.logged_observation
        
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
        self.logged_observation = data
        
        # return as dataframe
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
        # elif self.softmax :
        #     score = self._score_products(user_state)
        #     action = self.random_state.choice(
        #             self.config.num_products,
        #             p=score/np.sum(score)
        #         )
        else :
            action = np.argmax(self._score_products(user_state))
        
        ps = 1.0
        all_ps = np.zeros(self.num_products)
        all_ps[action] = 1.0        
        
        if done :
            self.reset()
        
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
        self.logged_observation = {'t': [],'u': [], 'z': [],'v': [], 'a': [],
                                   'c': [],'r': [],'ps': [], 'ps-a': []}


class SaleProductLikelihoodAgent(Agent):
    def __init__(self, feature_provider_list, reward_provider_list, discounts, discounts_with_action=True, 
                 kronecker_features = False, linear_reg = False, look_ahead = False, 
                 sample_weight = False, 
                 # softmax = False,
                 # multiply_features = False, 
                 epsilon_greedy = False, epsilon = 0.3, seed=43):
        self.feature_provider_list = feature_provider_list
        self.reward_provider_list = reward_provider_list
        self.discounts = np.array(discounts) # list that indicates which model is used to which other model
        assert len(self.feature_provider_list) == len(self.reward_provider_list)
        self.random_state = RandomState(seed)
        self.models = [] # each model will correspond to a fitted logistic regression
        self.discounts_with_action = discounts_with_action
        self.kronecker_features = kronecker_features
        self.look_ahead = look_ahead
        self.sample_weight = sample_weight
        # self.multiply_features = multiply_features
        # self.softmax = softmax
        self.epsilon_greedy = epsilon_greedy
        self.epsilon = epsilon
        self.num_models = len(self.reward_provider_list)
        # Keep information about the model
        self.info = {'Name':'','Other':[],'weights':[]}
        # Indices of models used to discount other models
        self.index_discount = np.where(self.discounts != 0)[0]
        # Indices of discounted models
        self.index_discounted = self.index_discount + self.discounts[self.index_discount]
        self.logged_observation = {'t': [],'u': [], 'z': [],'v': [], 'a': [],
                                   'c': [],'r': [],'ps': [], 'ps-a': []}
        self.linear_reg = linear_reg #whether to use a linear regression if labels are non-binary
        self.report_issue = {'discounts':[]}
        
    @property
    def num_products(self):
        return self.feature_provider_list[0].config.num_products
    
    def _create_features(self, user_state, action):
        """Create the features that are used to estimate the expected reward from the user state"""
        features = np.zeros(len(user_state) + self.num_products)
        # dummy variables
        features[:len(user_state)] = user_state
        features[int(len(user_state) + action)] = 1
        return features
    
    def _create_features_kronecker(self, user_state, action):
        """Create the features that are used to estimate the expected reward from the user state"""
        features = np.zeros(len(user_state) * self.num_products)
        # perform kronecker product directly on the flattened version of the features matrix
        features[action*len(user_state):(action+1)*len(user_state)] = user_state
        return features
    
    def train(self, logs):
        # Train all models
        for i in range(self.num_models):
            # Build training data
            feature_provider = self.feature_provider_list[i]
            reward_provider = self.reward_provider_list[i]
            
            if (self.look_ahead == False):
                user_states, actions, rewards, proba_actions, info = build_train_data(logs, 
                                                                                feature_provider, 
                                                                                reward_provider,
                                                                                weights = self.sample_weight)
                weights = np.array(info["weights"]) if (self.sample_weight == True) & (self.discounts[i] == 0) else None
    
            else:
                user_states, actions, rewards, proba_actions, info = build_lookahead_train_data(logs, 
                                                                                feature_provider, 
                                                                                reward_provider)
                weights = None
            
            self.info["weights"].append(weights)
            self.info["Other"].append(info)
            # Keep info
            self.info["Name"]+= "Discounted" if i in self.index_discounted else ""
            self.info["Name"]+= "Discount" if i in self.index_discount else ""
            self.info["Name"]+= info["Name"]+"_"
            
            # Convert rewards to binary rewards (just in case of multiple sales)
            if np.max(rewards) > 1 : 
                rewards = (rewards > 0)*1
            
            # Build features 
            if (self.discounts[i] == 0) or self.discounts_with_action :
                # Include action         
                if self.kronecker_features == False:   
                    actions_onehot = np.zeros((len(actions), self.num_products))
                    actions_onehot[np.arange(len(actions)),actions] = 1
                    
                    features = np.hstack([
                        user_states,
                        actions_onehot
                    ])
                else :
                    features = np.array([self._create_features_kronecker(user_states[i],
                                                                    actions[i]) 
                                         for i in range(len(actions))])
                
            else :
                # Models used as discounts don't use actions
                features = user_states
                    
                
            # Fit logistic regression and save model
            if len(features) > 0 :
                if self.linear_reg == False :
                    model = LogisticRegression(solver='lbfgs', max_iter=5000)
                    model.fit(features, rewards, sample_weight=weights)
                else: 
                    model = FracLogisticRegression()
                    model.fit(features, rewards, sample_weight=weights)
            else :
                model = None
                
            self.models.append(model)
    
    def _score_products(self, user_state, model):
        
        if self.kronecker_features == False:
            all_action_features = np.array([self._create_features(user_state,action)
                                            for action in range(self.num_products)])
        else :
            all_action_features = np.array([self._create_features_kronecker(user_state,action)
                                            for action in range(self.num_products)])
        try:
            score = model.predict_proba(all_action_features)[:, 1]
        except:
            score = model.predict_proba(all_action_features)
        # if self.multiply_features:
        #     try:
        #         score = score * np.array(user_state)
        #     except Exception as e: 
        #         print(e)
        return score
    
    
    def observation_to_log(self,observation):
        data = self.logged_observation
        
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
        self.logged_observation = data
        
        df = pd.DataFrame(data)
        df.sort_values('t')
        return df
        
    
    def act(self, observation, reward, done):
        """Act method returns an action based on current observation and past history"""
        if (self.epsilon_greedy == True) & (np.random.rand() < self.epsilon) : 
            print("Explore")
            action = np.random.randint(self.num_products())
            
        else :
            logged_observation = self.observation_to_log(observation)
            # Initialize the scores as 1
            score = np.ones(self.num_products)
            
            for i in range(len(self.index_discounted)):
                ## main (discounted) model 
                feature_provider = self.feature_provider_list[self.index_discounted[i]]
                feature_provider.observe(logged_observation) 
                user_state = feature_provider.features()
                before_discount = self._score_products(user_state, 
                                                       self.models[self.index_discounted[i]])
                
                ## model used as discount
                feature_provider = self.feature_provider_list[self.index_discount[i]]
                feature_provider.observe(logged_observation)      
                user_state = feature_provider.features()
                user_state = user_state.reshape(1, -1)
                if self.models[self.index_discount[i]] is not None :
                    if self.discounts_with_action == False :
                        discount_val = self.models[self.index_discount[i]].predict_proba(user_state)
                        try:
                            discount_val = discount_val[0][1]
                        except:
                            discount_val = discount_val[0]
                    else :
                        discount_val = self._score_products(user_state,
                                                            self.models[self.index_discount[i]])
                else :
                    print("No discount")
                    discount_val = np.zeros(before_discount.shape)
                    
                ## Combine the two models
                # # Clip discounted score to avoid negatives
                # after_discount = np.clip(before_discount - discount_val,a_min=0,a_max=1)
                # Clip discounted score to avoid negatives
                # print("before discount ",before_discount)
                # print("discount_val ",discount_val)
                after_discount = before_discount - discount_val
                self.report_issue['discounts'].append(after_discount)
                if len(np.where(after_discount<0)[0]) == self.num_products:
                    # print(after_discount)
                    # print(np.where(after_discount<0)[0])
                    after_discount = sig(after_discount)
                score = score*after_discount
            
            for i in set(range(self.num_models))-set.union(set(self.index_discount),set(self.index_discounted)):
                feature_provider = self.feature_provider_list[i]
                feature_provider.observe(logged_observation)      
                user_state = feature_provider.features()
                score = score*self._score_products(user_state, self.models[i])
            # if self.softmax == False:
            #     action = np.argmax(score)
            # else:
            #     action = self.random_state.choice(
            #             self.config.num_products,
            #             p=score/np.sum(score)
            #         )
            action = np.argmax(score)
        ps = 1.0
        all_ps = np.zeros(self.num_products)
        all_ps[action] = 1.0        
        
        if done :
            self.reset()
        
        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': ps,
                'ps-a': all_ps,
            }
        }

    def reset(self):
        for i in range(self.num_models):
            self.feature_provider_list[i].reset()  
            
        self.logged_observation = {'t': [],'u': [], 'z': [],'v': [], 'a': [],
                                   'c': [],'r': [],'ps': [], 'ps-a': []}




## Function to train all agents at once
def train_agents(name_logging,logs,feature_name,features, num_users, kronecker_features=False,
                 weights=False, repo = 'data/'):
    info = {}
    save_agents = {}
    data = logs[name_logging]
    feature = features[feature_name]
    
    # reward providers
    Click_rewards = ClickRewardProvider()
    MDP_rewards = MDPRewardProvider()
    MDP_rewards_all = MDPRewardProvider(clicks_only=False)
    MDP_rewards_pureorganic = MDPRewardProvider(clicks_only=False, organic_only=True)
    PostDisplay_rewards = PostDisplayRewardProvider()
    Noclick_rewards_provider = NoclickRewardProdivder()
    
    name_extension = 'pres'
    
    # name_extension = ''
    # if kronecker_features==True:
    #     print("Kronecker features")
    #     name_extension += 'kron'
        
    if weights==True:
        print("Weights")
        name_extension += 'weights'
    
    
    # click agent
    print("Click agent")
    likelihood_logreg_click = SaleLikelihoodAgent(feature, Click_rewards,
                                                  kronecker_features = kronecker_features,
                                                  sample_weight = weights)
    likelihood_logreg_click.train(data)
    info[likelihood_logreg_click.info["Name"]] = likelihood_logreg_click.info
    save_agents["likelihood_logreg_click"+name_extension] = likelihood_logreg_click
    
    # post display agent
    print("Post display agent")
    likelihood_logreg_postd = SaleLikelihoodAgent(feature, PostDisplay_rewards,
                                                  kronecker_features = kronecker_features,
                                                  sample_weight = weights)
    likelihood_logreg_postd.train(data)
    info[likelihood_logreg_postd.info["Name"]] = likelihood_logreg_postd.info
    save_agents["likelihood_logreg_postd"+name_extension] = likelihood_logreg_postd
        
    # No discount, clicked observations
    print("No discount")    
    likelihood_saleclickprod = SaleProductLikelihoodAgent(feature_provider_list=[feature,feature], 
                                                    reward_provider_list=[Click_rewards,MDP_rewards], 
                                                    discounts=[0,0],discounts_with_action=False,
                                                    kronecker_features = kronecker_features,
                                                  sample_weight = weights)
    likelihood_saleclickprod.train(data)
    info[likelihood_saleclickprod.info["Name"]] = likelihood_saleclickprod.info
    save_agents["likelihood_saleclickprod"+name_extension] = likelihood_saleclickprod
    
      
    # non-specific discount, clicked observations
    print("Discount")    
    likelihood_saleclickprod_discount = SaleProductLikelihoodAgent(feature_provider_list=[feature,feature,feature], 
                                                    reward_provider_list=[Click_rewards,MDP_rewards,Noclick_rewards_provider], 
                                                    discounts=[0,0,-1],discounts_with_action=False,
                                                    kronecker_features = kronecker_features,
                                                    sample_weight = weights)
    likelihood_saleclickprod_discount.train(data)
    info[likelihood_saleclickprod_discount.info["Name"]] = likelihood_saleclickprod_discount.info
    save_agents["likelihood_saleclickprod_discount"+name_extension] = likelihood_saleclickprod_discount
    
   
    pkl.dump([info,save_agents],open(str(repo) + str('agents'+str(num_users)+name_logging+feature_name+name_extension+'.pkl'),'wb'))
    return info, save_agents




def train_timeagents(name_logging,logs,feature_name,features, num_users, kronecker_features=False, linear_reg=False,
                     weights=False, repo = 'data/'):
    info = {}
    save_agents = {}
    data = logs[name_logging]
    feature = features[feature_name]
    
    # reward providers
    Click_rewards = ClickRewardProvider()
    MDP_rewards_time = MDPRewardProvider(clicks_only=True, organic_only=False, normalize = True)
    MDP_rewards_all_time = MDPRewardProvider(clicks_only=False, organic_only=False, normalize = True)
    MDP_rewards_pureorganic = MDPRewardProvider(clicks_only=False, organic_only=True)
    Noclick_rewards_provider = NoclickRewardProdivder()
    Noclick_rewards_provider_time = NoclickRewardProdivder(normalize = True)
    
    name_extension = 'prop'
    # name_extension = 'time'
    
    # if kronecker_features==True:
    #     print("Kronecker features")
    #     name_extension += 'kron'
    # if linear_reg ==True:
    #     print("Linear reg")
    #     name_extension += 'lin'
    
    if weights==True:
        print("Weights")
        name_extension += 'weights'
   

    # No discount
    print("No discount")
    likelihood_saleclickprod = SaleProductLikelihoodAgent(feature_provider_list=[feature,feature], 
                                                    reward_provider_list=[Click_rewards,MDP_rewards_time], 
                                                    discounts=[0,0],discounts_with_action=False,
                                                    kronecker_features = kronecker_features,
                                                    linear_reg = linear_reg,
                                                    sample_weight = weights)
    likelihood_saleclickprod.train(data)
    info[likelihood_saleclickprod.info["Name"]] = likelihood_saleclickprod.info
    save_agents["likelihood_saleclickprod"+name_extension] = likelihood_saleclickprod
    
    print("Discount")
    # non-specific discount, clicked observations
    likelihood_saleclickprod_discount = SaleProductLikelihoodAgent(feature_provider_list=[feature,feature,feature], 
                                                    reward_provider_list=[Click_rewards,MDP_rewards_time,Noclick_rewards_provider_time], 
                                                    discounts=[0,0,-1],discounts_with_action=False,
                                                    kronecker_features = kronecker_features,
                                                    linear_reg = linear_reg,
                                                    sample_weight = weights)
    likelihood_saleclickprod_discount.train(data)
    info[likelihood_saleclickprod_discount.info["Name"]] = likelihood_saleclickprod_discount.info
    save_agents["likelihood_saleclickprod_discount"+name_extension] = likelihood_saleclickprod_discount
        
    
    try:
        pkl.dump([info,save_agents],open(str(repo)+str('timeagents'+str(num_users)+name_logging+feature_name+name_extension+'.pkl'),'wb'))
    except:
        print(" //!!\\ Error when saving agents")
    return info, save_agents


