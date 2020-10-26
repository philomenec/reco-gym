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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, 
                             precision_score, recall_score)
from sklearn.metrics import (explained_variance_score,max_error,mean_absolute_error,
                             mean_squared_error,r2_score)


class FracLogisticRegression(LinearRegression):

    def fit(self, x, p, sample_weight):
        p = np.clip(np.array(p),a_min=1e-10,a_max =1-1e-10)
        y = logit(p)
        return super().fit(x, y, sample_weight)

    def predict(self, x):
        y = super().predict(x)
        return y

    def predict_proba(self, x):
        y = super().predict(x)
        p = 1 / (np.exp(-y) + 1)
        # print(p)
        return p





################################################################
################################ Reward shaping ################
################################################################


## (1) Attribute click boolean as reward

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


class NoclickRewardProdivder(FeatureProvider):
    """Reward shaping class"""

    def __init__(self, organic_only=False, normalize=False):
        self.data_rewards = None
        self.organic_only = organic_only
        self.normalize = normalize
        
    def observe(self, data):
        
        if self.organic_only :
            # Only keep data before first clicked reco 
            pure_organic_df = data[:0].copy()
            for u in data["u"].unique():
                beginning_index = data[data["u"]==u].index[0]
                if len(data[data["u"]==u].loc[data["c"]==1])>0:
                    end_index = data[data["u"]==u].loc[data["c"]==1].index[0] #dont include the first click
                else :
                    end_index = data[data["u"]==u].index[len(data[data["u"]==u])-1]+1 #Include the last index
                pure_organic_df = pd.concat([pure_organic_df,data.iloc[beginning_index:end_index]])
            data = pure_organic_df
          
        ## attribute rewards
        # define boolean for bandit data followed by an organic session (of the same user)
        data['bandit_to_organic'] = ((data['z']=="bandit") & (data.shift(-1)["z"]=="organic"))*1
        data['bandit_to_organic'] = data['bandit_to_organic']*((data["u"].shift(-1) == data['u'])*1)
        data['bandit_to_organic'].iloc[len(data)-1] = 0 #avoid NaN at the end
        # initialise the label
        data["y"] = 0.0 if self.normalize else 0
        
        # initialise final dataset
        data_new = data.iloc[:1]
        
        for u in data["u"].unique():
            data_u = data.loc[data["u"]==u]
            indices_u = list(data_u.index)
            # indices of bandit events
            indices_bandit = list(data_u[data_u["bandit_to_organic"]==1].index)
            for i in range(len(indices_bandit)):
                # select data either up until the end of the episode or up until next bandit event
                if sum(data_u[data_u.index > (indices_bandit[i])]["z"]=='bandit') > 0:
                    last_index = list(data_u[data_u.index > (indices_bandit[i])].loc[data_u["z"]=='bandit'].index)[0]
                else :
                    last_index = data_u.iloc[len(data_u)-1].name + 1
                # keep following organic session
                organic_session = data_u[(data_u.index > indices_bandit[i]) & (data_u.index < last_index)]
                # count number of sales
                data_u["y"][indices_bandit[i]] = np.sum(organic_session["z"]=='sale')
                # normalise by number of organic events if needed
                if (self.normalize & (data_u["y"][indices_bandit[i]] > 0)):
                    data_u["y"][indices_bandit[i]] = data_u["y"][indices_bandit[i]] / np.sum(organic_session["z"]=='organic')
            data_new = pd.concat([data_new,data_u])  
        # only keep bandit events followed by organic session
        self.data_rewards = data_new.loc[data_new['bandit_to_organic']==1]

    def features(self):
        return self.data_rewards
    
    def reset(self):
        self.data_rewards = None
   
    @property
    def name(self):
        name = "NoClickRewardProvider"
        if self.organic_only :
            name += "_pureorganic"
        if self.normalize : 
            name += "_time"
        return name



## (2) attribute the reward of the MDP
class MDPRewardProvider(FeatureProvider):
    """Reward shaping class"""

    def __init__(self, clicks_only=True, organic_only=False, normalize=False):
        self.data_rewards = None
        self.clicks_only = clicks_only
        self.organic_only = organic_only
        self.normalize = normalize
        
    def observe(self, data):
        
        if self.organic_only :
            # Only keep data before first clicked reco 
            pure_organic_df = data[:0].copy()
            for u in data["u"].unique():
                beginning_index = data[data["u"]==u].index[0]
                if len(data[data["u"]==u].loc[data["c"]==1])>0:
                    end_index = data[data["u"]==u].loc[data["c"]==1].index[0] #dont include the first click
                else :
                    end_index = data[data["u"]==u].index[len(data[data["u"]==u])-1]+1 #Include the last index
                pure_organic_df = pd.concat([pure_organic_df,data.iloc[beginning_index:end_index]])
                # remove bandit events
                pure_organic_df = pure_organic_df.loc[pure_organic_df["z"] != 'bandit']
            data = pure_organic_df
            # add a reward in case of a sale and artificially define the action as the view
            data["y"] = (data["z"]=='sale')*1
            data["a"] = data["v"]
            # remove organic events followed by a sale to avoid double counting
            data = data.loc[data.shift(-1)["z"]!="sale"]
            
            # if self.normalize : 
            #     # group by user and views
            #     user_views_counts = data.groupby(["u","v"]).count()
            #     # keep track of (user,view) pair and counts
            #     indices = np.array(user_views_counts.index)
            #     counts = np.array(user_views_counts["t"])
            #     # initialise a matrix of per user-action counts (also with views that never appeared)
            #     user_views_counts = np.zeros((len(data["u"].unique()),1+np.max([indices[i][1] for i in range(len(indices))])))
            #     user_views_counts[tuple(zip(*indices))] = counts
            #     # divide the reward by the number of "trials" of user u with product a
            #     denominator = user_views_counts[np.array(data["u"][data["y"]>0]).astype(int),np.array(data["v"][data["y"]>0]).astype(int)]
            #     data["y"][data["y"]>0] = data["y"][data["y"]>0]/denominator
            
            self.data_rewards = data
        
        else :
            
            if self.normalize :
                # list indices of positive rewards
                positive_rewards = data[data['r']>0].index
                for i in positive_rewards:
                    # for each successful bandit event, find the index of the next bandit event
                    next_bandit = data.iloc[(i+1):].loc[data["z"]=="bandit"]
                    # if no more bandit event: take the last index of the dataset
                    if len(next_bandit) == 0:
                        next_bandit = data.iloc[len(data)-1].name+1
                    else:
                        # otherwise: keep the next index
                        next_bandit = next_bandit.index[0]
                    # derive the length of the corresponding organic session
                    length_organic_session = len(data.iloc[(i+1):next_bandit].loc[data["z"]!="sale"])
                    if length_organic_session == 0:
                        print("organic session of length 0")
                    # divide the rewards by the number of trials
                    data["r"].iloc[i] = data["r"].iloc[i]/length_organic_session
            
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
        if self.organic_only :
            name += "_pureorganic"
        if self.normalize : 
            name += "_time"
        return name
       
    
## (3) : Only attribute reward if any product is sold before the next clicked recommendation
class NonPersoShortTermRewardProvider(FeatureProvider):
    """Reward shaping class"""

    def __init__(self, clicks_only=True, organic_only=False):
        self.data_rewards = None
        self.clicks_only = clicks_only
        self.organic_only = organic_only
        
    def observe(self, data):
        
        if self.organic_only :
            # Only keep data before first clicked reco 
            pure_organic_df = data[:0].copy()
            for u in data["u"].unique():
                beginning_index = data[data["u"]==u].index[0]
                if len(data[data["u"]==u].loc[data["c"]==1])>0:
                    end_index = data[data["u"]==u].loc[data["c"]==1].index[0] #dont include the first click
                else :
                    end_index = data[data["u"]==u].index[len(data[data["u"]==u])-1]+1 #Include the last index
                pure_organic_df = pd.concat([pure_organic_df,data.iloc[beginning_index:end_index]])
                # remove bandit events
                pure_organic_df = pure_organic_df.loc[pure_organic_df["z"] != 'bandit']
            data = pure_organic_df
            # add a reward in case of a sale and artificially define the action as the view
            data["y"] = (data["z"]=='sale')*1
            data["a"] = data["v"]
            # remove organic events followed by a sale to avoid double counting
            data = data.loc[data.shift(-1)["z"]!="sale"]
            self.data_rewards = data
        
        else :
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
                data_slice = data_slice.loc[data_slice["u"]==data_slice["u"].iloc[0]]
    
                # Count the number of times a sale for the recommended product is observed in the time frame of interest
                data_clicked["y"].iloc[i] = len(data_slice.loc[(data_slice["z"]=="sale")])
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
        if self.organic_only :
            name += "_pureorganic"
        return name
    
    
    
## (4) : Only attribute reward if the recommended product is sold before the next clicked recommendation
class ShortTermRewardProvider(FeatureProvider):
    """Reward shaping class"""

    def __init__(self, clicks_only=True, organic_only=False):
        self.data_rewards = None
        self.clicks_only = clicks_only
        self.organic_only = organic_only
        
    def observe(self, data):
        
        if self.organic_only :
            # Only keep data before first clicked reco 
            pure_organic_df = data[:0].copy()
            for u in data["u"].unique():
                beginning_index = data[data["u"]==u].index[0]
                if len(data[data["u"]==u].loc[data["c"]==1])>0:
                    end_index = data[data["u"]==u].loc[data["c"]==1].index[0] #dont include the first click
                else :
                    end_index = data[data["u"]==u].index[len(data[data["u"]==u])-1]+1 #Include the last index
                pure_organic_df = pd.concat([pure_organic_df,data.iloc[beginning_index:end_index]])
                # remove bandit events
                pure_organic_df = pure_organic_df.loc[pure_organic_df["z"] != 'bandit']
            data = pure_organic_df
            # add a reward in case of a sale and artificially define the action as the view
            data["y"] = (data["z"]=='sale')*1
            data["a"] = data["v"]
            # remove organic events followed by a sale to avoid double counting
            data = data.loc[data.shift(-1)["z"]!="sale"]
            self.data_rewards = data
            
        else :
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
                data_slice = data_slice.loc[data_slice["u"]==data_slice["u"].iloc[0]]
    
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
        if self.organic_only :
            name += "_pureorganic"
        return name

## (5) Attribute a reward if the product was sold later at one point during the user session
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


### Post display sales
class PostDisplayRewardProvider(FeatureProvider):
    """Reward shaping class"""

    def __init__(self,normalize=False):
        self.data_rewards = None
        # self.normalize = normalize
        
    def observe(self, data):
        # Consider only organic data
        data_orga = data.loc[data["z"] !='bandit']
        # reward of 1 if the view is followed by a sale
        data_orga["y"] = (data_orga.shift(-1)["z"]=="sale")*1
        data_orga = data_orga.loc[data_orga["z"] !='sale']
        data_orga["a"] = data_orga["v"] # artificially add views as actions
        # # per user normalization 
        # if self.normalize:
        #     nb_views_per_user = [sum(data_orga["u"]==i) for i in range(10)]
        #     data_orga["y"] = data_orga["y"]/np.array(nb_views_per_user)[np.array(data_orga["u"]).astype(int)]
        self.data_rewards = data_orga

    def features(self):
        return self.data_rewards
    
    def reset(self):
        self.data_rewards = None
        
    @property
    def name(self):
        name = "PostDisplayRewardProvider"
        return name


class DifferenceRewardProvider(FeatureProvider):
    """Reward shaping class"""

    def __init__(self,RewardProvider1, RewardProvider2, normalize=False):
        self.RewardProvider1 = RewardProvider1
        self.RewardProvider2 = RewardProvider2
        self.data_rewards = None
        self.normalize = normalize
        
    def observe(self, data):
        reward_model = self.RewardProvider1.observe(data)
        discount_model = self.RewardProvider2.observe(data)
        try:
            reward_features = reward_model.features()
        except: 
            print(reward_model)
            print(dir(reward_model))
        try:
            discount_features = discount_model.features()
        except:
            print(discount_features)
            print(dir(discount_features))
        # discount based on the second model
        per_user_discount = np.array(discount_features.groupby(["u"]).mean()["y"])
        if np.max(reward_features["y"] > 1):
            reward_features["y"] = (reward_features["y"] > 0)*1
        reward_features["y"] = reward_features["y"]- per_user_discount[reward_features["u"].astype(int)]
        # apply a transformation to obtain rewards between 0 and 1
        if self.normalize :
            reward_features["y"] = (1+reward_features["y"])/2
        self.data_rewards = reward_features

    def features(self):
        return self.data_rewards
    
    def reset(self):
        self.data_rewards = None
        
    @property
    def name(self):
        name = self.RewardProvider1.name + self.RewardProvider2.name
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
        self.user_features += np.array(views)
        
    def features(self):
        """Provide feature values adjusted to a particular feature set"""
        return self.user_features

    def reset(self):
        self.user_features = np.zeros(self.num_products)
        
    @property
    def weight(self):
        return np.sum(self.user_features)

    @property
    def name(self):
        return "CountViewsFeatureProvider"
    
    @property
    def feature_size(self):
        return self.num_products


## Consider share of views instead of counts
class ShareViewsFeatureProvider(FeatureProvider):
    """Feature provider as an abstract class that defines interface of setting/getting features
    The class counts both clicks and views """

    def __init__(self, config):
        super(ShareViewsFeatureProvider, self).__init__(config)
        self.num_products = config.num_products
        self.user_counts = np.zeros(self.num_products)
        self.user_features = np.zeros(self.num_products)
        
    def observe(self, data, memory = True):
        if type(data) == pd.core.series.Series:
            if data["z"]=="organic":
                views = [(data["v"]==p)*1 for p in range(self.num_products)]
            else :
                views = np.zeros(self.num_products)
        else :
            data_organic = data.loc[data["z"]=="organic"].loc[data["v"]>=0]
            views = [np.sum(data_organic["v"]==p) for p in range(self.num_products)]
        if not memory:
            self.user_counts = np.array(views)
        else:
            self.user_counts += np.array(views)
        nb_views = np.sum(self.user_counts)
        if nb_views>0:
            self.user_features = self.user_counts/nb_views
        
    def features(self):
        """Provide feature values adjusted to a particular feature set"""
        return self.user_features

    def reset(self):
        self.user_features = np.zeros(self.num_products)
        self.user_counts = np.zeros(self.num_products)

    @property
    def weight(self):
        return np.sum(self.user_counts)

    @property
    def name(self):
        return "CountViewsFeatureProvider"
    
    @property
    def feature_size(self):
        return self.num_products


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
                
        self.user_features = self.user_features + np.concatenate([self.view_feature,self.click_feature])
        
    def features(self):
        """Provide feature values adjusted to a particular feature set"""
        return self.user_features

    def reset(self):
        self.view_feature = np.zeros(self.num_products)
        self.click_feature = np.zeros(self.num_products)
        self.user_features = np.zeros(2*self.num_products)

    @property
    def weight(self):
        # Note : this could be defined differently
        return np.sum(self.view_count)+np.sum(self.click_count)
 

    @property
    def name(self):
        return "CountViewsClicksFeatureProvider"

    @property
    def feature_size(self):
        return 2*self.num_products
    
    
class ShareViewsClicksFeatureProvider(FeatureProvider):
    """Feature provider as an abstract class that defines interface of setting/getting features
    The class counts both clicks and views """

    def __init__(self, config):
        super(ShareViewsClicksFeatureProvider, self).__init__(config)
        self.num_products = config.num_products
        self.view_feature = np.zeros(self.num_products)
        self.click_feature = np.zeros(self.num_products)
        self.view_count = np.zeros(self.num_products)
        self.click_count = np.zeros(self.num_products)
        self.user_features = np.zeros(2*self.num_products)
        
    def observe(self, data):
        if type(data) == pd.core.series.Series:
            row = data
            if row["z"]=='organic':
                self.view_count[row["v"]] += 1 
            if (row["z"]=='bandit') & (row["c"]==1):
                self.click_count[row["a"]] += 1 
        else :
            for _, row in data.iterrows():
                if row["z"]=='organic':
                    self.view_count[int(row["v"])] += 1 
                if (row["z"]=='bandit') & (row["c"]==1):
                    self.click_count[int(row["a"])] += 1 
               
        nb_views = np.sum(self.view_count)
        if nb_views > 0:
            self.view_feature = self.view_count/nb_views
        nb_clicks = np.sum(self.click_count)
        if nb_clicks > 0:
            self.click_feature = self.click_count/nb_clicks
            
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
        return "CountViewsClicksFeatureProvider"

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
                 look_ahead = False, sample_weight = False, softmax = False, memory = False,
                 # multiply_features = False,
                 epsilon_greedy = False, epsilon = 0.3, seed=43):
        self.feature_provider = feature_provider
        self.reward_provider = reward_provider
        self.random_state = RandomState(seed)
        self.model = None
        # self.multiply_features = multiply_features
        # self.softmax = softmax
        self.memory = memory
        self.epsilon_greedy = epsilon_greedy
        self.epsilon = epsilon
        self.logged_observation = {'t': [],'u': [], 'z': [],'v': [], 'a': [],
                                   'c': [],'r': [],'ps': [], 'ps-a': []}
        self.kronecker_features = kronecker_features
        self.look_ahead = look_ahead
        self.sample_weight = sample_weight
        self.report_issue = {}
        self.performance_models = None
        
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
        
        
    def evaluate_model(self, logs):
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
        
        model = self.model
        pred_label = model.predict(features)
        performance = {'accuracy':accuracy_score(y_true=rewards,y_pred=pred_label),
                        'bal_accuracy':balanced_accuracy_score(y_true=rewards,y_pred=pred_label),
                        'f1':f1_score(y_true=rewards,y_pred=pred_label),
                        'recall':recall_score(y_true=rewards,y_pred=pred_label),
                        'precision':precision_score(y_true=rewards,y_pred=pred_label)}
        self.performance_models = performance 
        return self.performance_models
        
        
    
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
        self.feature_provider.observe(logged_observation,memory=self.memory)    
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
                 sample_weight = False, memory = False,
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
        self.memory = memory
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
        self.performance_models = None
        
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
                    # print('linear reg',self.linear_reg)
                    if self.linear_reg[i]==True:
                        model = FracLogisticRegression()
                        model.fit(features, rewards, sample_weight=weights)
                    else:
                        model = LogisticRegression(solver='lbfgs', max_iter=5000)
                        model.fit(features, rewards, sample_weight=weights)
            else :
                model = None
                
            self.models.append(model)
    
    def evaluate_models(self,logs):
        self.performance_models = []
        name_models = self.info['Name'].strip('_all').split('Provider_')
        for i in range(len(self.models)):
            print('--- Evaluate model : ',name_models[i])
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
                model = self.models[i]
                pred_label = model.predict(features)
                # print('predicted label:',pred_label)
                try:
                    performance = {'Name':name_models[i],
                                   'accuracy':accuracy_score(y_true=rewards,y_pred=pred_label),
                                   'bal_accuracy':balanced_accuracy_score(y_true=rewards,y_pred=pred_label),
                                   'f1':f1_score(y_true=rewards,y_pred=pred_label),
                                   'recall':recall_score(y_true=rewards,y_pred=pred_label),
                                   'precision':precision_score(y_true=rewards,y_pred=pred_label)}
                except:
                    clipped_proportion = np.clip(np.array(rewards),a_min=1e-10,a_max =1-1e-10)
                    true_label = logit(clipped_proportion)
                    performance = {'Name':name_models[i],
                                   'explained_variance':explained_variance_score(y_true=true_label,y_pred=pred_label),
                                   'max_error':max_error(y_true=true_label,y_pred=pred_label),
                                   'mean_absolute_error':mean_absolute_error(y_true=true_label,y_pred=pred_label),
                                   'mean_squared_error':mean_squared_error(y_true=true_label,y_pred=pred_label),
                                   'r2_score':r2_score(y_true=true_label,y_pred=pred_label)}

            else :
                performance = None

            self.performance_models.append(performance)
            
        return self.performance_models
    
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
                feature_provider.observe(logged_observation,memory=self.memory) 
                user_state = feature_provider.features()
                before_discount = self._score_products(user_state, 
                                                       self.models[self.index_discounted[i]])
                
                ## model used as discount
                feature_provider = self.feature_provider_list[self.index_discount[i]]
                feature_provider.observe(logged_observation,memory=self.memory)  
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
                feature_provider.observe(logged_observation,memory=self.memory)  
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
                 weights=False,  click=True, memory=False, repo = 'data/'):
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
                                                  sample_weight = weights,
                                                  memory = memory)
    likelihood_logreg_click.train(data)
    info[likelihood_logreg_click.info["Name"]] = likelihood_logreg_click.info
    if memory:
        info[likelihood_logreg_click.info["Name"]]["Name"] += '_mem'
    save_agents["likelihood_logreg_click"+name_extension] = likelihood_logreg_click
    
    # post display agent
    print("Post display agent")
    likelihood_logreg_postd = SaleLikelihoodAgent(feature, PostDisplay_rewards,
                                                  kronecker_features = kronecker_features,
                                                  sample_weight = weights,
                                                  memory=memory)
    likelihood_logreg_postd.train(data)
    info[likelihood_logreg_postd.info["Name"]] = likelihood_logreg_postd.info
    if memory:
        info[likelihood_logreg_postd.info["Name"]]["Name"] += '_mem'
    save_agents["likelihood_logreg_postd"+name_extension] = likelihood_logreg_postd
        
    # No discount, clicked observations
    
    if click :
        mdp_rew = MDP_rewards
    else:
        mdp_rew = MDP_rewards_all
        
    print("No discount")    
    likelihood_saleclickprod = SaleProductLikelihoodAgent(feature_provider_list=[feature,feature], 
                                                    reward_provider_list=[Click_rewards,mdp_rew], 
                                                    discounts=[0,0],discounts_with_action=False,
                                                    kronecker_features = kronecker_features,
                                                    sample_weight = weights,
                                                    memory=memory)
    likelihood_saleclickprod.train(data)
    info[likelihood_saleclickprod.info["Name"]] = likelihood_saleclickprod.info
    # if not click:
    #     print(likelihood_saleclickprod.info["Name"])
    if memory:
        info[likelihood_saleclickprod.info["Name"]]["Name"] += '_mem'
    save_agents["likelihood_saleclickprod"+name_extension] = likelihood_saleclickprod
    
      
    # non-specific discount, clicked observations
    print("Discount")    
    likelihood_saleclickprod_discount = SaleProductLikelihoodAgent(feature_provider_list=[feature,feature,feature], 
                                                    reward_provider_list=[Click_rewards,mdp_rew,Noclick_rewards_provider], 
                                                    discounts=[0,0,-1],discounts_with_action=False,
                                                    kronecker_features = kronecker_features,
                                                    sample_weight = weights,
                                                    memory=memory)
    likelihood_saleclickprod_discount.train(data)
    info[likelihood_saleclickprod_discount.info["Name"]] = likelihood_saleclickprod_discount.info
    # if not click:
    #     print(likelihood_saleclickprod_discount.info["Name"])
    #     # info[likelihood_saleclickprod_discount.info["Name"]]["Name"] += '_all'
    
    # Evaluate the submodels
    print('----- Evaluate submodels')
    data_test = logs[name_logging+'_test']
    performance = likelihood_saleclickprod_discount.evaluate_models(data_test)
    # print(likelihood_saleclickprod_discount.performance_models)
    
    if memory:
        info[likelihood_saleclickprod_discount.info["Name"]]["Name"] += '_mem'
    save_agents["likelihood_saleclickprod_discount"+name_extension] = likelihood_saleclickprod_discount
    
    pkl.dump([info,save_agents,performance],open(str(repo) + str('agents'+str(num_users)+name_logging+feature_name+name_extension+'.pkl'),'wb'))
    return info, save_agents




def train_timeagents(name_logging,logs,feature_name,features, num_users, kronecker_features=False, linear_reg=False,
                     weights=False, click=True, memory = False, repo = 'data/'):
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
   
    
    if click :
        mdp_rew = MDP_rewards_time
    else:
        mdp_rew = MDP_rewards_all_time
    
    # No discount
    print("No discount")
    likelihood_saleclickprod = SaleProductLikelihoodAgent(feature_provider_list=[feature,feature], 
                                                    reward_provider_list=[Click_rewards,mdp_rew], 
                                                    discounts=[0,0],discounts_with_action=False,
                                                    kronecker_features = kronecker_features,
                                                    linear_reg = [False,linear_reg],
                                                    sample_weight = weights,
                                                    memory=memory)
    likelihood_saleclickprod.train(data)
    info[likelihood_saleclickprod.info["Name"]] = likelihood_saleclickprod.info
    # if not click:
    #     print(likelihood_saleclickprod.info["Name"])
    #     # info[likelihood_saleclickprod.info["Name"]]["Name"] += '_all'
    if memory:
        info[likelihood_saleclickprod.info["Name"]]["Name"] += '_mem'
    save_agents["likelihood_saleclickprod"+name_extension] = likelihood_saleclickprod
    
    print("Discount")
    # non-specific discount, clicked observations
    likelihood_saleclickprod_discount = SaleProductLikelihoodAgent(feature_provider_list=[feature,feature,feature], 
                                                    reward_provider_list=[Click_rewards,mdp_rew,Noclick_rewards_provider_time], 
                                                    discounts=[0,0,-1],discounts_with_action=False,
                                                    kronecker_features = kronecker_features,
                                                    linear_reg = [False,linear_reg,linear_reg],
                                                    sample_weight = weights,
                                                    memory=memory)
    likelihood_saleclickprod_discount.train(data)
    info[likelihood_saleclickprod_discount.info["Name"]] = likelihood_saleclickprod_discount.info
    # if not click:
    #     print(likelihood_saleclickprod_discount.info["Name"])
    #     # info[likelihood_saleclickprod_discount.info["Name"]]["Name"] += '_all'
    
    
    # Evaluate the submodels
    print('----- Evaluate submodels')
    data_test = logs[name_logging+'_test']
    performance = likelihood_saleclickprod_discount.evaluate_models(data_test)
    print(likelihood_saleclickprod_discount.performance_models)
    
    if memory:
        info[likelihood_saleclickprod_discount.info["Name"]]["Name"] += '_mem'
    save_agents["likelihood_saleclickprod_discount"+name_extension] = likelihood_saleclickprod_discount
        
    
    try:
        pkl.dump([info,save_agents,performance],open(str(repo)+str('timeagents'+str(num_users)+name_logging+feature_name+name_extension+'.pkl'),'wb'))
    except:
        print(" //!!\\ Error when saving agents")
    return info, save_agents


