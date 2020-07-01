import numpy as np
from numba import njit

# from ..envs.configuration import Configuration
from ..envs.reco_env_v1_sale import env_1_args, ff, sig
from .abstract import Agent
from recogym.envs.reco_env_v1_sale import RecoEnv1Sale


@njit(nogil=True)
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


######################### The following oracles can be used if there is NO direct landing 
######################### On clicked products



class SaleOracleAgent(Agent, RecoEnv1Sale):
    """
    Sale Count Oracle 

    Has access to user features and product conversion features only
    The recommended product is  :
        argmax_{a} P(sale | a is recommended, a is clicked, a is viewed) - P(sale | a is not recommended, a is viewed)
    The oracle is incremental, it derives the impact of a reco compared to the impact in absence of reco
    This oracle can be used if there is no direct landing on the product's page after a click
    
    """

    def __init__(self, env):
        super(SaleOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done, info = None):
        """Make a recommendation"""
        self.delta = self.env.delta
        # Difference in dot product post- and before- click for each product
        embed_difference = np.array([(-self.delta[:,0]+self.Lambda[int(a),:]) @ self.Lambda[int(a),:] for a in range(self.env.config.num_products)])
        # the action is the argmax of the difference in the sale probability 
        action = np.argmax(embed_difference)
        self.list_actions.append(action)
        
        if self.env.config.with_ps_all:
            ps_all = np.zeros(self.config.num_products)
            ps_all[action] = 1.0
        else:
            ps_all = ()

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': self.env.config.psale_scale*sig(self.Lambda[int(action),:] @ (self.delta))[0],
                'ps-a': ps_all,
            },
        }

    def history(self):
        return self.list_actions

    def reset(self):
#         self.env.reset()
        self.list_actions = []
        self.delta = self.env.delta
        self.Lambda = self.env.Lambda
        self.list_actions = []


class ViewSaleOracleAgent(Agent, RecoEnv1Sale):
    """
    View Sale Oracle 

    Has access to user features and product Views + Conversion features 
    The recommended product is just :
        argmax_{a} P(a is viewed) [ P(sale | a is recommended, a is clicked, a is viewed) - P(sale | a is not recommended, a is viewed) ] 
    The oracle is incremental, it derives the impact of a reco compared to the impact in absence of reco
    This oracle can be used if there is no direct landing on the product's page after a click
    
    """

    def __init__(self, env):
        super(ViewSaleOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done, info = None):
        """Make a recommendation"""
        self.omega = self.env.omega
        self.delta = self.env.delta
        if ("delta_for_views" in dir(self.env.config) is not None) & (self.env.config.delta_for_views == True) :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        
        # Proba of viewing each product
        log_proba_view = np.array([self.user_feature_view[:,0]@self.Gamma[int(a),:] + self.mu_organic[int(a)] for a in range(self.env.config.num_products)])
        proba_view = np.exp(log_proba_view - max(log_proba_view))
        proba_view = proba_view / proba_view.sum()
        proba_view = proba_view[:,0]
        
        # Difference in sale mean for each product if the product is recommended
        proba_with_click = np.array([sig(((1-self.kappa)*self.delta[:,0] + self.kappa*self.Lambda[int(a),:])@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        proba_no_click = np.array([sig(self.delta[:,0]@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        
        proba_difference = proba_with_click - proba_no_click
        
        # Take argmax
        action = np.argmax(proba_view * proba_difference)
        self.list_actions.append(action)
        
        if self.env.config.with_ps_all:
            ps_all = np.zeros(self.config.num_products)
            ps_all[action] = 1.0
        else:
            ps_all = ()

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': self.env.config.psale_scale*sig(self.Lambda[int(action),:] @ (self.delta))[0],
                'ps-a': ps_all,
            },
        }

    def history(self):
        return self.list_actions

    def reset(self):
        self.list_actions = []
        self.kappa = self.env.config.kappa
        self.omega = self.env.omega
        self.delta = self.env.delta
        if ("delta_for_views" in dir(self.env.config) is not None) & (self.env.config.delta_for_views == True) :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        self.Lambda = self.env.Lambda
        self.Gamma = self.env.Gamma
        self.mu_organic = self.env.mu_organic
        self.list_actions = []



class ClickSaleOracleAgent(Agent, RecoEnv1Sale):
"""
    Click Sale Oracle 

    Has access to user features and product Click + Conversion features 
    The recommended product is just :
        argmax_{a} P(a is clicked)[P(sale | a is recommended, a is clicked, a is viewed) - P(sale | a is not recommended, a is viewed)]
    The oracle is incremental, it derives the impact of a reco compared to the impact in absence of reco
    This oracle can be used if there is no direct landing on the product's page after a click
    
    """

    def __init__(self, env):
        super(ClickSaleOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done, info = None):
        """Make a recommendation"""
        self.omega = self.env.omega
        self.delta = self.env.delta
        
        if ("delta_for_clicks" in dir(self.env.config)) & (self.env.config.delta_for_clicks == 1):
            self.user_feature_click = self.delta
        else :
            self.user_feature_click = self.omega
        

        # Proba of clicking for each product
        proba_click = np.array([ff(self.user_feature_click[:,0]@self.beta[int(a),:] + self.mu_bandit[int(a)]) for a in range(self.env.config.num_products)])
        proba_click = proba_click[:,0]
        
        # Difference in sale mean for each product if the product is recommended
        proba_with_click = np.array([sig(((1-self.kappa)*self.delta[:,0] + self.kappa*self.Lambda[int(a),:])@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        proba_no_click = np.array([sig(self.delta[:,0]@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        proba_difference = proba_with_click - proba_no_click
        
        # Take argmax
        action = np.argmax(proba_click * proba_difference)
        self.list_actions.append(action)
        
        if self.env.config.with_ps_all:
            ps_all = np.zeros(self.config.num_products)
            ps_all[action] = 1.0
        else:
            ps_all = ()

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': self.env.config.psale_scale*sig(self.Lambda[int(action),:] @ (self.delta))[0],
                'ps-a': ps_all,
            },
        }

    def history(self):
        return self.list_actions

    def reset(self):
        self.list_actions = []
        self.kappa = self.env.config.kappa
        self.omega = self.env.omega
        self.delta = self.env.delta
        
        if ("delta_for_clicks" in dir(self.env.config)) & (self.env.config.delta_for_clicks ==1) :
            self.user_feature_click = self.delta
        else :
            self.user_feature_click = self.omega
        self.beta = self.env.beta
        self.Lambda = self.env.Lambda
        self.mu_organic = self.env.mu_organic
        self.mu_bandit = self.env.mu_bandit
        self.list_actions = []




class ClickViewSaleOracleAgent(Agent, RecoEnv1Sale):
    """
    Click View Sale Oracle 

    Has access to user features and all product features (Views + Conversion + Click features)
    The recommended product is just :
        argmax_{a} P(a is viewed)P(a is clicked)[P(sale | a is recommended, a is clicked, a is viewed) - P(a is clicked)P(sale | a is not recommended, a is viewed)]
    The oracle is incremental, it derives the impact of a reco compared to the impact in absence of reco
    This oracle can be used if there is no direct landing on the product's page after a click
    
    """


    def __init__(self, env):
        super(ClickViewSaleOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done, info = None):
        """Make a recommendation"""
        self.omega = self.env.omega
        self.delta = self.env.delta
        
        if ("delta_for_views" in dir(self.env.config) is not None) & (self.env.config.delta_for_views == True) :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        
        if ("delta_for_clicks" in dir(self.env.config)) & (self.env.config.delta_for_clicks == 1):
            self.user_feature_click = self.delta
        else :
            self.user_feature_click = self.omega
        

        # Proba of clicking for each product
        proba_click = np.array([ff(self.user_feature_click[:,0]@self.beta[int(a),:] + self.mu_bandit[int(a)]) for a in range(self.env.config.num_products)])
        proba_click = proba_click[:,0]
        
        # Proba of viewing each product
        log_proba_view = np.array([self.user_feature_view[:,0]@self.Gamma[int(a),:] + self.mu_organic[int(a)] for a in range(self.env.config.num_products)])
        proba_view = np.exp(log_proba_view - max(log_proba_view))
        proba_view = proba_view / proba_view.sum()
        proba_view = proba_view[:,0]
        
        # Difference in sale mean for each product if the product is recommended
        proba_with_click = np.array([sig(((1-self.kappa)*self.delta[:,0] + self.kappa*self.Lambda[int(a),:])@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        proba_no_click = np.array([sig(self.delta[:,0]@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        proba_difference = proba_with_click - proba_no_click
        
        # Take argmax
        action = np.argmax(proba_view * proba_click * proba_difference)
        self.list_actions.append(action)
        
        if self.env.config.with_ps_all:
            ps_all = np.zeros(self.config.num_products)
            ps_all[action] = 1.0
        else:
            ps_all = ()

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': self.env.config.psale_scale*sig(self.Lambda[int(action),:] @ (self.delta))[0],
                'ps-a': ps_all,
            },
        }

    def history(self):
        return self.list_actions

    def reset(self):
        self.list_actions = []
        self.kappa = self.env.config.kappa
        self.omega = self.env.omega
        self.delta = self.env.delta
        
        if ("delta_for_views" in dir(self.env.config) is not None) & (self.env.config.delta_for_views == True) :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        
        if ("delta_for_clicks" in dir(self.env.config)) & (self.env.config.delta_for_clicks ==1) :
            self.user_feature_click = self.delta
        else :
            self.user_feature_click = self.omega
        self.beta = self.env.beta
        self.Lambda = self.env.Lambda
        self.Gamma = self.env.Gamma
        self.mu_organic = self.env.mu_organic
        self.mu_bandit = self.env.mu_bandit
        self.list_actions = []
        




####################################################################################################    
######################### The following oracles can be used if there is a direct landing 
######################### On clicked products

from ..envs.utils_sale import expected_sale_given_action_click

class ClickViewExpectSalesOracleAgent(Agent, RecoEnv1Sale):
    """
    Click View Expect Sale Oracle 

    Has access to all user and product features.
    The goal is to estimate the expected number of sale in the organic session following a reco,
    compared to the one if there is the same reco but no user embedding update. 
    For this, we maximize : 
        (E[#sales in next organic session given a click for a]-E[#sales in next organic session given a click for a without user update])*P(c=1|A=a)
    where :
    E[#sales in next organic session given a click for a] = P(buy a | A=a,c=1,view a) + 
                                              (E[length of organic session]-1)\sum_{product}P(view product)P(buy product | view product)
    
    The oracle is incremental, it derives the impact of a reco compared to the impact in absence of reco
    This oracle can be used if there is a direct landing on the product's page after a click
    
    """

    def __init__(self, env):
        super(ClickViewExpectSalesOracleAgent, self).__init__(env)
        self.env = env
        self.p_transition_out_of_organic = env.config.prob_leave_organic + env.config.prob_organic_to_bandit

    def act(self, observation, reward, done, info = None):
        """Make a recommendation"""
        self.omega = self.env.omega
        self.delta = self.env.delta
        
        if ("delta_for_clicks" in dir(self.env.config)) & (self.env.config.delta_for_clicks == 1):
            self.user_feature_click = self.delta
        else :
            self.user_feature_click = self.omega
        

        # Proba of clicking for each product
        proba_click = np.array([ff(self.user_feature_click[:,0]@self.beta[int(a),:] + self.mu_bandit[int(a)]) for a in range(self.env.config.num_products)])
        proba_click = proba_click[:,0]
        
        # Difference in expectation whether the embedding gets updated or not
        expectation_difference = [expected_sale_given_action_click(self.env, a, user_update = True) 
                                  - expected_sale_given_action_click(self.env, a, user_update = False) for a in range(self.env.config.num_products)]
        
        # Take argmax
        action = np.argmax(proba_click *  expectation_difference)
        self.list_actions.append(action)
        
        if self.env.config.with_ps_all:
            ps_all = np.zeros(self.config.num_products)
            ps_all[action] = 1.0
        else:
            ps_all = ()

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': self.env.config.psale_scale*sig(self.Lambda[int(action),:] @ (self.delta))[0],
                'ps-a': ps_all,
            },
        }

    def history(self):
        return self.list_actions

    def reset(self):
        self.list_actions = []
        self.omega = self.env.omega
        self.delta = self.env.delta
        
        if ("delta_for_views" in dir(self.env.config) is not None) & (self.env.config.delta_for_views == True) :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        
        if ("delta_for_clicks" in dir(self.env.config)) & (self.env.config.delta_for_clicks ==1) :
            self.user_feature_click = self.delta
        else :
            self.user_feature_click = self.omega
        self.Lambda = self.env.Lambda
        self.beta = self.env.beta
        self.mu_bandit = self.env.mu_bandit
        self.list_actions = []

        
        
        
        
class ClickViewExpectGhostSalesOracleAgent(Agent, RecoEnv1Sale):
     """
    Click View Expect Ghost Sale Oracle 

    Has access to all user and product features.
    The goal is to estimate the expected number of sale in the organic session following a reco,
    compared to the one if only ghost ads are shown during the length of the organic session we compare it with.
    If ghost ads are shown, the user needs to transition to the organic state by himself
    For this, we maximize : 
        (E[#sales in next organic session given a click for a]-E[#sales in next organic session given ghost ads])*P(c=1|A=a)
    where :
    E[#sales in next organic session given a click for a] = P(buy a | A=a,c=1,view a) + 
                                              (E[length of organic session]-1)\sum_{product}P(view product)P(buy product | view product)
    E[#sales in next organic session given ghost ads] = E[#steps in organic]E[#sales|organic state]
    
    The oracle is incremental, it derives the impact of a reco compared to the impact in absence of reco
    This oracle can be used if there is a direct landing on the product's page after a click
    
    """

    def __init__(self, env):
        super(ClickViewExpectGhostSalesOracleAgent, self).__init__(env)
        self.env = env
        self.p_transition_out_of_organic = env.config.prob_leave_organic + env.config.prob_organic_to_bandit

    def act(self, observation, reward, done, info = None):
        """Make a recommendation"""
        self.omega = self.env.omega
        self.delta = self.env.delta
        
        if ("delta_for_clicks" in dir(self.env.config)) & (self.env.config.delta_for_clicks == 1):
            self.user_feature_click = self.delta
        else :
            self.user_feature_click = self.omega
        

        # Proba of clicking for each product
        proba_click = np.array([ff(self.user_feature_click[:,0]@self.beta[int(a),:] + self.mu_bandit[int(a)]) for a in range(self.env.config.num_products)])
        proba_click = proba_click[:,0]
        
        # expectation of sales
        expectation_click = np.array([expected_sale_given_action_click(self.env, a, user_update = True) for a in range(self.env.config.num_products)])
        expectation_noclick = np.array([expected_sale_given_action_click(self.env, a, user_update = False) for a in range(self.env.config.num_products)])
        # probability of transition from bandit to organic
        p_bo = self.env.config.prob_bandit_to_organic
        nb_steps_left = np.array([int(1/self.p_transition_out_of_organic)-i for i in range(1,int(1/self.p_transition_out_of_organic))])
        p_bo_power = np.array([p_bo**i for i in range(1,int(1/self.p_transition_out_of_organic))])
        Cp_bo_power = np.array([(1-p_bo)**i for i in range(0,int(1/self.p_transition_out_of_organic)-1)])
        # Expected number of steps in the organic state
        proba_to_organic_after_p_steps = nb_steps_left*p_bo_power*Cp_bo_power
        # Expected number of sales given the organic state, and no click (no user embedding update)
        expectation_noclick = np.sum(proba_to_organic_after_p_steps)*expectation_noclick
        
        # Take argmax
        action = np.argmax((proba_click *  expectation_click)+((1-proba_click)*expectation_noclick))
        self.list_actions.append(action)
        
        if self.env.config.with_ps_all:
            ps_all = np.zeros(self.config.num_products)
            ps_all[action] = 1.0
        else:
            ps_all = ()

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': self.env.config.psale_scale*sig(self.Lambda[int(action),:] @ (self.delta))[0],
                'ps-a': ps_all,
            },
        }

    def history(self):
        return self.list_actions

    def reset(self):
        self.list_actions = []
        self.omega = self.env.omega
        self.delta = self.env.delta
        
        if ("delta_for_views" in dir(self.env.config) is not None) & (self.env.config.delta_for_views == True) :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        
        if ("delta_for_clicks" in dir(self.env.config)) & (self.env.config.delta_for_clicks ==1) :
            self.user_feature_click = self.delta
        else :
            self.user_feature_click = self.omega
        self.Lambda = self.env.Lambda
        self.beta = self.env.beta
        self.mu_bandit = self.env.mu_bandit
        self.list_actions = []