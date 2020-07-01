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

class SaleCountOracleAgent(Agent, RecoEnv1Sale):
    """
    Sale Count Oracle 

    Has access to user features and product conversion features only
    The recommended product is just argmax_{a} P(sale | a is recommended, a is clicked, a is viewed)
    The oracle just compares the different products, not the incremental increase of the probability of sale
    This oracle can be used if there is no direct landing on the product's page after a click
    
    """

    def __init__(self, env):
        super(SaleCountOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done, info=None):
        """Make a recommendation"""
        self.delta = self.env.delta
        embed = np.array([((1-self.kappa)*self.delta[:,0] + self.kappa*self.Lambda[int(a),:]) @ self.Lambda[int(a),:] for a in range(self.env.config.num_products)])
        # It is enough to take the argmax on the embedding dot products since the mapping to a proba is increasing
        action = np.argmax(embed)
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
        self.delta = self.env.delta
        self.Lambda = self.env.Lambda
        self.kappa = self.env.config.kappa
        self.list_actions = []


class ViewSaleCountOracleAgent(Agent, RecoEnv1Sale):
  """
    View Sale Count Oracle 

    Has access to user features and product Views + Conversion features 
    The recommended product is just argmax_{a} P(a is viewed)P(sale | a is recommended, a is clicked, a is viewed)
    The oracle just compares the different products, not the incremental increase of the probability of sale
    This oracle can be used if there is no direct landing on the product's page after a click
    
    """
    
    def __init__(self, env):
        super(ViewSaleCountOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done, info=None):
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
        
        # sale mean for each product if the product is recommended
        proba_with_click = np.array([sig(((1-self.kappa)*self.delta[:,0] + self.kappa*self.Lambda[int(a),:])@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        
        # Take argmax
        action = np.argmax(proba_view * proba_with_click)
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





class ClickSaleCountOracleAgent(Agent, RecoEnv1Sale):
    """
    Click Sale Count Oracle 

    Has access to user features and product Click + Conversion features 
    The recommended product is just argmax_{a} P(a is clicked)P(sale | a is recommended, a is clicked, a is viewed)
    The oracle just compares the different products, not the incremental increase of the probability of sale
    This oracle can be used if there is no direct landing on the product's page after a click
    
    """


    def __init__(self, env):
        super(ClickSaleCountOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done, info=None):
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
        
        # sale mean for each product if the product is recommended
        proba_with_click = np.array([sig(((1-self.kappa)*self.delta[:,0] + self.kappa*self.Lambda[int(a),:])@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        
        # Take argmax
        action = np.argmax(proba_with_click * proba_click)
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
        
        

        
        
        
class ClickViewSaleCountOracleAgent(Agent, RecoEnv1Sale):
    """
    Click View Sale Count Oracle 

    Has access to user features and all product features (Views + Conversion + Click features)
    The recommended product is just argmax_{a} P(a is viewed)P(a is clicked)P(sale | a is recommended, a is clicked, a is viewed)
    The oracle just compares the different products, not the incremental increase of the probability of sale
    This oracle can be used if there is no direct landing on the product's page after a click
    
    """

    def __init__(self, env):
        super(ClickViewSaleCountOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done, info=None):
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
        
        # sale mean for each product if the product is recommended
        proba_with_click = np.array([sig(((1-self.kappa)*self.delta[:,0] + self.kappa*self.Lambda[int(a),:])@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        
        # Take argmax
        action = np.argmax(proba_view * proba_with_click * proba_click)
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
        
class ClickViewExpectSalesCountOracleAgent(Agent, RecoEnv1Sale):
    """
    Click View Expect Sale Count Oracle 

    Has access to all user and product features.
    The goal is to estimate the expected number of sale in the organic session following a reco,
    divided by the probability of not clicking on the reco. 
    For this, we maximize : 
        E[#sales in next organic session given a click for a]*P(c=1|A=a)/P(c=0|A=a)
    where :
    E[#sales in next organic session given a click for a] = P(buy a | A=a,c=1,view a) + 
                                              (E[length of organic session]-1)\sum_{product}P(view product)P(buy product | view product)
    The ratio of probabilities may become very large, thus an upper bound M on the ratio can be provided
    
    The oracle just compares the different products, not the incremental increase of the probability of sale
    This oracle can be used if there is a direct landing on the product's page after a click
    
    """

    def __init__(self, env, M=1e3):
        super(ClickViewExpectSalesCountOracleAgent, self).__init__(env)
        self.env = env
        self.p_transition_out_of_organic = env.config.prob_leave_organic + env.config.prob_organic_to_bandit
        self.M = M

    def act(self, observation, reward, done, info=None):
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
        
        # Expectated sales if the embedding gets updated 
        expectation = [expected_sale_given_action_click(self.env, a, user_update = True) for a in range(self.env.config.num_products)]
        
        # Take argmax
        action = np.argmax(np.clip(proba_click/(1-proba_click), a_min=0, a_max=self.M) *  expectation)
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