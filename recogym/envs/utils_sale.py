import numpy as np
from numpy.random import choice
from recogym import Configuration, DefaultContext, Observation
from recogym.envs.session import OrganicSessions
from recogym.agents import Agent, FeatureProvider
from numpy.random.mtrand import RandomState
from sklearn.linear_model import LogisticRegression
# import pandas as pd
from copy import deepcopy
from scipy.stats.distributions import beta
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from recogym.util import FullBatchLBFGS
import pandas as pd
from IPython.display import display



P = 10  # Number of Products
U = 2000 # Number of Users

def share_states(data):
    ''' Dictionary with share of time spent in each state'''
    return {'organic' : np.sum(data["z"]=="organic")/len(data), 
            'bandit' : np.sum(data["z"]=="bandit")/len(data),
            'sale' : np.sum(data["z"]=="sale")/len(data)}

def share_sale(data):
    ''' Returns two metrics : 
        - sale_bin : Number of sale events/Number of recos
        - sal_tot : Total number of sales/Number of recos'''
    return {'sale_bin' : np.sum((data["z"]=='bandit') & (data["r"]>0))/np.sum(data["z"]=='bandit'), 
            'sale_tot' : np.sum(data["r"])/np.sum(data["z"]=='bandit')}

def env_infos(env):
    env = deepcopy(env)
    return {'Gamma' : env.Gamma, 'Lambda' : env.Lambda, 'beta' : env.beta, 
            'omega' : env.omega, 'user_ps' : env.user_ps_list,
           'proba_sales' : env.proba_sales, 'proba_sales_after_scaling' : env.proba_sales_after_scaling}

def count_sales_first_session(data):
    ''' Given a dataset, counts the number of times a sale happens before the first reco'''
    sales_first_session = {}
    no_event=[]
    for user in data['u'].unique() :
        data_user = data.loc[data['u']==user]
        try :
            first_bandit_index = data_user[data_user['z']=="bandit"].index[0]
            first_sale_index = data_user[data_user['z']=="sale"].index[0]
            sales_first_session[user]=first_sale_index<first_bandit_index
        except :
            no_event.append(user)
            
    return sales_first_session, no_event

def check_sales(agent, env, num_products=10):

    env.reset_random_seed()

    # Train on 1000 users offline.
    num_offline_users = 1000

    for _ in range(num_offline_users):

        # Reset env and set done to False.
        env.reset()
        done = False

        observation, reward, done, info = None, 0, False, {}
        while not done:
            old_observation = observation
            action, observation, reward, done, info = env.step_offline(observation, reward, done, info)
            agent.train(old_observation, action, reward, done)

    # Train on 100 users online and track click through rate.
    num_online_users = 100
    num_sales, num_events = 0, 0
    list_sales = []
    list_proba_sales = []
    list_proba_sales_after_scaling = []

    for i in range(num_online_users):

        # Reset env and set done to False.
        env.reset()
        observation, _, done, _ = env.step(None)
        reward = None
        done = None
        list_sales.append([])
        while not done:
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action['a'])
            list_sales[i].append(reward)

            # Used for calculating share of sales
            num_sales += reward if reward is not None else 0
            num_events += 1
        list_proba_sales.append(deepcopy(env).proba_sales)
        list_proba_sales_after_scaling.append(deepcopy(env).proba_sales_after_scaling)

    share_sales = num_sales / num_events
    print("Total number of sales:",num_sales)
    print(f"Share of sales (Nb sales/Nb recos): {share_sales:.4f}")

    return num_sales, share_sales, list_sales, list_proba_sales, list_proba_sales_after_scaling






# from https://github.com/criteo-research/bandit-reco/
class SingleActionAgent(Agent): 
    def __init__(self, preferred_action, config = Configuration({'num_products': 10})):
        Agent.__init__(self, config)
        self.preferred_action = preferred_action
        
    def act(self, observation, reward, done, info=None):
        probabilities = np.zeros(self.config.num_products)
        probabilities[self.preferred_action] = 1.
        return {
            **super().act(observation, reward, done),
            **{
                'a': self.preferred_action,
                'ps': probabilities[self.preferred_action],
                'ps-a': probabilities,
            },
        }
    
    
# (slightly modified) from https://github.com/criteo-research/bandit-reco/
class PopularityAgent(Agent):
    def __init__(self, config):
        # Set number of products as an attribute of the Agent.
        super(PopularityAgent, self).__init__(config)

        # Track number of times each item viewed in Organic session.
        self.organic_views = np.zeros(self.config.num_products)

    def logging_pretrain(self, data):
        """from logs"""
        self.organic_views = [np.sum(data["v"]==p) for p in range(self.config.num_products)]
    
    def train(self, observation, action, reward, done):
        """Train method learns from a tuple of data.
            this method can be called for offline or online learning"""
        
        # Adding organic session to organic view counts.
        if observation:
            for session in observation.sessions():
                self.organic_views[session['v']] += 1

    def act(self, observation, reward, done, info=None):
        """Act method returns an action based on current observation and past
            history"""
        # Choosing action randomly in proportion with number of views.
        prob = np.ones(len(self.organic_views))/len(self.organic_views) if sum(self.organic_views)==0 else self.organic_views / sum(self.organic_views)
        action = choice(self.config.num_products, p = prob)

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': prob[action]
            }
        }
    
    
    




# from https://github.com/criteo-research/bandit-reco/
def get_beta_confidence_interval(n_impressions, n_clicks):
    n_unclicked = n_impressions - n_clicks
    low_quantile = beta.ppf(0.025, n_clicks + 1, n_unclicked + 1)
    median = beta.ppf(0.500, n_clicks + 1, n_unclicked + 1)
    high_quantile = beta.ppf(0.975, n_clicks + 1, n_unclicked + 1)
    return median - low_quantile, high_quantile - median
    
# from https://github.com/criteo-research/bandit-reco/
class CountFeatureProvider(FeatureProvider):
    """Feature provider as an abstract class that defines interface of setting/getting features"""

    def __init__(self, config):
        super(CountFeatureProvider, self).__init__(config)
        self.feature_data = np.zeros((self.config.num_products))

    def observe(self, observation):
        """Consider an Organic Event for a particular user"""
        for session in observation.sessions():
            self.feature_data[int(session['v'])] += 1

    def features(self, observation):
        """Provide feature values adjusted to a particular feature set"""
        return self.feature_data

    def reset(self):
        self.feature_data = np.zeros((self.config.num_products))
        
# from https://github.com/criteo-research/bandit-reco/
def build_train_data(logs, feature_provider):
    user_states, actions, rewards, proba_actions = [], [], [], []

    current_user = None
    for _, row in logs.iterrows():
        if current_user != row['u']:
            # User has changed: start a new session and reset user state.
            current_user = row['u']
            sessions = OrganicSessions()
            feature_provider.reset()

        context = DefaultContext(row['u'], row['t'])

        if (row['z'] == 'organic') or (row['z'] == 'sale'):
            sessions.next(context, row['v'])

        else:
            # For each bandit event, generate one observation for the user state, 
            # the taken action the obtained reward and the used probabilities.
            feature_provider.observe(Observation(context, sessions))
            user_states.append(feature_provider.features(None).copy())
            actions.append(row['a'])
            rewards.append(row['r'])
            proba_actions.append(row['ps'])

            # Start a new organic session.
            sessions = OrganicSessions()
    return np.array(user_states), np.array(actions).astype(int), np.array(rewards), np.array(proba_actions)



# from https://github.com/criteo-research/bandit-reco/
class ProductCountFeatureProvider(FeatureProvider):
    """This feature provider creates a user state based on viewed product count.
    Namely, the feature vector of shape (n_products, ) contains for each product how many times the
    user has viewed them organically.
    """

    def __init__(self, config):
        super(ProductCountFeatureProvider, self).__init__(config)
        self.feature_data = np.zeros((self.config.num_products)).astype(int)

    def observe(self, observation):
        for session in observation.sessions():
            self.feature_data[int(session['v'])] += 1

    def features(self, observation):
        return self.feature_data.copy()

    def reset(self):
        self.feature_data[:] = 0

# from https://github.com/criteo-research/bandit-reco/
class LikelihoodAgent(Agent):
    def __init__(self, feature_provider, epsilon_greedy = False, epsilon = 0.3, seed=43):
        self.feature_provider = feature_provider
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
        user_states, actions, rewards, proba_actions = build_train_data(logs, self.feature_provider)
        rewards = (rewards > 0)*1
        # estimate sales rate (boolean)
        count_actions = np.unique(actions,return_counts = True)[1]
        assert len(count_actions) == self.num_products
        count_sales_bool = np.array([len(np.where((actions==_) & (rewards>0))[0]) for _ in range(self.num_products)])
        self.salesrate = count_sales_bool / count_actions
        print("Estimated sales rate : ",self.salesrate)
        
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
        
    def act(self, observation, reward, done, info=None):
        """Act method returns an action based on current observation and past history"""
        self.feature_provider.observe(observation)        
        user_state = self.feature_provider.features(observation)
        
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
        
  # from https://github.com/criteo-research/bandit-reco/         
def build_rectangular_data(logs, feature_provider):
    """Create a rectangular feature set from the logged data.
    For each taken action, we compute the state in which the user was when the action was taken
    """
    user_states, actions, rewards, proba_actions = [], [], [], []
    
    current_user = None
    for _, row in logs.iterrows():
        if current_user != row['u']:
            # Use has changed: start a new session and reset user state
            current_user = row['u']
            sessions = OrganicSessions()
            feature_provider.reset()
        
        context = DefaultContext(row['u'], row['t'])
        
        if (row['z'] == 'organic') or (row['z'] == 'sale'):
            sessions.next(context, row['v'])
            
        else:
            # For each bandit event, generate one observation for the user state, the taken action
            # the obtained reward and the used probabilities
            feature_provider.observe(Observation(context, sessions))
            user_states += [feature_provider.features(None)] 
            actions += [row['a']]
            rewards += [row['r']]
            proba_actions += [row['ps']] 
            
            # Start a new organic session
            sessions = OrganicSessions()
    
    return np.array(user_states), np.array(actions).astype(int), np.array(rewards), np.array(proba_actions)
     
        
   # from https://github.com/criteo-research/bandit-reco/     
class MultinomialLogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, init_zeros=True):
        torch.nn.Module.__init__(self)
        # Generate weights - initialise randomly
        self.weight = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
        if init_zeros == True :
            torch.nn.init.zeros_(self.weight)
        else :
            torch.nn.init.kaiming_uniform_(self.weight, a = np.sqrt(5))


    def forward(self, x):
        inner_product = F.linear(x, self.weight)
        return F.softmax(inner_product, dim = 1)


# from https://github.com/criteo-research/bandit-reco/
class VanillaContextualBandit(Agent):
    def __init__(self, config, U = U, P = P, clipping_value = 1e5, max_epoch=30, init_zeros = True):
        Agent.__init__(self, config)
        self.model = MultinomialLogisticRegressionModel(P, P, init_zeros)
        self.loss_history = []
        self.user_state = np.zeros(P)
        self.U = U
        self.P = P
        self.max_epoch = max_epoch
        self.clipping_value = clipping_value

    def loss(self, X, a, proba_logged_actions, r):
        # Compute action predictions for clicks
        predicted_proba_for_all_actions = self.model(X)
        
        # Only keep probabilities for the actions that were taken
        predicted_proba = torch.gather(predicted_proba_for_all_actions, 1, a.unsqueeze(1)).reshape(-1)
        
        # expectation of the rewards under the new policy
        rewards = predicted_proba / proba_logged_actions
        
        # code here
        # We can cap the weights here (or equivalently the rewards in our case)
        # to some value of our choosing
        rewards = torch.clamp(input = rewards, max = self.clipping_value)
        
        # Since pytorch is meant to perform convex optimization, we rather
        # output a loss that we will want to minimize
        loss = - rewards.mean()
        return loss

    def train(self, rectangular_logs):
        """Train the contextual bandit based on an offline log such that it 
        learns to minimize its loss function
        """
        user_states, actions, rewards, proba_actions = rectangular_logs
        X = user_states
        a = actions
        p = proba_actions
        r = (rewards>0)*1
        
        # Put into PyTorch variables - drop unsaled samples
        X = Variable(torch.Tensor(X[r != 0]))
        a = Variable(torch.LongTensor(a[r != 0]))
        w = torch.Tensor(p[r != 0])
        
        def closure():
            # Reset gradients
            optimiser.zero_grad()

            return self.loss(X, a, w, r)
        
        # Set up optimiser
        optimiser = FullBatchLBFGS(self.model.parameters())

        # Initial loss
        self.loss_history.append(closure())
        max_epoch = self.max_epoch
        for epoch in range(max_epoch):
            # Optimisation step
            obj, _, _, _, _, _, _, _ = optimiser.step({'closure': closure,
                                                       'current_loss': self.loss_history[-1],
                                                       'max_ls': 20})
            self.loss_history.append(obj)
        
        return

    def plot_loss_history(self):
        ''' Plot the training loss over epochs '''
        _,_ = plt.subplots()
        plt.plot(range(len(self.loss_history)),self.loss_history)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        
    def observe(self, observation):
        ''' Observe new organic views and capture them in the user state '''
        for session in observation.sessions():
            self.user_state[int(session['v'])] += 1

    def act(self, observation, reward, done, info=None):
        ''' Pick an action, based on the current observation and the history '''
        # Observe
        self.observe(observation)

        # Act
        p_a = self.model(torch.Tensor([self.user_state])).detach().numpy().ravel()
        action = np.argmax(p_a)
        prob = np.zeros_like(p_a)
        prob[action] = 1.0

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': 1.0,
                'ps-a': prob,
            },
        }

    def reset(self):
        ''' Reset the user state '''
        self.user_state = np.zeros(self.P)
        
        
# from https://github.com/criteo-research/bandit-reco/
class LogContextualBandit(VanillaContextualBandit):

    def loss(self, X, a, proba_logged_actions, r):
        # Compute action predictions for sales
        predicted_proba_for_all_actions = self.model(X)
        
        # Only keep probabilities for the actions that were taken
        predicted_proba = torch.gather(predicted_proba_for_all_actions, 1, a.unsqueeze(1)).reshape(-1)
        
        # expectation of the reward under the new policy
        # code here
        reward = predicted_proba / proba_logged_actions
        
        loss = -reward

        return loss.mean()
    
    
 # from https://github.com/criteo-research/bandit-reco/   
class PoemContextualBandit(VanillaContextualBandit):
    def __init__(self, config, U = U, P = P, max_epoch=30, variance_penalization_factor=0., init_zeros = False):
        VanillaContextualBandit.__init__(self, config, U=U, P=P, max_epoch=max_epoch, init_zeros = init_zeros)
        self.variance_penalization_factor = variance_penalization_factor

    def loss(self, X, a, proba_logged_actions, r):
        # Compute action predictions for clicks
        predicted_proba_for_all_actions = self.model(X)
        
        # Only keep probabilities for the actions that were taken
        predicted_proba = torch.gather(predicted_proba_for_all_actions, 1, a.unsqueeze(1)).reshape(-1)
        
        # expectation of the loss under the new policy
        # code here
        # Note: have a look at torch.sqrt and torch.var
        reward = predicted_proba / proba_logged_actions
        loss = - reward.mean() + self.variance_penalization_factor*torch.sqrt(torch.var(reward)/predicted_proba.size()[0])
        return loss
    
    
    
def share_user_with_sale(data):
    '''Input : data (the logs)
    Output : share of users with at least one sale over the entire episode'''
    grouped_data = data[["r","u"]].groupby("u").sum()["r"]
    share = sum(grouped_data>0)/len(data["u"].unique())
    return share

def log_count_organic_session(data):
    '''
    Input : data (the logs, pandas dataframe)
    Output : the same dataframe, where each session has been numbered (for each user)
    '''
    data_ = data.copy()
    list_organic_sessions = []
    for user in data["u"].unique():
        data_user = data_[data_["u"]==user]
        count_organic_session = 1
        prev_row = "organic"
        for row in range(len(data_user)):
            next_row = data_user["z"].iloc[row]
            if (prev_row == 'bandit') & ((next_row == 'sale') | (next_row == 'organic')):
                count_organic_session += 1
            prev_row = next_row
            list_organic_sessions.append(count_organic_session)
    data_["organic_session"] = list_organic_sessions
    return data_

def share_sale_over_click_session(data):
    '''
    Input : data (the logs)
    Output : (Number of organic sessions with at least one sale / Number of clicks ,
                Number of organic sessions with at least one sale / Number of organic sessions)
    '''
    # Create dataframe that counts the number of organic sessions for each user
    data_ = log_count_organic_session(data)
    # add a boolean for whether there is sale or not
    data_["sale_bool"] = (data_["z"]=='sale')*1
    # Count number of sales  per organic session and user
    sales_per_organic_session = data_[["u","organic_session","sale_bool"]].groupby(["u","organic_session"]).sum()
    # Count number of sales event (dummie) per organic session and user
    sales_per_organic_session["sale_bool"] = (sales_per_organic_session["sale_bool"]>0)*1
    sales_per_organic_session['organic_session'] = [sales_per_organic_session.index[i][1] for i in range(len(sales_per_organic_session))]
    # Don't count sales that may have happened during the first organic session (before a click)
    sales_per_organic_session["sale_bool"] = sales_per_organic_session["sale_bool"]*(sales_per_organic_session["organic_session"]!= 1)
    # Total number of organic sessions (all users)
    nb_organic_sessions = len(sales_per_organic_session)
    # Share of organic sessions that include at least one sale event (all users)
    share_session_with_sale = sum(sales_per_organic_session["sale_bool"])/nb_organic_sessions
    # Number of organic sessions that include at least one sale event over number of clicks (all users)
    share_clicks_with_sale = sum(sales_per_organic_session["sale_bool"])/sum(data["c"]==1)
    
    return share_clicks_with_sale, share_session_with_sale

def share_clicks_with_sale(data):
    ''' Number of clicks & sales / Number of clicks'''
    return len(data[(data["c"]==1) & (data["r"]>0)])/sum(data["c"]==1)  
    
from numba import njit
@njit(nogil=True)
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def expected_sale_given_action_click(env, action, user_update = True):
    ''' Estimates the proba of sale in the next organic session, after a clicked reco for product a
    Inputs : 
        env : the env
        action : the reco
        user_update : whether to update the user embedding after the clicked reco
    Output : 
        the expected number of sales in the following organic session, derived as :
            E[#sales in next organic session given a click for a] = P(buy a | A=a,c=1,view a) + 
                                              (E[length of organic session]-1)\sum_{product}P(view product)P(buy product | view product)
    '''
    proba_nomore_organic = env.config.prob_organic_to_bandit + env.config.prob_leave_organic
    
    if ("delta_for_views" in dir(env.config) is not None) & (env.config.delta_for_views == True) :
        user_feature_view = env.delta
    else :
        user_feature_view = env.omega
    
    # Proba of viewing each product
    log_proba_view = np.array([user_feature_view[:,0]@env.Gamma[int(a),:] + env.mu_organic[int(a)] for a in range(env.config.num_products)])
    proba_view = np.exp(log_proba_view - max(log_proba_view))
    proba_view = proba_view / proba_view.sum()
    proba_view = proba_view[:,0]
    
    # Difference in sale mean for each product if the product is recommended
    if user_update == True : 
        proba_with_clicked_action = np.array([sig(((1-env.config.kappa)*env.delta[:,0] + env.config.kappa*env.Lambda[int(action),:])@env.Lambda[int(a),:]) for a in range(env.config.num_products)])
    else :
        proba_with_clicked_action = np.array([sig((env.delta[:,0])@env.Lambda[int(a),:]) for a in range(env.config.num_products)])    
    
    E = proba_with_clicked_action[int(action)] + ((1/proba_nomore_organic)-1)*np.sum(proba_view*proba_with_clicked_action)
    
    return E




def avg_result(res_dict):
    std = {}
    for f in list(res_dict.keys()):
        r= deepcopy(res_dict[f])
        agents = list(r['CTR']['Agent'])
        if f == list(res_dict.keys())[0]:
            res_avg = deepcopy(r)
        std[f] = {f:{}}
        for m in list(res_avg.keys())[:5]:
            res_avg[m]['Agent'] = agents
            res_avg[m]['Mean'] = 0
            res_avg[m]['std'] = 0
            res_avg[m]['se'] = 0
            std[m] = []
    for nb in list(res_dict.keys()):
        r= deepcopy(res_dict)
        for m in list(res_avg.keys())[:5]:
            if m in ['CTR','Share user with sale ATT', 'Share user with sale']:
                mean = r[nb][m]['0.500'].loc[[agent in agents for agent in list(r[nb][m]["Agent"])]]
            elif m in ['Tot sales ATT']:
                mean = r[nb][m]['TotSalesAtt'][[agent in agents for agent in list(r[nb][m]["Agent"])]]
            else:
                mean = r[nb][m]['TotSales'][[agent in agents for agent in list(r[nb][m]["Agent"])]]
            mean = list(mean)
            res_avg[m]['Mean'] += mean
            std[m] += [mean]
            
    # for nb in list(res_dict.keys()):
    #     for m in list(res_avg.keys())[:5]:
    #         res_avg[m]['Mean'] = res_avg[m]['Mean']/len(r)
    #         res_avg[m]['std'] = [np.std([float(std[m][i][j]) for i in range(len(r))]) for j in range(len(std[m][0]))] 
    #         res_avg[m]['se'] = res_avg[m]['std']/np.sqrt(len(r))
    # for nb in list(res_dict.keys()):
    print("len r",len(r))
    for m in list(res_avg.keys())[:5]:
        res_avg[m]['Mean'] = res_avg[m]['Mean']/len(r)
        res_avg[m]['std'] = [np.std([float(std[m][i][j]) for i in range(len(r))]) for j in range(len(std[m][0]))] 
        res_avg[m]['se'] = res_avg[m]['std']/np.sqrt(len(r))
    return res_avg




def format_avg_result(res_avg,print_res=True):
    res_recap = pd.DataFrame(res_avg['CTR']['Agent'])
    res = deepcopy(res_avg)
    for m in list(res.keys())[:5]:
        if m in ['CTR']:
            res[m]['Mean'] = pd.Series(["{0:.3f}".format(val * 100) for val in res[m]["Mean"]], index = res[m].index)
            res[m]['std'] = pd.Series(["{0:.3f}".format(val * 100) for val in res[m]["std"]], index = res[m].index)
            res[m]['se'] = pd.Series(["{0:.3f}".format(val * 100) for val in res[m]["se"]], index = res[m].index)
            
            res[m]['res'] = res[m]['Mean'] +' ('+res[m]['std']+') %'
            res[m]['res_se'] = res[m]['Mean'] +' ('+res[m]['se']+') %'
        elif m in ['Share user with sale ATT','Share user with sale']:
            res[m]["Mean"] = pd.Series(["{0:.2f}".format(val * 100) for val in res[m]["Mean"]], index = res[m].index)
            res[m]["std"] = pd.Series(["{0:.2f}".format(val * 100) for val in res[m]["std"]], index = res[m].index)
            res[m]["se"] = pd.Series(["{0:.2f}".format(val * 100) for val in res[m]["se"]], index = res[m].index)
            
            res[m]['res'] = res[m]['Mean'] +' ('+res[m]['std']+') %'
            res[m]['res_se'] = res[m]['Mean'] +' ('+res[m]['se']+') %'
        else :
            res[m]["Mean"] = pd.Series([str(int(val)) for val in res[m]["Mean"]], index = res[m].index)
            res[m]["std"] = pd.Series([str(int(val)) for val in res[m]["std"]], index = res[m].index)
            res[m]["se"] = pd.Series([str(int(val)) for val in res[m]["se"]], index = res[m].index)
            
            res[m]['res'] = res[m]['Mean'] +' ('+res[m]['std']+')'
            res[m]['res_se'] = res[m]['Mean'] +' ('+res[m]['se']+')'
        res_recap[m] =  res[m]['res_se']
    if len(res_recap) == 5:
        res_recap['Agent'] = ['Rand','Click','PVS','PCS','DPCS']
    elif len(res_recap) == 4:
        res_recap['Agent'] = ['Rand','Click','PCS','DPCS']
    elif len(res_recap) == 3:
        res_recap['Agent'] = ['Rand','PCS','DPCS']
    res_recap.columns = ['Agent','CTR','Att Sales','Att CR','Sales','CR']
    if print_res:
        display(res_recap)
    # print(res_recap.to_latex(index = False))
    return res_recap, res_recap.to_latex(index = False)  


def avg_result_extended(res_dict):
    std = {}
    for f in list(res_dict.keys()):
        r= deepcopy(res_dict[f])
        agents = list(r['CTR']['Agent'])
        if f == list(res_dict.keys())[0]:
            res_avg = deepcopy(r)
        std[f] = {f:{}}
        for m in list(res_avg.keys())[:9]:
            res_avg[m]['Agent'] = agents
            res_avg[m]['Mean'] = 0
            res_avg[m]['std'] = -1
            res_avg[m]['se'] = -1
            std[m] = []
        for m in ['NDPC','DPCSO','DPCSN']:
            res_avg[m] = deepcopy(res_avg['CTR'])
            std[m] = []
            
    for nb in list(res_dict.keys()):
        r= deepcopy(res_dict)
        for m in list(res_avg.keys())[:9]:
            if m in ['CTR','Share user with sale ATT', 'Share user with sale']:
                mean = r[nb][m]['0.500'].loc[[agent in agents for agent in list(r[nb][m]["Agent"])]]
            elif m in ['Tot sales ATT']:
                mean = r[nb][m]['TotSalesAtt'][[agent in agents for agent in list(r[nb][m]["Agent"])]]
            elif m in ['Tot sales']:
                mean = r[nb][m]['TotSales'][[agent in agents for agent in list(r[nb][m]["Agent"])]]
            else:
                mean = r[nb][m]['Mean'][[agent in agents for agent in list(r[nb][m]["Agent"])]]
            mean = list(mean)
            res_avg[m]['Mean'] += mean
            std[m] += [mean]
        for m in ['NDPC','DPCSO','DPCSN']:
            mean_ctr = np.array(r[nb]['True CTR']['Mean'][[agent in agents for agent in list(r[nb]['CTR']["Agent"])]])
            mean_pcs = np.array(r[nb]['True PCS']['Mean'][[agent in agents for agent in list(r[nb]['CTR']["Agent"])]])
            if m =='NDPC':
                mean = mean_ctr*mean_pcs
                
            elif m == 'DPCSO':
                mean_os = np.array(r[nb]['True OS']['Mean'][[agent in agents for agent in list(r[nb]['CTR']["Agent"])]])
                mean = mean_ctr*(mean_pcs-mean_os)
                
            elif m == 'DPCSN':
                mean_ncs = np.array(r[nb]['True NCS']['Mean'][[agent in agents for agent in list(r[nb]['CTR']["Agent"])]])
                mean = mean_ctr*(mean_pcs-mean_ncs)
            mean = list(mean)
            res_avg[m]['Mean'] += mean
            std[m] += [mean]
            
    num_tests = len(res_dict)
    
    for m in ['CTR', 'Tot sales ATT', 'Share user with sale ATT', 'Tot sales', 'Share user with sale',
                     'True CTR','True PCS','True OS','True NCS','NDPC','DPCSO','DPCSN']:
        res_avg[m]['Mean'] = res_avg[m]['Mean']/num_tests
        res_avg[m]['std'] = [np.std([float(std[m][i][j]) for i in range(len(std[m]))]) for j in range(len(std[m][0]))] 
        res_avg[m]['se'] = res_avg[m]['std']/np.sqrt(num_tests)
    return res_avg

def format_avg_result_extended(res_avg,print_res=True):
    res_recap = pd.DataFrame(res_avg['CTR']['Agent'])
    res = deepcopy(res_avg)
    for m in ['CTR', 'Tot sales ATT', 'Share user with sale ATT', 'Tot sales', 'Share user with sale',
                         'True CTR','True PCS','True OS','True NCS','NDPC','DPCSO','DPCSN']:
        if m in ['CTR','True CTR']:
            res[m]['Mean'] = pd.Series(["{0:.3f}".format(val * 100) for val in res[m]["Mean"]], index = res[m].index)
            res[m]['std'] = pd.Series(["{0:.3f}".format(val * 100) for val in res[m]["std"]], index = res[m].index)
            res[m]['se'] = pd.Series(["{0:.3f}".format(val * 100) for val in res[m]["se"]], index = res[m].index)
            
            res[m]['res'] = res[m]['Mean'] +' ('+res[m]['std']+') %'
            res[m]['res_se'] = res[m]['Mean'] +' ('+res[m]['se']+') %'
        elif m in ['Share user with sale ATT','Share user with sale','True OS','True PCS','True NCS']:
            res[m]["Mean"] = pd.Series(["{0:.2f}".format(val * 100) for val in res[m]["Mean"]], index = res[m].index)
            res[m]["std"] = pd.Series(["{0:.2f}".format(val * 100) for val in res[m]["std"]], index = res[m].index)
            res[m]["se"] = pd.Series(["{0:.2f}".format(val * 100) for val in res[m]["se"]], index = res[m].index)
            
            res[m]['res'] = res[m]['Mean'] +' ('+res[m]['std']+') %'
            res[m]['res_se'] = res[m]['Mean'] +' ('+res[m]['se']+') %'
        elif m in ['NDPC','DPCSO','DPCSN']:
            res[m]['Mean'] = pd.Series(["{0:.3f}".format(val * 1000) for val in res[m]["Mean"]], index = res[m].index)
            res[m]['std'] = pd.Series(["{0:.3f}".format(val * 1000) for val in res[m]["std"]], index = res[m].index)
            res[m]['se'] = pd.Series(["{0:.3f}".format(val * 1000) for val in res[m]["se"]], index = res[m].index)
            
            res[m]['res'] = res[m]['Mean'] +' ('+res[m]['std']+') pm'
            res[m]['res_se'] = res[m]['Mean'] +' ('+res[m]['se']+') pm'
        else :
            res[m]["Mean"] = pd.Series([str(int(val)) for val in res[m]["Mean"]], index = res[m].index)
            res[m]["std"] = pd.Series([str(int(val)) for val in res[m]["std"]], index = res[m].index)
            res[m]["se"] = pd.Series([str(int(val)) for val in res[m]["se"]], index = res[m].index)
            
            res[m]['res'] = res[m]['Mean'] +' ('+res[m]['std']+')'
            res[m]['res_se'] = res[m]['Mean'] +' ('+res[m]['se']+')'
        res_recap[m] =  res[m]['res_se']
    if len(res_recap) == 5:
        res_recap['Agent'] = ['Rand','Click','PVS','PCS','DPCS']
    elif len(res_recap) == 4:
        res_recap['Agent'] = ['Rand','Click','PCS','DPCS']
    elif len(res_recap) == 3:
        res_recap['Agent'] = ['Rand','PCS','DPCS']
    res_recap.columns = ['Agent','CTR','Att Sales','Att CR','Sales','CR',
                         'True CTR','True PCS', 'True OS', 'True NCS',
                         'NDPC','DPCSO','DPCSN']
    # display(res_recap)
    res_AB = res_recap[list(res_recap.columns)[:6]]
    res_true = res_recap[['Agent']+list(res_recap.columns)[6:]]
    if print_res:
        print('-- A/B test --')
        display(res_AB)
        print('-- True metrics --')
        display(res_true)
    # print(res_recap.to_latex(index = False))
    return (res_recap, res_recap.to_latex(index = False), 
            res_AB, res_AB.to_latex(index = False), 
            res_true, res_true.to_latex(index = False))

 