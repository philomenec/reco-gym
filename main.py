import gym
from copy import deepcopy
import pandas as pd
import pickle as pkl
pd.options.mode.chained_assignment = None 
from recogym.run_agent_sale import (run_pres_noweight,run_pres_weight,run_prop_noweight,run_prop_weight)
from recogym import env_1_sale_args, Configuration
from joblib import Parallel, delayed
import os




# #### Settings --- to change!!


# Number of cores
num_cores = 40
# Number of users for the training
num_users = 5000
# Number of users for the A/B test
num_users_AB = 5000
# Number of A/B tests
num_AB_tests_list = [50,25,25]

# # tests
# # Number of cores
# num_cores = 4
# # Number of users for the training
# num_users = 5
# # Number of users for the A/B test
# num_users_AB = 2
# # Number of A/B tests
# num_AB_tests_list =[1,2,2]

#### Configuration !!
config_dict = {'cl_nomem':{'click':True,'memory':False},
               'cl_mem':{'click':True,'memory':True},
                'nocl_mem':{'click':False,'memory':True}}
names_runs = list(config_dict.keys())



################### Initialise environment
env_1_sale_args['random_seed'] = 0
env_1_sale_args['num_products'] = 10
env_1_sale_args['number_of_flips'] = 10 
num_products = env_1_sale_args['num_products']
print('Number of products =',num_products)
print('Number of flips =',env_1_sale_args['number_of_flips'])
nb_flips = env_1_sale_args['number_of_flips']
env_1_sale_args['mu_sale'] = False 
# env_1_sale_args['kappa'] = 0.5
print('Value of kappa =',env_1_sale_args['kappa'])
env_1_sale_args['num_users'] = num_users
env_1_sale_args['num_users_AB'] = num_users_AB


# Initialize the gym 
env = gym.make('reco-gym-sale-v1')
env.init_gym(env_1_sale_args)

# Repository to save pickles
data_repo = os.getcwd()+'/data_conversion/'


# User features
from recogym.agents.sale_agent import ShareViewsFeatureProvider
v_share_feature = ShareViewsFeatureProvider(env.config)
features = {'v_share':v_share_feature}
feature_name = 'v_share'
feature = features[feature_name]



############## Random agent
agents={}
name_agent = 'rand'+str(nb_flips)
from recogym.agents import RandomAgent, random_args
random_agent = RandomAgent(Configuration(random_args))
agents[name_agent] = random_agent


#### Logs
try:
    print("--- Load logs ---")
    logs = {name_agent:pd.read_csv(data_repo + 'data' + str(num_users) + name_agent + '.csv')}
except:
    print("--- Generate logs ---")
    logs = {name_agent:deepcopy(env).generate_logs(num_users)}
    logs[name_agent].to_csv(data_repo + 'data' + str(num_users) + name_agent + '.csv',index = False)


for i in range(len(config_dict)):
    print(f'------------------- Config nb {i}')
    name_run = names_runs[i]
    config = config_dict[name_run]
    num_AB_tests = num_AB_tests_list[i]
    # already run for the first configuration
    if i>0:
        # Equal sample weights
        run_pres_noweight(logs,name_agent,feature_name,features,num_users,num_users_AB,
                          num_AB_tests, env, agents,data_repo,num_cores,name_run, config,save=False)
        run_prop_noweight(logs,name_agent,feature_name,features,num_users,num_users_AB,
                          num_AB_tests, env, agents,data_repo,num_cores,name_run,config,save=False)
    # Sample Weights
    run_pres_weight(logs,name_agent,feature_name,features,num_users,num_users_AB,
                    num_AB_tests, env, agents,data_repo,num_cores,name_run,config,save=False)
    run_prop_weight(logs,name_agent,feature_name,features,num_users,num_users_AB,
                    num_AB_tests, env, agents,data_repo,num_cores,name_run,config,save=False)
    
