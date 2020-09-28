
from joblib import Parallel, delayed
from recogym.agents.sale_agent import train_agents, train_timeagents
from recogym.envs.utils_sale import format_avg_result_extended, avg_result_extended 
from recogym.evaluate_agent_sale import verify_agents_sale, display_metrics, verify_agents_sale_extended
from copy import deepcopy
import pickle as pkl

# function used for parallelisation, where i is the nb of the A/B test
def run_AB_test(i,name_ext,env,num_users,num_users_AB,agents,save_agents,name_logging,feature_name,data_repo):
    print("-------------- A/B test n°"+str(i)+'--------------')
    name_extension = name_ext+'_nb'+str(i)
    res=verify_agents_sale_extended(
        env,
        number_of_users=num_users_AB,
        agents={
            **agents, 
            **save_agents},
        name = name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension,
        seed = 0,
        same_env = False,
        repo = data_repo
    )

    r = deepcopy(res)
    for k in r.keys():
        if k not in ['CTR', 'Tot sales ATT', 'Share user with sale ATT', 'Tot sales', 'Share user with sale',
                     'True CTR','True PCS','True OS','True NCS']:
            r[k] = None
    r['name'] = name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension
    return r



def run_pres_noweight(logs,name_agent,feature_name,features,num_users,num_users_AB,num_AB_tests, env, agents,data_repo,num_cores):
    name_extension = 'pres'
    name_logging = name_agent
    info, save_agents = train_agents(name_logging,logs,feature_name,features, num_users=num_users, 
                                         kronecker_features=True,repo = data_repo)
    def run_func(i):
        return run_AB_test(i,name_ext=name_extension,env=env,num_users=num_users,num_users_AB=num_users_AB,
                                    agents=agents,save_agents=save_agents,name_logging=name_logging,feature_name=feature_name,
                                    data_repo=data_repo)
    r_list = Parallel(n_jobs=int(num_cores), verbose=50)(delayed(run_func)(i) for i in range(num_AB_tests))
    
    res_dict = {r_list[i]['name']:r_list[i] for i in range(len(r_list))}
    res_avg = avg_result_extended(res_dict)
    (res_recap, res_recap_latex, 
     res_AB, res_AB_latex, 
     res_true, res_true_latex) = format_avg_result_extended(res_avg) #get dataframe & corresponding latex table
    pkl.dump(res_dict, open(data_repo+"clean/res_dict_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".pkl",'wb'))
    pkl.dump(res_avg, open(data_repo+"clean/res_avg_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".pkl",'wb'))
    res_recap.to_csv(data_repo+"clean/res_recap_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".csv",index = False)
    res_true.to_csv(data_repo+"clean/res_true_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".csv",index = False)
    return (res_recap, res_recap_latex, res_AB, res_AB_latex, res_true, res_true_latex)

def run_pres_weight(logs,name_agent,feature_name,features,num_users,num_users_AB,num_AB_tests, env, agents,data_repo,num_cores):
    name_extension = 'presweights'
    name_logging = name_agent
    info, save_agents = train_agents(name_logging,logs,feature_name,features, num_users=num_users, 
                                         kronecker_features=True,weights = True, repo = data_repo)
    def run_func(i):
        return run_AB_test(i,name_ext=name_extension,env=env,num_users=num_users,num_users_AB=num_users_AB,
                                    agents=agents,save_agents=save_agents,name_logging=name_logging,feature_name=feature_name,
                                    data_repo=data_repo)
    r_list = Parallel(n_jobs=int(num_cores), verbose=50)(delayed(run_func)(i) for i in range(num_AB_tests))
    
    res_dict = {r_list[i]['name']:r_list[i] for i in range(len(r_list))}
    res_avg = avg_result_extended(res_dict)
    (res_recap, res_recap_latex, 
     res_AB, res_AB_latex, 
     res_true, res_true_latex) = format_avg_result_extended(res_avg) #get dataframe & corresponding latex table
    pkl.dump(res_dict, open(data_repo+"clean/res_dict_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".pkl",'wb'))
    pkl.dump(res_avg, open(data_repo+"clean/res_avg_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".pkl",'wb'))
    res_recap.to_csv(data_repo+"clean/res_recap_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".csv",index = False)
    res_true.to_csv(data_repo+"clean/res_true_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".csv",index = False)
    return (res_recap, res_recap_latex, res_AB, res_AB_latex, res_true, res_true_latex)


def run_prop_noweight(logs,name_agent,feature_name,features,num_users,num_users_AB,num_AB_tests, env, agents,data_repo,num_cores):
    name_extension = 'prop'
    name_logging = name_agent
    info, save_agents = train_timeagents(name_logging,logs,feature_name,features, num_users=num_users, 
                                         kronecker_features=True,linear_reg=True, repo = data_repo)
    def run_func(i):
        return run_AB_test(i,name_ext=name_extension,env=env,num_users=num_users,num_users_AB=num_users_AB,
                                    agents=agents,save_agents=save_agents,name_logging=name_logging,feature_name=feature_name,
                                    data_repo=data_repo)
    r_list = Parallel(n_jobs=int(num_cores), verbose=50)(delayed(run_func)(i) for i in range(num_AB_tests))
    
    res_dict = {r_list[i]['name']:r_list[i] for i in range(len(r_list))}
    res_avg = avg_result_extended(res_dict)
    (res_recap, res_recap_latex, 
     res_AB, res_AB_latex, 
     res_true, res_true_latex) = format_avg_result_extended(res_avg) #get dataframe & corresponding latex table
    pkl.dump(res_dict, open(data_repo+"clean/res_dict_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".pkl",'wb'))
    pkl.dump(res_avg, open(data_repo+"clean/res_avg_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".pkl",'wb'))
    res_recap.to_csv(data_repo+"clean/res_recap_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".csv",index = False)
    res_true.to_csv(data_repo+"clean/res_true_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".csv",index = False)
    return (res_recap, res_recap_latex, res_AB, res_AB_latex, res_true, res_true_latex)

def run_prop_weight(logs,name_agent,feature_name,features,num_users,num_users_AB,num_AB_tests, env, agents,data_repo,num_cores):
    name_extension = 'propweights'
    name_logging = name_agent
    info, save_agents = train_timeagents(name_logging,logs,feature_name,features, num_users=num_users, 
                                         kronecker_features=True,linear_reg=True, weights = True, repo = data_repo)
    def run_func(i):
        return run_AB_test(i,name_ext=name_extension,env=env,num_users=num_users,num_users_AB=num_users_AB,
                                    agents=agents,save_agents=save_agents,name_logging=name_logging,feature_name=feature_name,
                                    data_repo=data_repo)
    r_list = Parallel(n_jobs=int(num_cores), verbose=50)(delayed(run_func)(i) for i in range(num_AB_tests))
    
    res_dict = {r_list[i]['name']:r_list[i] for i in range(len(r_list))}
    res_avg = avg_result_extended(res_dict)
    (res_recap, res_recap_latex, 
     res_AB, res_AB_latex, 
     res_true, res_true_latex) = format_avg_result_extended(res_avg) #get dataframe & corresponding latex table
    pkl.dump(res_dict, open(data_repo+"clean/res_dict_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".pkl",'wb'))
    pkl.dump(res_avg, open(data_repo+"clean/res_avg_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".pkl",'wb'))
    res_recap.to_csv(data_repo+"clean/res_recap_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".csv",index = False)
    res_true.to_csv(data_repo+"clean/res_true_"+name_logging+str(num_users)+"_"+str(num_users_AB)+"_"+feature_name+name_extension+".csv",index = False)
    return (res_recap, res_recap_latex, res_AB, res_AB_latex, res_true, res_true_latex)


