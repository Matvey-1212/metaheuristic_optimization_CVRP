import os

import configparser
import vrplib
from CVRP.annealing import simulated_annealing
from CVRP.ants import ant_colony_optimization
from CVRP.hybrid import hybrid_aco_sa
from CVRP.hybrid_partial import hybrid_aco_sa_partial
from CVRP.hybrid_partial_inverse import hybrid_aco_sa_partial_inverse

def load_run_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config    

def load_vrp_data(vrp_type_list, dataset_path):
    if vrp_type_list is None or not isinstance(vrp_type_list, list):
        return None
    
    all_data = {}
    
    for vrp_type in vrp_type_list:
        folder = os.path.join(dataset_path, vrp_type)
        
        all_data[vrp_type] = {}
        
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            
            if not file_name.endswith('.vrp'):
                continue
            
            all_data[vrp_type][file_name.replace('.vrp', '')] = {}
            instance = vrplib.read_instance(file_path)
            all_data[vrp_type][file_name.replace('.vrp', '')]['vrp'] = instance
            
            if os.path.isfile(file_path.replace('.vrp', '.sol')):
                solution = vrplib.read_solution(file_path.replace('.vrp', '.sol'))
                # with open(file_path, 'r') as f:
                #     best_score = f.readlines()
                # print(best_score)
                all_data[vrp_type][file_name.replace('.vrp', '')]['sol'] = solution
                
    return all_data
            
                
def get_optim_algorithm(name, config):
    if name == 'annealing':
        return simulated_annealing, config['Annealing']
    elif name == 'ants':
        return ant_colony_optimization, config['Ants']
    elif name == 'hybrid':
        return hybrid_aco_sa, {'Ants':config['Ants'], 'Annealing':config['Annealing']}
    elif name == 'hybrid_partial':
        return hybrid_aco_sa_partial, {'Ants':config['Ants'], 'Annealing':config['Annealing']}
    elif name == 'hybrid_partial_inverse':
        return hybrid_aco_sa_partial_inverse, {'Ants':config['Ants'], 'Annealing':config['Annealing']}