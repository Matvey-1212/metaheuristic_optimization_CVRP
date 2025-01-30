import os
import json
import time

import cv2
import numpy as np

from CVRP.visualizer import visualize_routes_live
from CVRP.utils import *

if __name__ == "__main__":
    cd = os.getcwd()
    run_config = load_run_config('config.ini')
    vrp_data = load_vrp_data(eval(run_config['Paths']['type']), os.path.join(cd, run_config['Paths']['dataset_path']))
    
    optim_algorithm, current_params = get_optim_algorithm(run_config['Analysis']['algorithm'], run_config)
    
    algorithm_type = run_config['Analysis']['algorithm']
    
    res_dict = {}
    global_diff = []
    for vrp_type, vrp_type_item in vrp_data.items():
        local_diff = []
        for vrp_name, vrp_item in vrp_type_item.items():
            
            name = vrp_item['vrp']['name']

            print(f'name: {name}')
            
            solution = vrp_item['sol']['cost']
            routes = vrp_item['sol']['routes']
            routes = [[0] + r + [0] for r in routes]
            print(f'    sol cost: {solution}')
            
            t1 = time.time()
            results = optim_algorithm(vrp_item['vrp'], current_params, debug=eval(run_config['Analysis']['debug']), true_cost=solution)
            t2 = time.time() - t1
            
            aco_best_cost = None
            if len(results) == 3:
                best_solution, best_cost, aco_best_cost = results
            else:
                best_solution, best_cost = results

            img1 = visualize_routes_live(vrp_item['vrp'], best_solution, cost=best_cost)
            img2 = visualize_routes_live(vrp_item['vrp'], routes, true_cost=solution)
            
            cv2.imwrite(os.path.join('result/img/', algorithm_type, vrp_type, name + '_best.jpg'), img1)
            cv2.imwrite(os.path.join('result/img/', algorithm_type, vrp_type, name + '_sol.jpg'), img2)
            
            diff = (best_cost - solution)/solution * 100
            print(f'    best cost: {best_cost}')
            print(f'    diff: {diff}')
            
            res_dict[name] = {}
            res_dict[name]['sol'] = solution
            if aco_best_cost is not None:
                res_dict[name]['aco_best_cost'] = aco_best_cost
            res_dict[name]['best_sol'] = best_cost
            res_dict[name]['diff'] = diff
            res_dict[name]['time'] = t2
            local_diff.append(diff)
            global_diff.append(diff)

        print(f'mean diff per {vrp_type}: {np.mean(local_diff)}')
        print('_'*50)
        
    print(f'mean diff per all: {np.mean(global_diff)}')
            
    path = os.path.join(cd, f'result/res_{algorithm_type}.json')
    with open(path, 'w') as f:
        json.dump(res_dict, f)
            
            