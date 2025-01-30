
import numpy as np
import random
from CVRP.annealing import annealing_procces
from CVRP.ants import ant_colony_optimization

def hybrid_aco_sa(vrp_data, config, debug, true_cost):
    best_solution, best_cost = ant_colony_optimization(vrp_data, config['Ants'], debug=debug, true_cost=true_cost)

    # flattened_solution = [node for route in best_solution for node in route[1:-1]]
    config = config['Annealing']
    T = eval(config['T'])
    Tmin = eval(config['Tmin'])
    alpha = eval(config['alpha'])
    max_iterations = eval(config['max_iterations'])
    max_no_improve = eval(config['max_no_improve'])
    
    improved_solution, improved_cost = annealing_procces(vrp_data,
                                                 T, 
                                                 Tmin, 
                                                 alpha, 
                                                 max_iterations, 
                                                 max_no_improve, 
                                                 best_solution, debug, true_cost)

    if improved_cost < best_cost:
        best_solution = improved_solution
        best_cost = improved_cost

    return best_solution, best_cost