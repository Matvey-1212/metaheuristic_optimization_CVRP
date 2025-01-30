import numpy as np
import random
from CVRP.annealing import annealing_procces, calculate_route_cost, simulated_annealing, sweep_algorithm, sweep_best_initialization
from CVRP.ants import ant_colony_optimization


def hybrid_aco_sa_partial_inverse(vrp_data, config, debug, true_cost):
    
    
    config_sa = config['Annealing']
    T = eval(config_sa['T'])
    Tmin = eval(config_sa['Tmin'])
    alpha = eval(config_sa['alpha'])
    max_iterations = eval(config_sa['max_iterations'])
    max_no_improve = eval(config_sa['max_no_improve'])
    
    best_solution = sweep_algorithm(vrp_data)
    # best_solution = sweep_best_initialization(vrp_data)
    
    centroids = np.array([
        np.mean([vrp_data['node_coord'][point] for point in route], axis=0, dtype=np.float64)
        for route in best_solution
    ])
    
    center = np.array(vrp_data['node_coord'][0])
    vectors = (centroids - center).reshape(-1, 2)
    
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_indices = np.argsort(angles)
    
    grouped_triples = []
    
    num_valid_routes = len(sorted_indices)
    for i in range(num_valid_routes):
        triplet = [
        sorted_indices[i % num_valid_routes],
        sorted_indices[(i + 1) % num_valid_routes],
        sorted_indices[(i + 2) % num_valid_routes],
        sorted_indices[(i + 3) % num_valid_routes],
        sorted_indices[(i + 4) % num_valid_routes],
    ]
        grouped_triples.append(triplet)
    
    improvements = []
    for triplet in grouped_triples:
        sub_solution = [best_solution[j] for j in triplet]
        
        initial_cost = sum(calculate_route_cost(route, vrp_data["edge_weight"]) for route in sub_solution)
        
        improved_sub_solution, improved_cost = annealing_procces(vrp_data, T, Tmin, alpha, max_iterations, max_no_improve, sub_solution, debug, true_cost)
        
        cost_difference = (initial_cost - improved_cost)/initial_cost
        improvements.append((cost_difference, triplet, improved_sub_solution))
    

    improvements.sort(reverse=True, key=lambda x: x[0])    
    applied_changes = set()
    
    for cost_gain, triplet, new_routes in improvements:
        if any(idx in applied_changes for idx in triplet):
            continue
        
        for idx, new_route in zip(triplet, new_routes):
            best_solution[idx] = new_route
            applied_changes.add(idx)
    
    best_cost = sum(calculate_route_cost(route, vrp_data["edge_weight"]) for route in best_solution)
    
    best_solution, best_cost = annealing_procces(vrp_data,
                                                 T, 
                                                 Tmin, 
                                                 alpha, 
                                                 max_iterations, 
                                                 max_no_improve,
                                                 best_solution, debug, true_cost
                                                 )
    
    return best_solution, best_cost
