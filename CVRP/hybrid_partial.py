import numpy as np
import random
from CVRP.annealing import annealing_procces, calculate_route_cost, simulated_annealing, sweep_algorithm, sweep_best_initialization
from CVRP.ants import ant_colony_optimization


def hybrid_aco_sa_partial(vrp_data, config, debug, true_cost):
    # best_solution, best_cost = simulated_annealing(vrp_data, config['Annealing'], debug=debug, true_cost=true_cost)
    aco_best_solution, aco_best_cost = ant_colony_optimization(vrp_data, config['Ants'], debug=debug, true_cost=true_cost)
    print(f'    ACO cost: {aco_best_cost}')
    best_solution = aco_best_solution.copy()
    
    parts_volume = eval(config['Annealing']['parts_volume'])

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
    if num_valid_routes >= 3:
        for i in range(num_valid_routes):
            triplet = []
            for j in range(min(num_valid_routes-1, parts_volume)):
                triplet.append(sorted_indices[(i + j) % num_valid_routes])
            grouped_triples.append(triplet)
    else:
        grouped_triples = [sorted_indices]
    
    improvements = []
    for triplet in grouped_triples:
        sub_solution = [best_solution[j] for j in triplet]
        
        initial_cost = sum(calculate_route_cost(route, vrp_data["edge_weight"]) for route in sub_solution)
        
        improved_sub_solution, improved_cost = annealing_procces(vrp_data, sub_solution, config['Annealing'], debug, true_cost)
        
        cost_difference = initial_cost - improved_cost
        # cost_difference = (initial_cost - improved_cost)/initial_cost
        improvements.append((cost_difference, triplet, improved_sub_solution))
    

    improvements.sort(reverse=True, key=lambda x: x[0])    
    applied_changes = set()
    
    for _, triplet, new_routes in improvements:
        if any(idx in applied_changes for idx in triplet):
            continue
        
        for idx, new_route in zip(triplet, new_routes):
            best_solution[idx] = new_route
            applied_changes.add(idx)
    
    best_cost = sum(calculate_route_cost(route, vrp_data["edge_weight"]) for route in best_solution)
    
    if aco_best_cost < best_cost:
        best_cost = aco_best_cost
    
    return best_solution, best_cost, aco_best_cost