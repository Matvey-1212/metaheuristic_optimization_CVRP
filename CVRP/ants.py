import numpy as np
import cv2
import random
from tqdm import tqdm
from CVRP.visualizer import visualize_routes_live
from CVRP.annealing import two_opt, three_opt, or_opt, swap_between_routes

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def initialize_pheromones(num_nodes, initial_pheromone):
    # if initial_pheromone < 0:
    initial_pheromone = initial_pheromone #/ num_nodes
    return np.ones((num_nodes, num_nodes)) * initial_pheromone

def initialize_pheromones_by_dist(distance_matrix, initial_pheromone, epsilon=1e-6):
    avg_distance = np.mean(distance_matrix)
    return (avg_distance / (distance_matrix + 1e-6)) * initial_pheromone
    # return (distance_matrix / (avg_distance + epsilon)) * initial_pheromone

def calculate_probability(current_node, unvisited, pheromones, distance_matrix, alpha, beta):
    probabilities = []
    for node in unvisited:
        probabilities.append(pheromones[current_node][node] ** alpha * (1 / (distance_matrix[current_node][node] + 1e-6)) ** beta)

    probabilities = np.array(probabilities)
    total = probabilities.sum()
    if total == 0 or np.isnan(total):
        return np.ones(len(unvisited)) / len(unvisited)
    
    probabilities /= total
    return probabilities

def construct_ant_solution(pheromones, distance_matrix, demand, vehicle_capacity, depot, alpha, beta):
    num_nodes = len(distance_matrix)
    unvisited = set(range(num_nodes)) - {depot}
    routes = []

    while unvisited:
        route = [depot]
        load = 0
        current_node = depot

        while unvisited:
            probabilities = calculate_probability(current_node, list(unvisited), pheromones, distance_matrix, alpha, beta)
            if len(probabilities) == 0:
                break

            next_node = random.choices(list(unvisited), weights=probabilities, k=1)[0]

            if load + demand[next_node] > vehicle_capacity:
                break

            route.append(next_node)
            load += demand[next_node]
            unvisited.remove(next_node)
            current_node = next_node

        route.append(depot)
        routes.append(route)

    return routes

def calculate_cost(solution, distance_matrix):
    total_cost = 0
    for routes in solution:
        for route in routes:
            cost = 0
            for i in range(len(route) - 1):
                cost += distance_matrix[route[i]][route[i + 1]]
            total_cost += cost
    return total_cost

def update_pheromones(pheromones, solutions, costs, Q, evaporation):
    pheromones *= (1 - evaporation)
    for routes, cost in zip(solutions, costs):
        for route in routes:
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                pheromones[from_node][to_node] += Q / (cost + 1e-6)
                pheromones[to_node][from_node] = pheromones[from_node][to_node] #Q / (cost + 1e-6)

def ant_colony_optimization(vrp_data, config, debug=False, true_cost=None):
    num_ants = eval(config['num_ants'])
    num_iterations = eval(config['max_iterations'])
    max_num_iterations_with_no_updates = eval(config['max_num_iterations_with_no_updates'])
    alpha = eval(config['alpha'])
    beta = eval(config['beta'])
    evaporation = eval(config['evaporation'])
    Q = eval(config['Q'])
    initial_pheromone = eval(config['initial_pheromone'])
    init_by_dist = eval(config['init_by_dist'])

    depot = int(vrp_data['depot'][0])
    demand = vrp_data['demand']
    vehicle_capacity = vrp_data['capacity']
    distance_matrix = vrp_data['edge_weight']
    
    
    num_nodes = len(distance_matrix)

    if init_by_dist:
        pheromones = initialize_pheromones_by_dist(distance_matrix, initial_pheromone)
    else:
        pheromones = initialize_pheromones(num_nodes, initial_pheromone)

    best_solution = None
    best_cost = float('inf')
    iter_without_update = 0

    with tqdm(desc="Progress", leave=False, total=num_iterations) as pbar:
        for iteration in range(num_iterations):
            solutions = [construct_ant_solution(pheromones, distance_matrix, demand, vehicle_capacity, depot, alpha, beta) for _ in range(num_ants)]
            
            costs = [calculate_cost([solution], distance_matrix) for solution in solutions]
            
            top_k = max(1, len(solutions) // 2)
            best_solutions_i = np.argsort(costs)[:top_k]
            new_cost = []
            new_sol = []
            for i in best_solutions_i:
                new_cost.append(costs[i])
                new_sol.append(solutions[i])

            update_pheromones(pheromones, new_sol, new_cost, Q, evaporation)

            min_cost = min(costs)
            if min_cost < best_cost:
                best_cost = min_cost
                best_solution = solutions[costs.index(min_cost)]
                iter_without_update = 0
            else:
                iter_without_update+=1
            if iter_without_update > max_num_iterations_with_no_updates:
                break

            if debug and iteration % 5 == 0:
                img = visualize_routes_live(vrp_data, best_solution, iteration, best_cost, true_cost)
                cv2.imshow("VRP Optimization", img)
                key = cv2.waitKey(1) 
                if key == 27: 
                    cv2.destroyAllWindows()
                    exit(0)
            
            pbar.set_description(f"ACO-Cost: {best_cost:.2f}")
            pbar.update(1)

    return best_solution, best_cost
