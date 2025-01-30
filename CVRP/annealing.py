import random
import math
import copy
import numpy as np
import cv2
from tqdm import tqdm
from CVRP.visualizer import visualize_routes_live
from sklearn.cluster import KMeans

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def calculate_route_cost(route, distance_matrix):
    cost = 0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i], route[i + 1]]
    return cost

def calculate_total_cost(routes, distance_matrix):
    return sum(calculate_route_cost(route, distance_matrix) for route in routes)

def greedy_initial_solution(vrp_data):
    customers = list(range(1, vrp_data["dimension"]))
    depot = vrp_data["depot"][0]
    capacity = vrp_data["capacity"]
    demands = vrp_data["demand"]
    
    routes = []
    while customers:
        route = [depot]
        load = 0
        for customer in customers[:]: 
            if load + demands[customer] <= capacity:
                route.append(customer)
                load += demands[customer]
                customers.remove(customer)
        route.append(depot)
        routes.append(route)

    return routes

def sweep_algorithm(vrp_data, start_index=1):
    depot = vrp_data["depot"][0]
    demands = vrp_data["demand"]
    capacity = vrp_data["capacity"]
    node_coords = vrp_data["node_coord"]

    angles = []
    for i in range(1, vrp_data["dimension"]):
        dx = node_coords[i][0] - node_coords[depot][0]
        dy = node_coords[i][1] - node_coords[depot][1]
        angle = math.atan2(dy, dx)
        angles.append((angle, i))

    angles.sort()

    angles = angles[start_index:] + angles[:start_index]
    routes = []
    current_route = [depot]
    current_load = 0

    for _, customer in angles:
        demand = demands[customer]
        if current_load + demand > capacity:
            current_route.append(depot)
            routes.append(current_route)
            current_route = [depot]
            current_load = 0
        current_route.append(customer)
        current_load += demand

    current_route.append(depot)
    routes.append(current_route)

    return routes

def sweep_best_initialization(vrp_data):
    distance_matrix = vrp_data["edge_weight"]
    best_routes = None
    best_cost = float('inf')

    for start_index in range(vrp_data["dimension"] - 1):
        routes = sweep_algorithm(vrp_data, start_index)
        cost = calculate_total_cost(routes, distance_matrix)

        if cost < best_cost:
            best_routes = routes
            best_cost = cost

    return best_routes

def is_valid_solution(routes, vrp_data):
    depot = vrp_data["depot"][0]

    for route in routes:
        if route[0] != depot or route[-1] != depot:
            return False

        total_demand = sum(vrp_data["demand"][node] for node in route if node != depot)
        if total_demand > vrp_data["capacity"]:
            return False

    return True

def two_opt(route):
    if len(route) <= 3:
        return route
    i, j = sorted(random.sample(range(1, len(route) - 1), 2))
    route[i:j + 1] = reversed(route[i:j + 1])
    return route

def three_opt(route):
    if len(route) <= 4:
        return route
    i, j, k = sorted(random.sample(range(1, len(route) - 1), 3))
    return route[:i] + route[j:k] + route[i:j] + route[k:]

def or_opt(routes):
    if len(routes) < 2:
        return routes
    route1, route2 = random.sample(routes, 2)
    if len(route1) > 4 and len(route2) > 4:
        idx1 = random.randint(1, len(route1) - 3)
        idx2 = random.randint(1, len(route2) - 3)
        segment = route1[idx1:idx1 + 2]
        del route1[idx1:idx1 + 2]
        route2[idx2:idx2] = segment
    return routes

def swap_between_routes(routes):
    if len(routes) < 2:
        return routes
    route1, route2 = random.sample(routes, 2)
    if len(route1) > 2 and len(route2) > 2:
        idx1 = random.randint(1, len(route1) - 2)
        idx2 = random.randint(1, len(route2) - 2)
        route1[idx1], route2[idx2] = route2[idx2], route1[idx1]
    return routes

def annealing_procces(vrp_data, current_solution, config, debug=False, true_cost=None):
    T = eval(config['T'])
    Tmin = eval(config['Tmin'])
    alpha = eval(config['alpha'])
    max_iterations = eval(config['max_iterations'])
    max_no_improve = eval(config['max_no_improve'])
    use_multi_alpha = eval(config['use_multi_alpha'])
    multi_alpha = eval(config['multi_alpha'])
    T_tresholds = eval(config['T_tresholds'])
    
    current_cost = calculate_total_cost(current_solution, vrp_data["edge_weight"])
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost
    iteration = 0
    no_improve_counter = 0
    # num_iterations = math.ceil(math.log(Tmin / T, alpha))
    num_iterations = 0
    
    Tt = T
    while Tt > Tmin:
        if not use_multi_alpha:
            Tt *= alpha
        else:
            if Tt > T_tresholds[0]:
                Tt *= multi_alpha[0]  
            elif T_tresholds[1] < Tt <= T_tresholds[0]:
                Tt *= multi_alpha[1]  
            else:
                Tt *= multi_alpha[2]  
        num_iterations += 1
    
    with tqdm(desc="Progress", leave=False, total=num_iterations) as pbar:
        while T > Tmin:
            no_improve_counter = 0
            for _ in range(max_iterations):
                new_solution = copy.deepcopy(current_solution)
                route_index = random.randint(0, len(new_solution) - 1)

                improvement_rate = max_no_improve / max_iterations

                if improvement_rate > 0.5:
                    p_two_opt, p_three_opt, p_or_opt, p_swap = 0.3, 0.4, 0.2, 0.1
                else:  
                    p_two_opt, p_three_opt, p_or_opt, p_swap = 0.6, 0.2, 0.1, 0.1

                r = random.uniform(0, 1)

                if r < p_two_opt:
                    new_solution[route_index] = two_opt(new_solution[route_index])
                elif r < p_two_opt + p_three_opt:
                    new_solution[route_index] = three_opt(new_solution[route_index])
                elif r < p_two_opt + p_three_opt + p_or_opt:
                    new_solution = or_opt(new_solution)
                else:
                    new_solution = swap_between_routes(new_solution)
                #     new_solution = or_opt_nearest(new_solution, vrp_data["edge_weight"])
                # else:
                #     new_solution = swap_between_routes_nearest(new_solution, vrp_data["edge_weight"])

                if not is_valid_solution(new_solution, vrp_data):
                    continue

                new_cost = calculate_total_cost(new_solution, vrp_data["edge_weight"])
                delta_E = new_cost - current_cost

                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    current_solution = new_solution
                    current_cost = new_cost

                    if current_cost < best_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_cost = current_cost
                        no_improve_counter = 0 
                    else:
                        no_improve_counter += 1
            
                    
                if no_improve_counter >= max_no_improve:
                    # pbar.close()
                    # return best_solution, best_cost
                    break
            pbar.set_description(f"SA-Cost: {best_cost:.2f}")
            
            if debug and iteration % 5 == 0:
                img = visualize_routes_live(vrp_data, best_solution, iteration, cost=best_cost, true_cost=true_cost, T=T)
                cv2.imshow("VRP Optimization", img)
                key = cv2.waitKey(1) 
                if key == 27: 
                    cv2.destroyAllWindows()
                    exit(0)
            
            
            if not use_multi_alpha:
                T *= alpha
            else:
                if T > T_tresholds[0]:
                    T *= multi_alpha[0]  
                elif T_tresholds[1] < T <= T_tresholds[0]:
                    T *= multi_alpha[1]  
                else:
                    T *= multi_alpha[2]  

            iteration += 1
            pbar.update(1) 
    return best_solution, best_cost

def simulated_annealing(vrp_data, config, debug=False, true_cost=None):
    # current_solution = greedy_initial_solution(vrp_data)
    current_solution = sweep_algorithm(vrp_data)
    # current_solution = sweep_best_initialization(vrp_data)
    # current_solution = cluster_customers(vrp_data)
    
    best_solution, best_cost = annealing_procces(vrp_data, current_solution, config, debug=debug, true_cost=true_cost)

    return best_solution, best_cost
