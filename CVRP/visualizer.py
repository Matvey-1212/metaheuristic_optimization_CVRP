import cv2
import numpy as np


def transform(point, node_coords, img_size):
        x, y = point
        min_x, min_y = np.min(node_coords, axis=0)
        max_x, max_y = np.max(node_coords, axis=0)
        scale_x = (img_size - 40) / (max_x - min_x)
        scale_y = (img_size - 40) / (max_y - min_y)
        x = int((x - min_x) * scale_x + 20)
        y = int((y - min_y) * scale_y + 20)
        return x, img_size - y

def visualize_routes_live(vrp_data, routes, iteration=None, cost=None, true_cost=None, T=None):
    node_coords = vrp_data["node_coord"]
    depot = vrp_data["depot"][0]

    img_size = 600
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  
    
    for i, (x, y) in enumerate(node_coords):
        px, py = transform((x, y), node_coords, img_size)
        color = (0, 0, 255) if i == depot else (0, 255, 0)
        cv2.circle(img, (px, py), 5, color, -1)
        cv2.putText(img, str(i), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    colors = [(255, 0, 0), (0, 255, 255), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
    for route_idx, route in enumerate(routes):
        color = colors[route_idx % len(colors)]
        for j in range(len(route) - 1):
            p1 = transform(node_coords[route[j]], node_coords, img_size)
            p2 = transform(node_coords[route[j + 1]], node_coords, img_size)
            cv2.line(img, p1, p2, color, 2)

    start = [20, 30]
    if iteration is not None:
        cv2.putText(img, f"Iteration {iteration}", start, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        start[1] += 30
    if cost is not None:
        cv2.putText(img, f"cur cost {int(cost)}", start, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        start[1] += 30
    if true_cost is not None:
        cv2.putText(img, f"true cost {true_cost}", start, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        start[1] += 30
    if T is not None:
        cv2.putText(img, f"T {T:.4f}", start, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        start[1] += 30

    return img
